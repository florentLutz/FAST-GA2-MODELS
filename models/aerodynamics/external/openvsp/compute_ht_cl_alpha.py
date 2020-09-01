"""
    Estimation of HTP lift coefficient using OPENVSP
"""
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import multiprocessing
import os
import os.path as pth
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory

import math
import numpy as np
from fastoad.utils.physics import Atmosphere
from fastoad.utils.resource_management.copy import copy_resource
from importlib_resources import path
from openmdao.components.external_code_comp import ExternalCodeComp
from openmdao.utils.file_wrap import InputFileGenerator

from . import resources
from . import openvsp351

OPTION_OPENVSP_EXE_PATH = "openvsp_exe_path"

_INPUT_SCRIPT_FILE_NAME = "polar_session_2.vspscript"
_INPUT_AERO_FILE_NAME = "polar_session.vspaero"
_INPUT_AOAList = [2.0, 4.0] # ???: why mid point is 3° ?
_AIRFOIL_0_FILE_NAME = "naca23012.af"
_AIRFOIL_1_FILE_NAME = "naca23012.af"
_AIRFOIL_2_FILE_NAME = "naca23012.af"
VSPSCRIPT_EXE_NAME = "vspscript.exe"
VSPAERO_EXE_NAME = "vspaero.exe"

class ComputeHTPCLALPHAopenvsp(ExternalCodeComp):

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)
        self.options.declare(OPTION_OPENVSP_EXE_PATH, default="", types=str, allow_none=True)
        
    def setup(self):
        
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_0", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("ata:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:height", val=np.nan, units="m")
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_alpha")
        else:
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:mission:sizing:cruise:altitude", val=np.nan, units='ft')
            self.add_output("data:aerodynamics:horizontal_tail:cruise:CL_alpha")
        
        self.declare_partials("*", "*", method="fd")        
    
    def compute(self, inputs, outputs):
        
        # Get inputs (and calculate missing ones)
        x0_wing = inputs["data:geometry:wing:MAC:leading_edge:x:local"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        y1_wing = width_max/2.0
        y2_wing = inputs["data:geometry:wing:root:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        sweep_0_wing = inputs["data:geometry:wing:sweep_0"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        Sref_wing = inputs['data:geometry:wing:area']
        span_wing = inputs['data:geometry:wing:span']
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        sweep_25_htp = inputs["data:geometry:horizontal_tail:sweep_25"]
        span_htp = inputs["data:geometry:horizontal_tail:span"]/2.0
        root_chord_htp = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord_htp = inputs["data:geometry:horizontal_tail:tip:chord"]
        lp_htp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"] 
        l0_htp = inputs["data:geometry:horizontal_tail:MAC:length"] 
        x0_htp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:local"]
        height_htp = inputs["data:geometry:horizontal_tail:height"]
        if self.options["low_speed_aero"]:
            altitude = 0.0
            atm = Atmosphere(altitude)
            mach = inputs["data:aerodynamics:low_speed:mach"]
        else:
            altitude = inputs["data:mission:sizing:cruise:altitude"]
            atm = Atmosphere(altitude)
            mach = inputs["data:aerodynamics:cruise:mach"]
        
        # Compute remaining inputs
        x_wing = fa_length-x0_wing-0.25*l0_wing
        z_wing = -(height_max - 0.12*l2_wing)*0.5
        span2_wing = y4_wing - y2_wing
        distance_htp = fa_length + lp_htp - 0.25 * l0_htp - x0_htp
        AOAList = str(_INPUT_AOAList)
        AOAList = AOAList[1:len(AOAList)-1]
        speed_of_sound = atm.speed_of_sound
        viscosity = atm.kinematic_viscosity
        rho = atm.density
        V_inf = max(speed_of_sound * mach, 0.01) # avoid V=0 m/s crashes
        reynolds = V_inf * l0_wing / viscosity
        
        # OPENVSP-SCRIPT: Geometry generation ######################################################
        
        # I/O files --------------------------------------------------------------------------------
        tmp_directory = self._create_tmp_directory()
        if self.options[OPTION_OPENVSP_EXE_PATH]:
            self.stdin1 = pth.join(self.options[OPTION_OPENVSP_EXE_PATH], pth.splitext(_INPUT_AERO_FILE_NAME)[0], '.vspscript')
            self.stdin2 = pth.abspath(pth.join(self.options[OPTION_OPENVSP_EXE_PATH], _AIRFOIL_0_FILE_NAME))
            self.stdin3 = pth.abspath(pth.join(self.options[OPTION_OPENVSP_EXE_PATH], _AIRFOIL_1_FILE_NAME))
            self.stdin4 = pth.abspath(pth.join(self.options[OPTION_OPENVSP_EXE_PATH], _AIRFOIL_2_FILE_NAME)) 
        else:
            self.stdin1 = pth.join(tmp_directory.name, pth.splitext(_INPUT_AERO_FILE_NAME)[0], '.vspscript')
            self.stdin2 = pth.abspath(pth.join(tmp_directory.name, _AIRFOIL_0_FILE_NAME))
            self.stdin3 = pth.abspath(pth.join(tmp_directory.name, _AIRFOIL_1_FILE_NAME))
            self.stdin4 = pth.abspath(pth.join(tmp_directory.name, _AIRFOIL_2_FILE_NAME))  
        
        # Pre-processing (populating temp directory) -----------------------------------------------
        if self.options[OPTION_OPENVSP_EXE_PATH]:
            # if a path for openvsp has been provided, simply use it
            self.options["command"] = [pth.join(self.options[OPTION_OPENVSP_EXE_PATH], VSPSCRIPT_EXE_NAME) + ' -script ' + self.stdin1]
        else:
            # otherwise, copy the embedded resource in tmp dir
            copy_resource(openvsp351, VSPSCRIPT_EXE_NAME, VSPAERO_EXE_NAME, tmp_directory.name)
            copy_resource(_AIRFOIL_0_FILE_NAME, _AIRFOIL_1_FILE_NAME, _AIRFOIL_2_FILE_NAME, tmp_directory.name)
            self.options["command"] = [pth.join(tmp_directory.name, VSPSCRIPT_EXE_NAME) + ' -script ' + self.stdin1]
        
        # standard SCRIPT input file ----------------------------------------------------------------
        tmp_result_file_path = pth.join(pth.splitext(self.stdin1)[0], '.csv')
        parser = InputFileGenerator()
        with path(resources, _INPUT_SCRIPT_FILE_NAME) as input_template_path:
            parser.set_template_file(input_template_path)
            parser.set_generated_file(self.stdin1)
            parser.mark_anchor("x_wing")
            parser.transfer_var(float(x_wing) + ');', 0, 57)
            parser.mark_anchor("z_wing")
            parser.transfer_var(float(z_wing) + ');', 0, 57)
            parser.mark_anchor("y1_wing")
            parser.transfer_var(float(y1_wing) + ');', 0, 57)
            parser.mark_anchor("l2_wing")
            parser.transfer_var(float(l2_wing) + ');', 0, 57)
            parser.reset_anchor()
            parser.mark_anchor("span2_wing")
            parser.transfer_var(float(span2_wing) + ');', 0, 57)
            parser.mark_anchor("l4_wing")
            parser.transfer_var(float(l4_wing) + ');', 0, 57)
            parser.mark_anchor("sweep_0_wing")
            parser.transfer_var(float(sweep_0_wing) + ');', 0, 57)
            parser.mark_anchor("airfoil_0_file")
            parser.transfer_var(self.stdin2 + ');', 0, 27)
            parser.mark_anchor("airfoil_1_file")
            parser.transfer_var(self.stdin3 + ');', 0, 27)
            parser.mark_anchor("airfoil_2_file")
            parser.transfer_var(self.stdin4 + ');', 0, 27)
            parser.mark_anchor("distance_htp")
            parser.transfer_var(float(distance_htp) + ');', 0, 57)
            parser.mark_anchor("height_htp")
            parser.transfer_var(float(height_htp) + ');', 0, 57)
            parser.mark_anchor("span_htp")
            parser.transfer_var(float(span_htp) + ');', 0, 57)
            parser.mark_anchor("root_chord_htp")
            parser.transfer_var(float(root_chord_htp) + ');', 0, 57)
            parser.mark_anchor("tip_chord_htp")
            parser.transfer_var(float(tip_chord_htp) + ');', 0, 57)
            parser.mark_anchor("sweep_25_htp")
            parser.transfer_var(float(sweep_25_htp) + ');', 0, 57)
            parser.generate()
            
        # Run SCRIPT --------------------------------------------------------------------------------
        self.options["external_input_files"] = [self.stdin1, self.stdin2, self.stdin3, self.stdin4]
        self.options["external_output_files"] = [tmp_result_file_path]
        super().compute(inputs, outputs)
 
        # OPENVSP-AERO: aero calculation ############################################################
       
        # I/O files --------------------------------------------------------------------------------
        self.stdin1 = tmp_result_file_path
        if self.options[OPTION_OPENVSP_EXE_PATH]:
            self.stdin2 = pth.join(self.options[OPTION_OPENVSP_EXE_PATH], _INPUT_AERO_FILE_NAME)
        else:
            self.stdin2 = pth.join(tmp_directory.name, _INPUT_AERO_FILE_NAME)
        
        # Pre-processing (populating temp directory) -----------------------------------------------
        if self.options[OPTION_OPENVSP_EXE_PATH]:
            self.options["command"] = [pth.join(self.options[OPTION_OPENVSP_EXE_PATH], VSPAERO_EXE_NAME) + " " + self.stdin2]
        else:
            self.options["command"] = [pth.join(tmp_directory.name, VSPAERO_EXE_NAME) + " " + self.stdin2]
        
        # standard AERO input file -----------------------------------------------------------------
        tmp_result_file_path = pth.join(tmp_directory.name, pth.splitext(self.stdin1)[0], '.lod')
        parser = InputFileGenerator()
        with path(resources, _INPUT_AERO_FILE_NAME) as input_template_path:
            parser.set_template_file(input_template_path)
            parser.set_generated_file(self.stdin1)
            parser.mark_anchor("Sref")
            parser.transfer_var(float(Sref_wing), 0, 8)
            parser.mark_anchor("Cref")
            parser.transfer_var(float(l0_wing), 0, 8)
            parser.mark_anchor("Bref")
            parser.transfer_var(float(span_wing), 0, 8)
            parser.mark_anchor("X_cg")
            parser.transfer_var(float(fa_length), 0, 8)
            parser.mark_anchor("Mach")
            parser.transfer_var(float(mach), 0, 8)
            parser.mark_anchor("AOA")
            parser.transfer_var(AOAList, 0, 8)
            parser.mark_anchor("Vinf")
            parser.transfer_var(float(V_inf), 0, 8)
            parser.mark_anchor("Rho")
            parser.transfer_var(float(rho), 0, 8)
            parser.mark_anchor("ReCref")
            parser.transfer_var(float(reynolds), 0, 12)
            parser.mark_anchor("NumWakeNodes")
            parser.transfer_var(str(multiprocessing.cpu_count()), 0, 17)
            parser.generate()
        
        # Run AERO --------------------------------------------------------------------------------
        self.options["external_input_files"] = [self.stdin1, self.stdin2]
        self.options["external_output_files"] = [tmp_result_file_path]
        super().compute(inputs, outputs)
        
        # Post-processing --------------------------------------------------------------------------
        cl_htp = self._read_lod_file(tmp_result_file_path, AOAList)
        # Calculate derivative
        cl_alpha = (cl_htp[2] + cl_htp[3] - cl_htp[0] - cl_htp[1]) \
                    / ((_INPUT_AOAList[1]-_INPUT_AOAList[0]) * math.pi/180)  
        # Delete temporary directory    
        tmp_directory.cleanup()    

        if self.options["low_speed_aero"]:
            outputs['data:aerodynamics:horizontal_tail:low_speed:CL_alpha'] = cl_alpha
        else:
            outputs['data:aerodynamics:horizontal_tail:cruise:CL_alpha'] = cl_alpha            
        
    @staticmethod
    def _read_lod_file(tmp_result_file_path: str) -> np.ndarray:
        cl_htp = []
        cm_wing = []
        # Colect data from .lod file
        with open(tmp_result_file_path, 'r') as lf:
            data = lf.readlines()
            for i in range(len(data)):
                line = data[i].split()
                line.append('**')
                if line[0] == 'Comp':
                    cl_htp.append(float(data[i+3].split()[5]))
                    cl_htp.append(float(data[i+4].split()[5]))
                    cm_wing.append(float(data[i+1].split()[12]))
                    cm_wing.append(float(data[i+2].split()[12]))
                
        return np.array(cl_htp)
    
    @staticmethod
    def _create_tmp_directory() -> TemporaryDirectory:
        
        """Provide temporary directory for calculation."""
        
        for tmp_base_path in [None, pth.join(str(Path.home()), ".fast")]:
            if tmp_base_path is not None:
                os.makedirs(tmp_base_path, exist_ok=True)
            tmp_directory = tempfile.TemporaryDirectory(prefix="x", dir=tmp_base_path)
            break
            
        return tmp_directory