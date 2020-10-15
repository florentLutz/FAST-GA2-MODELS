"""
    Estimation of HTP low-speed lift and induced moment using OPENVSP
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
import shutil
import os
import os.path as pth
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from fastoad.utils.physics import Atmosphere
from fastoad.utils.resource_management.copy import copy_resource, copy_resource_folder
from importlib_resources import path
from openmdao.components.external_code_comp import ExternalCodeComp
from openmdao.utils.file_wrap import InputFileGenerator

from . import resources
from . import openvsp3201
from ...constants import HT_POINT_COUNT

OPTION_OPENVSP_EXE_PATH = "openvsp_exe_path"
OPTION_RESULT_FOLDER_PATH = "result_folder_path"

_INPUT_SCRIPT_FILE_NAME = "wing_ht_openvsp.vspscript"
_INPUT_AERO_FILE_NAME = "wing_ht_openvsp_DegenGeom"
_LIFT_EFFECIVESS_FILE_NAME = "lift_effectiveness.txt"
_AIRFOIL_0_FILE_NAME = "naca23012.af"
_AIRFOIL_1_FILE_NAME = "naca23012.af"
_AIRFOIL_2_FILE_NAME = "naca23012.af"
VSPSCRIPT_EXE_NAME = "vspscript.exe"
VSPAERO_EXE_NAME = "vspaero.exe"

ALPHA_LIMIT = 30.0 # Limit angle for calculation
_INPUT_AOAList = list(np.linspace(0.0, ALPHA_LIMIT, HT_POINT_COUNT))

class ComputeHTPCLCMopenvsp(ExternalCodeComp):

    def initialize(self):
        self.options.declare(OPTION_RESULT_FOLDER_PATH, default="", types=str)
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
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:z:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
        
        self.add_output("data:aerodynamics:horizontal_tail:low_speed:alpha", shape=len(_INPUT_AOAList), units="deg")
        self.add_output("data:aerodynamics:horizontal_tail:low_speed:CL", shape=len(_INPUT_AOAList))
        self.add_output("data:aerodynamics:horizontal_tail:low_speed:CM", shape=len(_INPUT_AOAList))
        
        self.declare_partials("*", "*", method="fd")        
    
    def compute(self, inputs, outputs):

        # Create result folder first (if it must fail, let it fail as soon as possible)
        result_folder_path = self.options[OPTION_RESULT_FOLDER_PATH]
        if result_folder_path != "":
            os.makedirs(pth.join(result_folder_path,'ClCmHT'), exist_ok=True)

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
        height_htp = inputs["data:geometry:horizontal_tail:z:from_wingMAC25"]
        mach = inputs["data:aerodynamics:low_speed:mach"]
        altitude = 0.0
        
        # Compute remaining inputs
        x_wing = fa_length-x0_wing-0.25*l0_wing
        z_wing = -(height_max - 0.12*l2_wing)*0.5
        span2_wing = y4_wing - y2_wing
        distance_htp = fa_length + lp_htp - 0.25 * l0_htp - x0_htp
        atm = Atmosphere(altitude)
        viscosity = atm.kinematic_viscosity
        rho = atm.density
        V_inf = max(atm.speed_of_sound * mach, 0.01) # avoid V=0 m/s crashes
        reynolds = V_inf * l0_wing / viscosity
        
        # OPENVSP-SCRIPT: Geometry generation ######################################################
        
        # I/O files --------------------------------------------------------------------------------
        tmp_directory = self._create_tmp_directory()
        if self.options[OPTION_OPENVSP_EXE_PATH]:
            target_directory = pth.abspath(self.options[OPTION_OPENVSP_EXE_PATH])
        else:
            target_directory = tmp_directory.name
        input_file_list = [pth.join(target_directory, _INPUT_SCRIPT_FILE_NAME)]
        input_file_list.append(pth.join(target_directory, _AIRFOIL_0_FILE_NAME))
        input_file_list.append(pth.join(target_directory, _AIRFOIL_1_FILE_NAME))
        input_file_list.append(pth.join(target_directory, _AIRFOIL_2_FILE_NAME))
        tmp_result_file_path = pth.join(target_directory, _INPUT_AERO_FILE_NAME + '0.csv')
        output_file_list = [tmp_result_file_path]
        self.options["external_input_files"] = input_file_list
        self.options["external_output_files"] = output_file_list
        
        # Pre-processing (populating temp directory) -----------------------------------------------
        # Copy resource in temp directory if needed
        if not (self.options[OPTION_OPENVSP_EXE_PATH]):
            copy_resource_folder(openvsp3201, target_directory)
            copy_resource(resources, _AIRFOIL_0_FILE_NAME, target_directory)
            copy_resource(resources, _AIRFOIL_1_FILE_NAME, target_directory)
            copy_resource(resources, _AIRFOIL_2_FILE_NAME, target_directory)
        # Create corresponding .bat file
        self.options["command"] = [pth.join(target_directory, 'vspscript.bat')]
        command = pth.join(target_directory, VSPSCRIPT_EXE_NAME) + ' -script ' + pth.join(target_directory, _INPUT_SCRIPT_FILE_NAME) + ' >nul 2>nul\n'
        batch_file = open(self.options["command"][0], "w+")
        batch_file.write("@echo off\n")
        batch_file.write(command)
        batch_file.close()
        
        # standard SCRIPT input file ----------------------------------------------------------------
        parser = InputFileGenerator()
        with path(resources, _INPUT_SCRIPT_FILE_NAME) as input_template_path:
            parser.set_template_file(input_template_path)
            parser.set_generated_file(input_file_list[0])
            parser.mark_anchor("x_wing")
            parser.transfer_var(float(x_wing), 0, 5)
            parser.mark_anchor("z_wing")
            parser.transfer_var(float(z_wing), 0, 5)
            parser.mark_anchor("y1_wing")
            parser.transfer_var(float(y1_wing), 0, 5)
            for i in range(3):
                parser.mark_anchor("l2_wing")
                parser.transfer_var(float(l2_wing), 0, 5)
            parser.reset_anchor()
            parser.mark_anchor("span2_wing")
            parser.transfer_var(float(span2_wing), 0, 5)
            parser.mark_anchor("l4_wing")
            parser.transfer_var(float(l4_wing), 0, 5)
            parser.mark_anchor("sweep_0_wing")
            parser.transfer_var(float(sweep_0_wing), 0, 5)
            parser.mark_anchor("airfoil_0_file")
            parser.transfer_var(self._rewrite_path(input_file_list[1]), 0, 3)
            parser.mark_anchor("airfoil_1_file")
            parser.transfer_var(self._rewrite_path(input_file_list[2]), 0, 3)
            parser.mark_anchor("airfoil_2_file")
            parser.transfer_var(self._rewrite_path(input_file_list[3]), 0, 3)
            parser.mark_anchor("distance_htp")
            parser.transfer_var(float(distance_htp), 0, 5)
            parser.mark_anchor("height_htp")
            parser.transfer_var(float(height_htp), 0, 5)
            parser.mark_anchor("span_htp")
            parser.transfer_var(float(span_htp), 0, 5)
            parser.mark_anchor("root_chord_htp")
            parser.transfer_var(float(root_chord_htp), 0, 5)
            parser.mark_anchor("tip_chord_htp")
            parser.transfer_var(float(tip_chord_htp), 0, 5)
            parser.mark_anchor("sweep_25_htp")
            parser.transfer_var(float(sweep_25_htp), 0, 5)
            parser.mark_anchor("csv_file")
            parser.transfer_var(self._rewrite_path(tmp_result_file_path), 0, 3)
            parser.generate()
            
        # Run SCRIPT --------------------------------------------------------------------------------
        super().compute(inputs, outputs)

        # Getting input/output files if needed
        if self.options[OPTION_RESULT_FOLDER_PATH] != "":
            for file_path in input_file_list:
                new_path = pth.join(result_folder_path, 'ClCmHT', pth.split(file_path)[1])
                if pth.exists(file_path):
                    shutil.copyfile(file_path, new_path)
            for file_path in output_file_list:
                new_path = pth.join(result_folder_path, 'ClCmHT', pth.split(file_path)[1])
                if pth.exists(file_path):
                    shutil.copyfile(file_path, new_path)
 
        # OPENVSP-AERO: aero calculation ############################################################
       
        # I/O files --------------------------------------------------------------------------------
        # Duplicate .csv file for multiple run
        input_file_list = [tmp_result_file_path]
        for idx in range(len(_INPUT_AOAList) - 1):
            shutil.copy(tmp_result_file_path, pth.join(target_directory, _INPUT_AERO_FILE_NAME + str(idx + 1) + '.csv'))
            input_file_list.append(pth.join(target_directory, _INPUT_AERO_FILE_NAME + str(idx + 1) + '.csv'))
        output_file_list = []
        for idx in range(len(_INPUT_AOAList)):
            input_file_list.append(pth.join(target_directory, _INPUT_AERO_FILE_NAME) + str(idx) + '.vspaero')
            output_file_list.append(pth.join(target_directory, _INPUT_AERO_FILE_NAME) + str(idx) + '.lod')
        self.options["external_input_files"] = input_file_list
        self.options["external_output_files"] = output_file_list

        # Pre-processing (create batch file) -------------------------------------------------------
        self.options["command"] = [pth.join(target_directory, 'vspaero.bat')]
        batch_file = open(self.options["command"][0], "w+")
        batch_file.write("@echo off\n")
        for idx in range(len(_INPUT_AOAList)):
            command = pth.join(target_directory, VSPAERO_EXE_NAME) + ' ' + pth.join(target_directory, _INPUT_AERO_FILE_NAME + str(idx) + ' >nul 2>nul\n')
            batch_file.write(command)
        batch_file.close()
        
        # standard AERO input file -----------------------------------------------------------------
        parser = InputFileGenerator()
        for idx in range(len(_INPUT_AOAList)):
            with path(resources, _INPUT_AERO_FILE_NAME + '.vspaero') as input_template_path:
                parser.set_template_file(input_template_path)
                parser.set_generated_file(input_file_list[len(_INPUT_AOAList) + idx])
                parser.reset_anchor()
                parser.mark_anchor("Sref")
                parser.transfer_var(float(Sref_wing), 0, 3)
                parser.mark_anchor("Cref")
                parser.transfer_var(float(l0_wing), 0, 3)
                parser.mark_anchor("Bref")
                parser.transfer_var(float(span_wing), 0, 3)
                parser.mark_anchor("X_cg")
                parser.transfer_var(float(fa_length), 0, 3)
                parser.mark_anchor("Mach")
                parser.transfer_var(float(mach), 0, 3)
                parser.mark_anchor("AOA")
                parser.transfer_var(float(_INPUT_AOAList[idx]), 0, 3)
                parser.mark_anchor("Vinf")
                parser.transfer_var(float(V_inf), 0, 3)
                parser.mark_anchor("Rho")
                parser.transfer_var(float(rho), 0, 3)
                parser.mark_anchor("ReCref")
                parser.transfer_var(float(reynolds), 0, 3)
                parser.generate()
        
        # Run AERO --------------------------------------------------------------------------------
        super().compute(inputs, outputs)
        
        # Post-processing --------------------------------------------------------------------------
        result_cl = []
        result_cm = []
        for idx in range(len(_INPUT_AOAList)):
            cl_htp, cm_wing = self._read_lod_file(output_file_list[idx])
            result_cl.append(cl_htp)
            result_cm.append(cm_wing)

        outputs['data:aerodynamics:horizontal_tail:low_speed:alpha'] = np.array(_INPUT_AOAList)
        outputs['data:aerodynamics:horizontal_tail:low_speed:CL'] = np.array(result_cl)
        outputs['data:aerodynamics:horizontal_tail:low_speed:CM'] = np.array(result_cm)

        # Getting input/output files if needed
        if self.options[OPTION_RESULT_FOLDER_PATH] != "":
            for file_path in input_file_list:
                new_path = pth.join(result_folder_path, 'ClCmHT', pth.split(file_path)[1])
                if pth.exists(file_path):
                    shutil.copyfile(file_path, new_path)
            for file_path in output_file_list:
                new_path = pth.join(result_folder_path, 'ClCmHT', pth.split(file_path)[1])
                if pth.exists(file_path):
                    shutil.copyfile(file_path, new_path)

        # Delete temporary directory
        tmp_directory.cleanup()
        
    @staticmethod
    def _read_lod_file(tmp_result_file_path: str):
        """
        Collect data from .lod file
        """
        cl_htp = 0.0
        cm_htp = 0.0
        with open(tmp_result_file_path, 'r') as lf:
            data = lf.readlines()
            for i in range(len(data)):
                line = data[i].split()
                line.append('**') # avoid void line error
                if line[0] == 'Comp':
                    cl_htp = float(data[i+3].split()[5])+float(data[i+4].split()[5]) # sum CL on left and right part of htp
                    cm_htp = float(data[i+3].split()[12])+float(data[i+4].split()[12]) # sum CM on left and right part of htp
                
        return cl_htp, cm_htp
    
    @staticmethod
    def _create_tmp_directory() -> TemporaryDirectory:
        
        """Provide temporary directory for calculation."""
        
        for tmp_base_path in [None, pth.join(str(Path.home()), ".fast")]:
            if tmp_base_path is not None:
                os.makedirs(tmp_base_path, exist_ok=True)
            tmp_directory = tempfile.TemporaryDirectory(prefix="x", dir=tmp_base_path)
            break
            
        return tmp_directory

    @staticmethod
    def _rewrite_path(file_path: str) -> str:
        file_path = '\"' + file_path.replace('\\', '\\\\') + '\"'
        return file_path