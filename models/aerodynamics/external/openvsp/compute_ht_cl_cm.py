"""
    Estimation of HTP lift and induced moment using OPENVSP
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
from scipy import interpolate
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
_LIFT_EFFECIVESS_FILE_NAME = "lift_effectiveness.txt"
_AIRFOIL_0_FILE_NAME = "naca23012.af"
_AIRFOIL_1_FILE_NAME = "naca23012.af"
_AIRFOIL_2_FILE_NAME = "naca23012.af"
VSPSCRIPT_EXE_NAME = "vspscript.exe"
VSPAERO_EXE_NAME = "vspaero.exe"

class ComputeHTPCLCMopenvsp(ExternalCodeComp):

    def initialize(self):
        
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
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:flap:chord_ratio", val=np.nan)
        self.add_input("data:geometry:flap:span_ratio", val=np.nan)
        self.add_input("data:geometry:flap:type", val=1.0)
        self.add_input("openvsp:altitude", val=np.nan, units="ft")
        self.add_input("openvsp:mach", val=np.nan)
        self.add_input("openvsp:alpha", val=np.nan, units="deg")
        self.add_input("openvsp:elevator_angle", val=-25.0, units="deg")
        self.add_input("openvsp:flaps_angle", val=10.0, units="deg")
        self.add_input("openvsp:cl_alpha_wing", val=np.nan)
        
        self.add_output("openvsp:cl_htp")
        self.add_output("openvsp:cm_wing")
        
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
        taper_ratio = inputs['data:geometry:wing:taper_ratio']
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        sweep_25_htp = inputs["data:geometry:horizontal_tail:sweep_25"]
        span_htp = inputs["data:geometry:horizontal_tail:span"]/2.0
        root_chord_htp = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord_htp = inputs["data:geometry:horizontal_tail:tip:chord"]
        lp_htp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"] 
        l0_htp = inputs["data:geometry:horizontal_tail:MAC:length"] 
        x0_htp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:local"]
        height_htp = inputs["data:geometry:horizontal_tail:height"]
        Sref_htp = inputs['data:geometry:horizontal_tail:area']
        altitude = inputs["openvsp:altitude"]
        mach = inputs["openvsp:mach"]
        alpha = inputs["openvsp:alpha"]
        elevator_angle = inputs["openvsp:elevator_angle"]
        flaps_angle = inputs["openvsp:flaps_angle"]
        x_wing = fa_length-x0_wing-0.25*l0_wing
        z_wing = -(height_max - 0.12*l2_wing)*0.5
        span2_wing = y4_wing - y2_wing
        distance_htp = fa_length + lp_htp - 0.25 * l0_htp - x0_htp
        AOAList = str(alpha)
        atm = Atmosphere(altitude)
        speed_of_sound = atm.speed_of_sound
        viscosity = atm.kinematic_viscosity
        rho = atm.density
        V_inf = min(speed_of_sound * mach, 0.1) # avoid V=0 m/s crashes
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
            parser.transfer_var(float(x_wing), 1, 1)
            parser.mark_anchor("z_wing")
            parser.transfer_var(float(z_wing), 1, 1)
            parser.mark_anchor("y1_wing")
            parser.transfer_var(float(y1_wing), 1, 1)
            parser.mark_anchor("l2_wing")
            parser.transfer_var(float(l2_wing), 1, 1)
            parser.mark_anchor("span2_wing")
            parser.transfer_var(float(span2_wing), 1, 1)
            parser.mark_anchor("l4_wing")
            parser.transfer_var(float(l4_wing), 1, 1)
            parser.mark_anchor("sweep_0_wing")
            parser.transfer_var(float(sweep_0_wing), 1, 1)
            parser.mark_anchor("airfoil_0_file")
            parser.transfer_var(self.stdin2, 1, 1)
            parser.mark_anchor("airfoil_1_file")
            parser.transfer_var(self.stdin3, 1, 1)
            parser.mark_anchor("airfoil_2_file")
            parser.transfer_var(self.stdin4, 1, 1)
            parser.mark_anchor("distance_htp")
            parser.transfer_var(float(distance_htp), 1, 1)
            parser.mark_anchor("height_htp")
            parser.transfer_var(float(height_htp), 1, 1)
            parser.mark_anchor("span_htp")
            parser.transfer_var(float(span_htp), 1, 1)
            parser.mark_anchor("root_chord_htp")
            parser.transfer_var(float(root_chord_htp), 1, 1)
            parser.mark_anchor("tip_chord_htp")
            parser.transfer_var(float(tip_chord_htp), 1, 1)
            parser.mark_anchor("sweep_25_htp")
            parser.transfer_var(float(sweep_25_htp), 1, 1)
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
            parser.mark_anchor("Sref_wing")
            parser.transfer_var(float(Sref_wing), 1, 1)
            parser.mark_anchor("l0_wing")
            parser.transfer_var(float(l0_wing), 1, 1)
            parser.mark_anchor("span_wing")
            parser.transfer_var(float(span_wing), 1, 1)
            parser.mark_anchor("fa_length")
            parser.transfer_var(float(fa_length), 1, 1)
            parser.mark_anchor("mach_")
            parser.transfer_var(float(mach), 1, 1)
            parser.mark_anchor("AOAList")
            parser.transfer_var(AOAList, 1, 1)
            parser.mark_anchor("V_inf")
            parser.transfer_var(float(V_inf), 1, 1)
            parser.mark_anchor("rho_")
            parser.transfer_var(float(rho), 1, 1)
            parser.mark_anchor("reynolds")
            parser.transfer_var(float(reynolds), 1, 1)
            parser.mark_anchor("cpu_count")
            parser.transfer_var(str(multiprocessing.cpu_count()), 1, 1)
            parser.generate()
        
        # Run AERO --------------------------------------------------------------------------------
        self.options["external_input_files"] = [self.stdin1, self.stdin2]
        self.options["external_output_files"] = [tmp_result_file_path]
        super().compute(inputs, outputs)
        
        # Post-processing --------------------------------------------------------------------------
        cl_htp, cm_wing = self._read_lod_file(tmp_result_file_path, AOAList)
        # Calculate totals
        cl_htp = cl_htp[0] + cl_htp[1]
        cm_wing = cm_wing[0] + cm_wing[1]
        # Corrections due to elevators. Default: maximum deflection (-25deg)
        cldelta_theory = 4.5 #Fig 8.14, for t/c=0.10 and cf/c=0.3
        if abs(elevator_angle)<=12: #Fig 8.13. Quick linear approximation of graph (not intended to be used)
            k = 1.0 
        elif abs(elevator_angle)<=20:
            k = 1.0 - 0.0263*(abs(elevator_angle) - 12)
        elif abs(elevator_angle)<=25:
            k = 0.79 - 0.024* (abs(elevator_angle)-20)
        else:
            k = 0.67 - 0.008 * (abs(elevator_angle) - 25) 
        k1 = 1.05 #cf/c = 0.3 (Roskam 3D flap parameters)
        if abs(elevator_angle)<=15: #Fig 8.33. Quick linear approximation of graph (not intended to be used)
            k2 = 0.46 / 15 * abs(elevator_angle) 
        else:
            k2 = 0.46 + 0.22/10 * (abs(elevator_angle) - 15)
        delta_cl_elev = (cldelta_theory * k*k1*k2 * elevator_angle * math.pi/180) \
                        * Sref_htp/Sref_wing
        # Corrections due to flaps : method from Roskam (sweep=0, flaps 60%, simple slotted and not extensible,
        # at 25% MAC, cf/c+0.25)
        k_p = interpolate.interp1d([0.,0.2,0.33,0.5,1.],[0.65,0.75,0.7,0.63,0.5]) # Figure 8.105, interpolated function of taper ratio (span ratio fixed)
        delta_cl_wing = self.compute_delta_cz_highlift(flaps_angle, 0., mach=0.1)
        delta_cm = k_p(taper_ratio) * (-0.27)*(delta_cl_wing) #-0.27: Figure 8.106
        
        outputs["openvsp:cl_htp"] = cl_htp + delta_cl_elev
        outputs["openvsp:cm_wing"] = cm_wing + delta_cm
            
        # Delete temporary directory    
        tmp_directory.cleanup()                
        
    @staticmethod
    def _read_lod_file(tmp_result_file_path: str):
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
                
        return np.array(cl_htp), np.array(cm_wing)
    
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
    def _compute_alpha_flap(flap_angle, ratio_cf_flap):
        """Method to use a Roskam graph to calculate the effectiveness of a 
        simple slotted flap
        """
        temp_array = []
        fichier = open(path(resources, _LIFT_EFFECIVESS_FILE_NAME), "r")
        for line in fichier:
            temp_array.append([float(x) for x in line.split(',')])
        fichier.close()
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        x3 = []
        y3 = []
        x4 = []
        y4 = []
        x5 = []
        y5 = []
        for arr in temp_array:
            x1.append(arr[0])
            y1.append(arr[1])
            x2.append(arr[2])
            y2.append(arr[3])
            x3.append(arr[4])
            y3.append(arr[5])
            x4.append(arr[6])
            y4.append(arr[7])
            x5.append(arr[8])
            y5.append(arr[9])
        tck1 = interpolate.splrep(x1, y1, s=0)
        tck2 = interpolate.splrep(x2, y2, s=0)
        tck3 = interpolate.splrep(x3, y3, s=0)
        tck4 = interpolate.splrep(x4, y4, s=0)
        tck5 = interpolate.splrep(x5, y5, s=0)
        ynew1 = interpolate.splev(flap_angle, tck1, der=0)
        ynew2 = interpolate.splev(flap_angle, tck2, der=0)
        ynew3 = interpolate.splev(flap_angle, tck3, der=0)
        ynew4 = interpolate.splev(flap_angle, tck4, der=0)
        ynew5 = interpolate.splev(flap_angle, tck5, der=0)
        zs = [0.15, 0.20, 0.25, 0.30, 0.40]
        y_final = [ynew1, ynew2, ynew3, ynew4, ynew5]
        tck6 = interpolate.splrep(zs, y_final, s=0)
        
        alpha_flap = interpolate.splev(ratio_cf_flap, tck6, der=0)
        
        return alpha_flap
    
    def compute_delta_cl_flaps(self, inputs, flaps_angle):
        """  Calculates the Cz produced by flap and slat based on Roskam book and Raymer book."""
        
        flap_chord_ratio = inputs['data:geometry:flap:chord_ratio']
        flap_span_ratio = inputs['data:geometry:flap:span_ratio']
        flap_type = inputs['data:geometry:flap:type']
        mach = inputs['openvsp:mach']
        cl_alpha_wing = inputs['openvsp:cl_alpha_wing']

        #2D flap lift coefficient
        if flap_type == 1.0: #Slotted flap
        #Roskam vol6 efficiency factor for single slotted flaps
            alpha_flap = self._compute_alpha_flap(flaps_angle, flap_chord_ratio)
            delta_cl_airfoil = 2*math.pi / math.sqrt(1 - mach**2)* alpha_flap * (flaps_angle / 180 * math.pi)
        else: #Plain flap
            cldelta_theory = 4.1 #Fig 8.14, for t/c=0.12 and cf/c=0.25
            if flaps_angle <= 13: #Fig 8.13. Quick linear approximation of graph (not intended to be used)
                k = 1.0 
            elif flaps_angle <= 20:
                k = 0.83
            elif flaps_angle <= 30:
                k = 0.83 - 0.018* (flaps_angle-20)
            else:
                k = 0.65 - 0.008 * (flaps_angle - 30) 
            delta_cl_airfoil = cldelta_theory * k * (flaps_angle / 180 * math.pi)
        #Roskam 3D flap parameters
        kb = 1.25*flap_span_ratio #PROVISIONAL (fig 8.52)
        effect  = 1.04 #fig 8.53 (cf/c=0.25, small effect of AR)
        
        delta_cl_flap = kb * delta_cl_airfoil * (cl_alpha_wing/(2*math.pi)) * effect

        return delta_cl_flap   # Cz due to high lift devices
    