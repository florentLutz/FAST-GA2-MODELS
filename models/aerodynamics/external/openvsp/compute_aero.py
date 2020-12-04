"""
    Estimation of all the aero coefficients using OPENVSP
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

from importlib_resources import path
import math
import numpy as np
import pandas as pd
from openmdao.components.external_code_comp import ExternalCodeComp
from openmdao.utils.file_wrap import InputFileGenerator
import os
import os.path as pth
from pathlib import Path
import shutil
import tempfile
from tempfile import TemporaryDirectory
import warnings
from typing import Tuple, List

from fastoad.utils.physics import Atmosphere
from fastoad.utils.resource_management.copy import copy_resource, copy_resource_folder

from ... import resources
from . import resources as local_resources
from . import openvsp3201
from ...constants import SPAN_MESH_POINT_OPENVSP

OPTION_SPEED = "low_speed_aero"
OPTION_WING_AIRFOIL = "wing_airfoil_file"
OPTION_HTP_AIRFOIL = "htp_airfoil_file"
OPTION_OPENVSP_EXE_PATH = "openvsp_exe_path"
OPTION_RESULT_FOLDER_PATH = "result_folder_path"

INPUT_AOA = 4.0  # only one value given since calculation is done by default around 0.0!
INPUT_SCRIPT_FILE_NAME_1 = "wing_openvsp.vspscript"
INPUT_SCRIPT_FILE_NAME_2 = "wing_ht_openvsp.vspscript"
DEFAULT_WING_AIRFOIL = "naca23012.af"
DEFAULT_HTP_AIRFOIL = "naca0012.af"
STDERR_FILE_NAME = "vspaero_calc.err"
VSPSCRIPT_EXE_NAME = "vspscript.exe"
VSPAERO_EXE_NAME = "vspaero.exe"


class ComputeAEROopenvsp(ExternalCodeComp):

    def initialize(self):
        self.options.declare(OPTION_SPEED, default=False, types=bool)
        self.options.declare(OPTION_RESULT_FOLDER_PATH, default="", types=str)
        self.options.declare(OPTION_OPENVSP_EXE_PATH, default="", types=str, allow_none=True)
        self.options.declare(OPTION_WING_AIRFOIL, default=DEFAULT_WING_AIRFOIL, types=str, allow_none=True)
        self.options.declare(OPTION_HTP_AIRFOIL, default=DEFAULT_HTP_AIRFOIL, types=str, allow_none=True)
        
    def setup(self):

        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
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
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:z:from_wingMAC25", val=np.nan, units="m")
        if self.options[OPTION_SPEED]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
        else:
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units='ft')

        if self.options[OPTION_SPEED]:
            self.add_output("data:aerodynamics:wing:low_speed:CL0_clean")
            self.add_output("data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1")
            self.add_output("data:aerodynamics:wing:low_speed:CM0_clean")
            self.add_output("data:aerodynamics:wing:low_speed:CM_alpha", units="rad**-1")
            self.add_output("data:aerodynamics:wing:low_speed:Y_vector", shape=SPAN_MESH_POINT_OPENVSP, units="m")
            self.add_output("data:aerodynamics:wing:low_speed:CL_vector", shape=SPAN_MESH_POINT_OPENVSP)
            self.add_output("data:aerodynamics:wing:low_speed:induced_drag_coefficient")
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:CL0")
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1")
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:CM0")
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:CM_alpha", units="rad**-1")
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient")
        else:
            self.add_output("data:aerodynamics:wing:cruise:CL0_clean")
            self.add_output("data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1")
            self.add_output("data:aerodynamics:wing:cruise:CM0_clean")
            self.add_output("data:aerodynamics:wing:cruise:CM_alpha", units="rad**-1")
            self.add_output("data:aerodynamics:wing:cruise:induced_drag_coefficient")
            self.add_output("data:aerodynamics:horizontal_tail:cruise:CL0")
            self.add_output("data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1")
            self.add_output("data:aerodynamics:horizontal_tail:cruise:CM0")
            self.add_output("data:aerodynamics:horizontal_tail:cruise:CM_alpha", units="rad**-1")
            self.add_output("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient")
        
        self.declare_partials("*", "*", method="fd")

    def check_config(self, logger):
        # let void to avoid logger error on "The command cannot be empty"
        pass
    
    def compute(self, inputs, outputs):

        # Check AOA input length to be =2
        if not (type(INPUT_AOA) == float):
            raise TypeError('INPUT_AOA should be a float!')
        else:
            INPUT_AOAList = [0.0, INPUT_AOA]

        # Define mach
        if self.options["low_speed_aero"]:
            altitude = 0.0
            atm = Atmosphere(altitude)
            mach = inputs["data:aerodynamics:low_speed:mach"]
        else:
            altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
            atm = Atmosphere(altitude)
            mach = inputs["data:aerodynamics:cruise:mach"]

        # Create result folder first (if it must fail, let it fail as soon as possible)
        result_folder_path = self.options[OPTION_RESULT_FOLDER_PATH]
        if result_folder_path != "":
            if not os.path.exists(result_folder_path):
                os.makedirs(pth.join(result_folder_path), exist_ok=True)

        # Get the primary form factors for wing/htp and store it to geometryX.csv file (to avoid re-computation)
        # or save result file path if geometry already computed (to load results afterwards)
        already_computed = False
        result_file_path = None
        if result_folder_path != "":
            sweep25_wing = float(inputs["data:geometry:wing:sweep_25"])
            taper_ratio_wing = float(inputs["data:geometry:wing:taper_ratio"])
            aspect_ratio_wing = float(inputs["data:geometry:wing:aspect_ratio"])
            sweep25_htp = float(inputs["data:geometry:horizontal_tail:sweep_25"])
            taper_ratio_htp = float(inputs["data:geometry:horizontal_tail:taper_ratio"])
            geometry_set = np.array([sweep25_wing, taper_ratio_wing, aspect_ratio_wing, sweep25_htp, taper_ratio_htp,
                                     float(mach)])
            geometry_set_labels = ["sweep25_wing", "taper_ratio_wing", "aspect_ratio_wing", "sweep25_htp",
                                   "taper_ratio_htp", "mach"]
            # If some results already stored search for corresponding geometry
            if pth.exists(pth.join(result_folder_path, "geometry_0.csv")):
                idx = 0
                while pth.exists(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv")):
                    if pth.exists(pth.join(result_folder_path, "results_" + str(idx) + ".csv")):
                        data = pd.read_csv(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv"))
                        values = data.to_numpy()[:, 1].tolist()
                        labels = data.to_numpy()[:, 0].tolist()
                        data = pd.DataFrame(values, index=labels)
                        # noinspection PyBroadException
                        try:
                            if np.size(data.loc[geometry_set_labels, 0].to_numpy()) == 6:
                                if np.sum(data.loc[geometry_set_labels, 0].to_numpy() == geometry_set) == 6:
                                    result_file_path = pth.join(result_folder_path, "results_" + str(idx) + ".csv")
                                    already_computed = True
                                    break
                        except:
                            break
                    idx += 1
            # Save geometry if not already computed
            if not already_computed:
                data = pd.DataFrame(geometry_set, index=geometry_set_labels)
                # Find available index
                idx = 0
                while pth.exists(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv")):
                    idx += 1
                data.to_csv(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv"))
                result_file_path = pth.join(result_folder_path, "results_" + str(idx) + ".csv")

        if not already_computed:
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
            sref_wing = inputs['data:geometry:wing:area']
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

            # Compute remaining inputs
            x_wing = fa_length-x0_wing-0.25*l0_wing
            z_wing = -(height_max - 0.12*l2_wing)*0.5
            span2_wing = y4_wing - y2_wing
            distance_htp = fa_length + lp_htp - 0.25 * l0_htp - x0_htp
            speed_of_sound = atm.speed_of_sound
            viscosity = atm.kinematic_viscosity
            rho = atm.density
            v_inf = max(speed_of_sound * mach, 0.01)  # avoid V=0 m/s crashes
            reynolds = v_inf * l0_wing / viscosity

            # OPENVSP-SCRIPT: Geometry generation ######################################################

            # I/O files --------------------------------------------------------------------------------
            tmp_directory = self._create_tmp_directory()
            # avoid to dump void xternal_code_comp_error.out error file
            self.stderr = pth.join(tmp_directory.name, STDERR_FILE_NAME)
            if self.options[OPTION_OPENVSP_EXE_PATH]:
                target_directory = pth.abspath(self.options[OPTION_OPENVSP_EXE_PATH])
            else:
                target_directory = tmp_directory.name
            input_file_list = [pth.join(target_directory, INPUT_SCRIPT_FILE_NAME_1),  # wing script
                               pth.join(target_directory, INPUT_SCRIPT_FILE_NAME_2),  # wing+htp script
                               pth.join(target_directory, self.options['wing_airfoil_file']),
                               pth.join(target_directory, self.options['htp_airfoil_file'])]
            self.options["external_input_files"] = input_file_list


            # Pre-processing (populating temp directory) -----------------------------------------------
            # Copy resource in temp directory if needed
            if not (self.options[OPTION_OPENVSP_EXE_PATH]):
                # noinspection PyTypeChecker
                copy_resource_folder(openvsp3201, target_directory)
                # noinspection PyTypeChecker
                copy_resource(resources, self.options['wing_airfoil_file'], target_directory)
                # noinspection PyTypeChecker
                copy_resource(resources, self.options['htp_airfoil_file'], target_directory)
            # Create corresponding .bat files (one for each geometry configuration)
            self.options["command"] = [pth.join(target_directory, 'vspscript.bat')]
            batch_file = open(self.options["command"][0], "w+")
            batch_file.write("@echo off\n")
            for idx in range(2):
                if idx == 0:
                    input_script = INPUT_SCRIPT_FILE_NAME_1  # create wing geometry file
                else:
                    input_script = INPUT_SCRIPT_FILE_NAME_2  # create wing+htp geometry file
                command = pth.join(target_directory, VSPSCRIPT_EXE_NAME) + ' -script ' \
                          + pth.join(target_directory, input_script) + ' >nul 2>nul\n'
                batch_file.write(command)
            batch_file.close()

            # standard SCRIPT input file ----------------------------------------------------------------
            output_file_list = []
            for idx in range(2):
                if idx == 0:
                    input_script = INPUT_SCRIPT_FILE_NAME_1  # create wing geometry file
                else:
                    input_script = INPUT_SCRIPT_FILE_NAME_2  # create wing+htp geometry file
                tmp_result_file_path = (pth.join(target_directory, input_script[0:-9] + 'csv'))
                output_file_list.append(tmp_result_file_path)
                parser = InputFileGenerator()
                with path(local_resources, input_script) as input_template_path:
                    parser.set_template_file(str(input_template_path))
                    parser.set_generated_file(input_file_list[idx])
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
                    parser.transfer_var('\"' + input_file_list[-2].replace('\\', '/') + '\"', 0, 3)
                    parser.mark_anchor("airfoil_1_file")
                    parser.transfer_var('\"' + input_file_list[-2].replace('\\', '/') + '\"', 0, 3)
                    parser.mark_anchor("airfoil_2_file")
                    parser.transfer_var('\"' + input_file_list[-2].replace('\\', '/') + '\"', 0, 3)
                    if idx == 1:
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
                        parser.mark_anchor("airfoil_3_file")
                        parser.transfer_var('\"' + input_file_list[-1].replace('\\', '/') + '\"', 0, 3)
                        parser.mark_anchor("airfoil_4_file")
                        parser.transfer_var('\"' + input_file_list[-1].replace('\\', '/') + '\"', 0, 3)
                    parser.mark_anchor("csv_file")
                    csv_name = input_file_list[idx].replace('vspscript', 'csv')
                    parser.transfer_var('\"' + csv_name.replace('\\', '/') + '\"',  0, 3)
                    parser.generate()

            # Run SCRIPT --------------------------------------------------------------------------------
            self.options["external_output_files"] = output_file_list
            super().compute(inputs, outputs)

            # OPENVSP-AERO: aero calculation ############################################################

            # I/O files --------------------------------------------------------------------------------
            # Duplicate .csv file for 2nd AOA run
            input_file_list = output_file_list
            shutil.copy(input_file_list[0], pth.join(target_directory, INPUT_SCRIPT_FILE_NAME_1[0:-10]
                                                     + '_DegenGeom0.csv'))
            input_file_list.append(pth.join(target_directory, INPUT_SCRIPT_FILE_NAME_1[0:-10] + '_DegenGeom0.csv'))
            shutil.copy(input_file_list[0], pth.join(target_directory, INPUT_SCRIPT_FILE_NAME_1[0:-10]
                                                     + '_DegenGeom1.csv'))
            input_file_list.append(pth.join(target_directory, INPUT_SCRIPT_FILE_NAME_1[0:-10] + '_DegenGeom1.csv'))
            os.remove(input_file_list[0])
            input_file_list.pop(0)
            shutil.copy(input_file_list[0], pth.join(target_directory, INPUT_SCRIPT_FILE_NAME_2[0:-10]
                                                     + '_DegenGeom0.csv'))
            input_file_list.append(pth.join(target_directory, INPUT_SCRIPT_FILE_NAME_2[0:-10] + '_DegenGeom0.csv'))
            shutil.copy(input_file_list[0], pth.join(target_directory, INPUT_SCRIPT_FILE_NAME_2[0:-10]
                                                     + '_DegenGeom1.csv'))
            input_file_list.append(pth.join(target_directory, INPUT_SCRIPT_FILE_NAME_2[0:-10] + '_DegenGeom1.csv'))
            os.remove(input_file_list[0])
            input_file_list.pop(0)
            output_file_list = []
            for idx in range(4):
                input_file_list.append(input_file_list[idx].replace('csv', 'vspaero'))
                output_file_list.append(input_file_list[idx].replace('csv', 'lod'))
            self.options["external_input_files"] = input_file_list
            self.options["external_output_files"] = output_file_list

            # Pre-processing (populating temp directory) -----------------------------------------------
            self.options["command"] = [pth.join(target_directory, 'vspaero.bat')]
            batch_file = open(self.options["command"][0], "w+")
            batch_file.write("@echo off\n")
            for idx in range(4):
                command = pth.join(target_directory, VSPAERO_EXE_NAME) + ' ' \
                          + input_file_list[4+idx].replace('.vspaero', '') + ' >nul 2>nul\n'
                batch_file.write(command)
            batch_file.close()

            # standard AERO input file -----------------------------------------------------------------
            parser = InputFileGenerator()
            for idx in range(4):
                template_file = pth.split(input_file_list[4+idx])[1].replace(str(idx % 2), '')
                with path(local_resources, template_file) as input_template_path:
                    parser.set_template_file(str(input_template_path))
                    parser.set_generated_file(input_file_list[4+idx])
                    parser.reset_anchor()
                    parser.mark_anchor("Sref")
                    parser.transfer_var(float(sref_wing), 0, 3)
                    parser.mark_anchor("Cref")
                    parser.transfer_var(float(l0_wing), 0, 3)
                    parser.mark_anchor("Bref")
                    parser.transfer_var(float(span_wing), 0, 3)
                    parser.mark_anchor("X_cg")
                    parser.transfer_var(float(fa_length), 0, 3)
                    parser.mark_anchor("Mach")
                    parser.transfer_var(float(mach), 0, 3)
                    parser.mark_anchor("AOA")
                    parser.transfer_var(float(INPUT_AOAList[idx % 2]), 0, 3)
                    parser.mark_anchor("Vinf")
                    parser.transfer_var(float(v_inf), 0, 3)
                    parser.mark_anchor("Rho")
                    parser.transfer_var(float(rho), 0, 3)
                    parser.mark_anchor("ReCref")
                    parser.transfer_var(float(reynolds), 0, 3)
                    parser.generate()

            # Run AERO --------------------------------------------------------------------------------
            super().compute(inputs, outputs)

            # Post-processing: STEP1/2 - wing coefficients ---------------------------------------------
            cl_0_wing, cm_0_wing, y_vector, cl_vector, _, _ = self._read_lod_file(output_file_list[0])
            cl_1_wing, cm_1_wing, _, _, _, _ = self._read_lod_file(output_file_list[1])
            cl_wing_vect = [cl_0_wing, cl_1_wing]
            cm_wing_vect = [cm_0_wing, cm_1_wing]
            # Fuselage correction
            k_fus = 1 + 0.025 * width_max / span_wing - 0.025 * (width_max / span_wing) ** 2
            cl_0_wing = float(cl_wing_vect[0] * k_fus)
            cl_1_wing = float(cl_wing_vect[1] * k_fus)
            # Calculate derivative
            cl_alpha_wing = (cl_1_wing - cl_0_wing) / (INPUT_AOAList[1] * math.pi / 180)
            cm_alpha_wing = (cm_1_wing - cm_0_wing) / (INPUT_AOAList[1] * math.pi / 180)
            if SPAN_MESH_POINT_OPENVSP < len(y_vector):
                y_interp = np.linspace(y_vector[0], y_vector[-1], SPAN_MESH_POINT_OPENVSP)
                cl_vector = np.interp(y_interp, y_vector, cl_vector)
                y_vector = y_interp
                warnings.warn("Defined maximum span mesh in fast aerodynamics\\constants.py exceeded!")
            elif SPAN_MESH_POINT_OPENVSP >= len(y_vector):
                additional_zeros = list(np.zeros(SPAN_MESH_POINT_OPENVSP-len(y_vector)))
                y_vector.extend(additional_zeros)
                cl_vector.extend(additional_zeros)
            # Calculate oswald
            oswald = self._read_polar_file(output_file_list[0].replace('lod', 'polar'))
            k_fus = 1 - 2 * (width_max / span_wing) ** 2  # Fuselage correction
            # Full aircraft correction: Wing lift is 105% of total lift.
            # This means CDind = (CL*1.05)^2/(piAe) -> e' = e/1.05^2
            coef_e = float(oswald * k_fus / 1.05 ** 2)
            coef_k_wing = float(1. / (math.pi * span_wing ** 2 / sref_wing * coef_e))

            # Post-processing: STEP2/2 - HTP coefficients ----------------------------------------------
            cl_htp_vect = []
            cm_htp_vect = []
            for idx in range(2, 4):
                cm_wing, _, _, _, cl_htp, cm_htp = self._read_lod_file(output_file_list[idx])
                # calculate aero-center
                x_aero_center = (cm_wing - cm_wing_vect[idx-2])/cl_wing_vect[idx-2]
                # correct htp CM
                cm_htp = cm_htp + cl_htp * (lp_htp-x_aero_center)
                cl_htp_vect.append(cl_htp)
                cm_htp_vect.append(cm_htp)
            cl_0_htp = cl_htp_vect[0]
            cm_0_htp = float(cm_htp_vect[0])
            # Calculate derivative
            cl_alpha_htp = float((cl_htp_vect[1] - cl_htp_vect[0]) / (INPUT_AOAList[1] * math.pi/180))
            cm_alpha_htp = float((cm_htp_vect[1] - cm_htp_vect[0]) / (INPUT_AOAList[1] * math.pi / 180))
            # Read oswald
            coef_e = self._read_polar_file(output_file_list[2].replace('lod', 'polar')) - oswald
            coef_k_htp = float(1. / (math.pi * span_htp ** 2 / sref_wing * coef_e))

            # Save results to defined path -------------------------------------------------------------
            if self.options[OPTION_RESULT_FOLDER_PATH] != "":
                results = [cl_0_wing, cl_alpha_wing, cm_0_wing, cm_alpha_wing, y_vector, cl_vector, cl_0_htp,
                           cl_alpha_htp, cm_0_htp, cm_alpha_htp, coef_k_wing, coef_k_htp]
                labels = ["cl_0_wing", "cl_alpha_wing", "cm_0_wing", "cm_alpha_wing", "y_vector", "cl_vector",
                          "cl_0_htp", "cl_alpha_htp", "cm_0_htp", "cm_alpha_htp", "coef_k_wing", "coef_k_htp"]
                data = pd.DataFrame(results, index=labels)
                data.to_csv(result_file_path)

        else:
            # Read values from result file -------------------------------------------------------------
            data = pd.read_csv(result_file_path)
            values = data.to_numpy()[:, 1].tolist()
            labels = data.to_numpy()[:, 0].tolist()
            data = pd.DataFrame(values, index=labels)
            cl_0_wing = float(data.loc["cl_0_wing", 0])
            cl_alpha_wing = float(data.loc["cl_alpha_wing", 0])
            cm_0_wing = float(data.loc["cm_0_wing", 0])
            cm_alpha_wing = float(data.loc["cm_alpha_wing", 0])
            y_vector = np.array([float(i) for i in data.loc["y_vector", 0][1:-2].split(',')])
            cl_vector = np.array([float(i) for i in data.loc["cl_vector", 0][1:-2].split(',')])
            cl_0_htp = float(data.loc["cl_0_htp", 0])
            cl_alpha_htp = float(data.loc["cl_alpha_htp", 0])
            cm_0_htp = float(data.loc["cm_0_htp", 0])
            cm_alpha_htp = float(data.loc["cm_alpha_htp", 0])
            coef_k_wing = float(data.loc["coef_k_wing", 0])
            coef_k_htp = float(data.loc["coef_k_htp", 0])

        # Save and clean-up ----------------------------------------------------------------------------
        # Defining outputs
        if self.options["low_speed_aero"]:
            outputs['data:aerodynamics:wing:low_speed:CL0_clean'] = cl_0_wing
            outputs['data:aerodynamics:wing:low_speed:CL_alpha'] = cl_alpha_wing
            outputs['data:aerodynamics:wing:low_speed:CM0_clean'] = cm_0_wing
            outputs['data:aerodynamics:wing:low_speed:CM_alpha'] = cm_alpha_wing
            outputs['data:aerodynamics:wing:low_speed:Y_vector'] = y_vector
            outputs['data:aerodynamics:wing:low_speed:CL_vector'] = cl_vector
            outputs["data:aerodynamics:wing:low_speed:induced_drag_coefficient"] = coef_k_wing
            outputs['data:aerodynamics:horizontal_tail:low_speed:CL0'] = cl_0_htp
            outputs['data:aerodynamics:horizontal_tail:low_speed:CL_alpha'] = cl_alpha_htp
            outputs['data:aerodynamics:horizontal_tail:low_speed:CM0'] = cm_0_htp
            outputs['data:aerodynamics:horizontal_tail:low_speed:CM_alpha'] = cm_alpha_htp
            outputs["data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"] = coef_k_htp
        else:
            outputs['data:aerodynamics:wing:cruise:CL0_clean'] = cl_0_wing
            outputs['data:aerodynamics:wing:cruise:CL_alpha'] = cl_alpha_wing
            outputs['data:aerodynamics:wing:cruise:CM0_clean'] = cm_0_wing
            outputs['data:aerodynamics:wing:cruise:CM_alpha'] = cm_alpha_wing
            outputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"] = coef_k_wing
            outputs['data:aerodynamics:horizontal_tail:cruise:CL0'] = cl_0_htp
            outputs['data:aerodynamics:horizontal_tail:cruise:CL_alpha'] = cl_alpha_htp
            outputs['data:aerodynamics:horizontal_tail:cruise:CM0'] = cm_0_htp
            outputs['data:aerodynamics:horizontal_tail:cruise:CM_alpha'] = cm_alpha_htp
            outputs["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"] = coef_k_htp

    @staticmethod
    def _read_lod_file(tmp_result_file_path: str) -> Tuple[float, float, List, List, float, float]:
        """
        Collect data from .lod file
        """
        totals = False
        cl_wing = 0.0
        cm_wing = 0.0
        y_vector = []
        cl_vector = []
        cl_htp = 0.0
        cm_htp = 0.0
        with open(tmp_result_file_path, 'r') as lf:
            data = lf.readlines()
            for i in range(len(data)):
                line = data[i].split()
                line.append('**')
                if line[0] == '1':
                    y_vector.append(float(line[2]))
                    cl_vector.append(float(line[5]))
                if line[0] == 'Comp':
                    totals = True
                if totals:
                    cl_wing = float(data[i + 1].split()[5]) + float(data[i + 2].split()[5])  # sum CL left/right
                    cm_wing = float(data[i + 1].split()[12]) + float(data[i + 2].split()[12])  # sum CM left/right
                    # noinspection PyBroadException
                    try:
                        cl_htp = float(data[i + 3].split()[5]) + float(data[i + 4].split()[5])  # sum CL left/right
                        cm_htp = float(data[i + 3].split()[12]) + float(data[i + 4].split()[12])  # sum CM left/right
                    except:
                        pass
                    break

        return cl_wing, cm_wing, cl_vector, y_vector, cl_htp, cm_htp

    @staticmethod
    def _read_polar_file(tmp_result_file_path: str) -> float:
        """
        Collect oswald from .polar file
        """

        with open(tmp_result_file_path, 'r') as hf:
            line = hf.readlines()
            oswald = float(line[1][100:110].replace(' ', ''))

        return oswald


    @staticmethod
    def _create_tmp_directory() -> TemporaryDirectory:
        
        """Provide temporary directory for calculation."""
        
        for tmp_base_path in [None, pth.join(str(Path.home()), ".fast")]:
            if tmp_base_path is not None:
                os.makedirs(tmp_base_path, exist_ok=True)
            tmp_directory = tempfile.TemporaryDirectory(prefix="x", dir=tmp_base_path)
            break
            
        return tmp_directory
