"""
    Estimation of cl/cm/oswald aero coefficients using OPENVSP
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
import numpy as np
import warnings
import math
import pandas as pd
from openmdao.components.external_code_comp import ExternalCodeComp
from openmdao.utils.file_wrap import InputFileGenerator
import os
import os.path as pth
from pathlib import Path
import tempfile
from tempfile import TemporaryDirectory

from fastoad.utils.physics import Atmosphere
from fastoad.utils.resource_management.copy import copy_resource, copy_resource_folder

from ... import resources
from . import resources as local_resources
from . import openvsp3201
from ...constants import SPAN_MESH_POINT, MACH_NB_PTS

DEFAULT_WING_AIRFOIL = "naca23012.af"
DEFAULT_HTP_AIRFOIL = "naca0012.af"
INPUT_WING_SCRIPT = "wing_openvsp.vspscript"
INPUT_HTP_SCRIPT = "ht_openvsp.vspscript"
INPUT_AIRCRAFT_SCRIPT = "wing_ht_openvsp.vspscript"
STDERR_FILE_NAME = "vspaero_calc.err"
VSPSCRIPT_EXE_NAME = "vspscript.exe"
VSPAERO_EXE_NAME = "vspaero.exe"


class OPENVSPSimpleGeometry(ExternalCodeComp):

    def initialize(self):
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare("openvsp_exe_path", default="", types=str, allow_none=True)
        self.options.declare("wing_airfoil_file", default=DEFAULT_WING_AIRFOIL, types=str, allow_none=True)
        self.options.declare("htp_airfoil_file", default=DEFAULT_HTP_AIRFOIL, types=str, allow_none=True)
        
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
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:z:from_wingMAC25", val=np.nan, units="m")

    def check_config(self, logger):
        # let void to avoid logger error on "The command cannot be empty"
        pass


    def compute_cl_alpha_aircraft(self, inputs, outputs, altitude, mach, aoa_angle):
        """
        Function that perform a complete calculation of aerodynamic parameters under OpenVSP and returns only the
        cl_alpha_aircraft parameter.

        """
        _, cl_alpha_wing, _, _, _, _, _, _, cl_alpha_htp, _, _, _, _ = self.compute_aero_coef(
            inputs, outputs, altitude, mach, aoa_angle)
        return float(cl_alpha_wing + cl_alpha_htp)

    def compute_cl_alpha_mach(self, inputs, outputs, aoa_angle, altitude, cruise_mach):
        """
        Function that performs multiple run of OpenVSP to get an interpolation of Cl_alpha as a function of Mach
        for later use in the computation of the V-n diagram
        """
        mach_interp = np.log(np.linspace(np.exp(0.15), np.exp(1.55 * cruise_mach), MACH_NB_PTS))
        cl_alpha_interp = np.zeros(np.size(mach_interp))
        for idx in range(len(mach_interp)):
            cl_alpha_interp[idx] = self.compute_cl_alpha_aircraft(inputs, outputs, altitude, mach_interp[idx],
                                                                  aoa_angle)
            test = 1.0

        # We add the case were M=0, for thoroughness and since we are in an incompressible flow, the Cl_alpha is
        # approximately the same as for the first Mach of the interpolation
        mach_interp = np.insert(mach_interp, 0, 0.)
        cl_alpha_inc = cl_alpha_interp[0]
        cl_alpha_interp = np.insert(cl_alpha_interp, 0, cl_alpha_inc)

        return mach_interp, cl_alpha_interp

    def compute_aero_coef(self, inputs, outputs, altitude, mach, aoa_angle):
        """
        Function that computes in OpenVSP environment all the aerodynamic parameters @0° and aoa_angle and calculate
        the associated derivatives.

        @param inputs: inputs parameters defined within FAST-OAD-GA
        @param outputs: outputs parameters defined within FAST-OAD-GA
        @param altitude: altitude for aerodynamic calculation in meters
        @param mach: air speed expressed in mach
        @param aoa_angle: air speed angle of attack with respect to aircraft
        @return: cl_0_wing, cl_alpha_wing, cm_0_wing, y_vector_wing, cl_vector_wing, coef_k_wing, cl_0_htp,\
               cl_X_htp, cl_alpha_htp, cl_alpha_htp_isolated, y_vector_htp, cl_vector_htp, coef_k_htp parameters.
        """

        # Fix mach number of digits to consider similar results
        mach = round(float(mach) * 1e3) / 1e3

        # Get inputs necessary to define global geometry
        sref_wing = float(inputs['data:geometry:wing:area'])
        sref_htp = float(inputs['data:geometry:horizontal_tail:area'])
        area_ratio = sref_htp / sref_wing
        sweep25_wing = float(inputs["data:geometry:wing:sweep_25"])
        taper_ratio_wing = float(inputs["data:geometry:wing:taper_ratio"])
        aspect_ratio_wing = float(inputs["data:geometry:wing:aspect_ratio"])
        sweep25_htp = float(inputs["data:geometry:horizontal_tail:sweep_25"])
        aspect_ratio_htp = float(inputs["data:geometry:horizontal_tail:aspect_ratio"])
        taper_ratio_htp = float(inputs["data:geometry:horizontal_tail:taper_ratio"])
        geometry_set = np.around(np.array(
            [sweep25_wing, taper_ratio_wing, aspect_ratio_wing, sweep25_htp, taper_ratio_htp, aspect_ratio_htp, mach,
             area_ratio]), decimals=6)

        # Search if results already exist:
        result_folder_path = self.options["result_folder_path"]
        result_file_path = None
        saved_area_ratio = 1.0
        if result_folder_path != "":
            result_file_path, saved_area_ratio = self.search_results(result_folder_path, geometry_set)


        # If no result saved for that geometry under this mach condition, computation is done
        if result_file_path is None:

            # Create result folder first (if it must fail, let it fail as soon as possible)
            if result_folder_path != "":
                if not os.path.exists(result_folder_path):
                    os.makedirs(pth.join(result_folder_path), exist_ok=True)

            # Save the geometry (result_file_path is None entering the function)
            if self.options["result_folder_path"] != "":
                result_file_path = self.save_geometry(result_folder_path, geometry_set)

            # Compute wing alone @ 0°/X° angle of attack
            wing_0 = self.compute_wing(inputs, outputs, altitude, mach, 0.0)
            wing_X = self.compute_wing(inputs, outputs, altitude, mach, aoa_angle)

            # Compute complete aircraft @ 0°/X° angle of attack
            _, htp_0, _ = self.compute_aircraft(inputs, outputs, altitude, mach, 0.0)
            _, htp_X, _ = self.compute_aircraft(inputs, outputs, altitude, mach, aoa_angle)

            # Compute isolated HTP @ 0°/X° angle of attack
            htp_0_isolated = self.compute_isolated_htp(inputs, outputs, altitude, mach, 0.0)
            htp_X_isolated = self.compute_isolated_htp(inputs, outputs, altitude, mach, aoa_angle)

            # Post-process wing data -----------------------------------------------------------------------------------
            width_max = inputs["data:geometry:fuselage:maximum_width"]
            span_wing = inputs['data:geometry:wing:span']
            k_fus = 1 + 0.025 * width_max / span_wing - 0.025 * (width_max / span_wing) ** 2
            cl_0_wing = float(wing_0["cl"] * k_fus)
            cl_X_wing = float(wing_X["cl"] * k_fus)
            cm_0_wing = float(wing_0["cm"] * k_fus)
            cl_alpha_wing = (cl_X_wing - cl_0_wing) / (aoa_angle * math.pi / 180)
            y_vector_wing = wing_0["y_vector"]
            cl_vector_wing = (np.array(wing_0["cl_vector"]) * k_fus).tolist()
            k_fus = 1 - 2 * (width_max / span_wing) ** 2  # Fuselage correction
            # Full aircraft correction: Wing lift is 105% of total lift, so: CDi = (CL*1.05)^2/(piAe) -> e' = e/1.05^2
            coef_e = float(wing_X["coef_e"] * k_fus / 1.05 ** 2)
            coef_k_wing = float(1. / (math.pi * span_wing ** 2 / sref_wing * coef_e))

            # Post-process HTP-aircraft data ---------------------------------------------------------------------------
            cl_0_htp = float(htp_0["cl"])
            cl_X_htp = float(htp_X["cl"])
            cl_alpha_htp = float((cl_X_htp - cl_0_htp) / (aoa_angle * math.pi / 180))
            coef_k_htp = float(htp_X["cdi"]) / cl_X_htp ** 2
            y_vector_htp = htp_X["y_vector"]
            cl_vector_htp = (np.array(htp_X["cl_vector"]) * area_ratio).tolist()

            # Post-process HTP-isolated data ---------------------------------------------------------------------------
            cl_alpha_htp_isolated = float(htp_X_isolated["cl"] - htp_0_isolated["cl"]) * area_ratio\
                                    / (aoa_angle * math.pi / 180)

            # Resize vectors -------------------------------------------------------------------------------------------
            if SPAN_MESH_POINT < len(y_vector_wing):
                y_interp = np.linspace(y_vector_wing[0], y_vector_wing[-1], SPAN_MESH_POINT)
                cl_vector_wing = np.interp(y_interp, y_vector_wing, cl_vector_wing)
                y_vector_wing = y_interp
                warnings.warn("Defined maximum span mesh in fast aerodynamics\\constants.py exceeded!")
            else:
                additional_zeros = list(np.zeros(SPAN_MESH_POINT - len(y_vector_wing)))
                y_vector_wing.extend(additional_zeros)
                cl_vector_wing.extend(additional_zeros)
            if SPAN_MESH_POINT < len(y_vector_htp):
                y_interp = np.linspace(y_vector_htp[0], y_vector_htp[-1], SPAN_MESH_POINT)
                cl_vector_htp = np.interp(y_interp, y_vector_htp, cl_vector_htp)
                y_vector_htp = y_interp
                warnings.warn("Defined maximum span mesh in fast aerodynamics\\constants.py exceeded!")
            else:
                additional_zeros = list(np.zeros(SPAN_MESH_POINT - len(y_vector_htp)))
                y_vector_htp.extend(additional_zeros)
                cl_vector_htp.extend(additional_zeros)

            # Save results to defined path -----------------------------------------------------------------------------
            if self.options["result_folder_path"] != "":
                results = [cl_0_wing, cl_alpha_wing, cm_0_wing, y_vector_wing, cl_vector_wing, coef_k_wing, cl_0_htp,
                           cl_X_htp, cl_alpha_htp, cl_alpha_htp_isolated, y_vector_htp, cl_vector_htp, coef_k_htp]
                self.save_results(result_file_path, results)

        # Else retrieved results are used, eventually adapted with new area ratio
        else:
            # Read values from result file -----------------------------------------------------------------------------
            data = self.read_results(result_file_path)
            cl_0_wing = float(data.loc["cl_0_wing", 0])
            cl_alpha_wing = float(data.loc["cl_alpha_wing", 0])
            cm_0_wing = float(data.loc["cm_0_wing", 0])
            y_vector_wing = np.array([float(i) for i in data.loc["y_vector_wing", 0][1:-2].split(',')])
            cl_vector_wing = np.array([float(i) for i in data.loc["cl_vector_wing", 0][1:-2].split(',')])
            coef_k_wing = float(data.loc["coef_k_wing", 0])
            cl_0_htp = float(data.loc["cl_0_htp", 0]) * (area_ratio / saved_area_ratio)
            cl_X_htp = float(data.loc["cl_X_htp", 0]) * (area_ratio / saved_area_ratio)
            cl_alpha_htp = float(data.loc["cl_alpha_htp", 0]) * (area_ratio / saved_area_ratio)
            cl_alpha_htp_isolated = float(data.loc["cl_alpha_htp_isolated", 0])
            y_vector_htp = np.array([float(i) for i in data.loc["y_vector_htp", 0][1:-2].split(',')])
            cl_vector_htp = np.array([float(i) for i in data.loc["cl_vector_htp", 0][1:-2].split(',')])
            coef_k_htp = float(data.loc["coef_k_htp", 0]) * (area_ratio / saved_area_ratio)

        return cl_0_wing, cl_alpha_wing, cm_0_wing, y_vector_wing, cl_vector_wing, coef_k_wing, cl_0_htp,\
               cl_X_htp, cl_alpha_htp, cl_alpha_htp_isolated, y_vector_htp, cl_vector_htp, coef_k_htp

    def compute_wing(self, inputs, outputs, altitude, mach, aoa_angle):
        """
        Function that computes in OpenVSP environment the wing alone and returns the different aerodynamic parameters.

        @param inputs: inputs parameters defined within FAST-OAD-GA
        @param outputs: outputs parameters defined within FAST-OAD-GA
        @param altitude: altitude for aerodynamic calculation in meters
        @param mach: air speed expressed in mach
        @param aoa_angle: air speed angle of attack with respect to wing (degree)
        @return: wing dictionary including aero parameters as keys: y_vector, cl_vector, cd_vector, cm_vector, cl
        cdi, cm, coef_e
        """

        # STEP 1/XX - DEFINE OR CALCULATE INPUT DATA FOR AERODYNAMIC EVALUATION ########################################
        ################################################################################################################

        # Get inputs (and calculate missing ones)
        sref_wing = float(inputs['data:geometry:wing:area'])
        x0_wing = inputs["data:geometry:wing:MAC:leading_edge:x:local"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        y1_wing = width_max / 2.0
        y2_wing = inputs["data:geometry:wing:root:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        sweep_0_wing = inputs["data:geometry:wing:sweep_0"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        span_wing = inputs['data:geometry:wing:span']
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        # Compute remaining inputs
        atm = Atmosphere(altitude, altitude_in_feet=False)
        x_wing = fa_length - x0_wing - 0.25 * l0_wing
        z_wing = -(height_max - 0.12 * l2_wing) * 0.5
        span2_wing = y4_wing - y2_wing
        rho = atm.density
        v_inf = max(atm.speed_of_sound * mach, 0.01)  # avoid V=0 m/s crashes
        reynolds = v_inf * l0_wing / atm.kinematic_viscosity

        # STEP 2/XX - DEFINE WORK DIRECTORY, COPY RESOURCES AND CREATE COMMAND BATCH ###################################
        ################################################################################################################

        # If a folder path is specified for openvsp .exe, it becomes working directory (target), if not temporary folder
        # is created
        if self.options["openvsp_exe_path"]:
            target_directory = pth.abspath(self.options["openvsp_exe_path"])
        else:
            tmp_directory = self._create_tmp_directory()
            target_directory = tmp_directory.name
        # Define the list of necessary input files: geometry script and foil file for both wing/HTP
        input_file_list = [pth.join(target_directory, INPUT_WING_SCRIPT),
                           pth.join(target_directory, self.options['wing_airfoil_file'])]
        self.options["external_input_files"] = input_file_list
        # Define standard error file by default to avoid error code return
        self.stderr = pth.join(target_directory, STDERR_FILE_NAME)
        # Copy resource in working (target) directory
        # noinspection PyTypeChecker
        copy_resource_folder(openvsp3201, target_directory)
        # noinspection PyTypeChecker
        copy_resource(resources, self.options['wing_airfoil_file'], target_directory)
        # Create corresponding .bat files (one for each geometry configuration)
        self.options["command"] = [pth.join(target_directory, 'vspscript.bat')]
        batch_file = open(self.options["command"][0], "w+")
        batch_file.write("@echo off\n")
        command = pth.join(target_directory, VSPSCRIPT_EXE_NAME) + ' -script ' \
                  + pth.join(target_directory, INPUT_WING_SCRIPT) + ' >nul 2>nul\n'
        batch_file.write(command)
        batch_file.close()

        # STEP 3/XX - OPEN THE TEMPLATE SCRIPT FOR GEOMETRY GENERATION, MODIFY VALUES AND SAVE TO WORKDIR ##############
        ################################################################################################################

        output_file_list = [pth.join(target_directory, INPUT_WING_SCRIPT.replace('.vspscript', '_DegenGeom.csv'))]
        parser = InputFileGenerator()
        with path(local_resources, INPUT_WING_SCRIPT) as input_template_path:
            parser.set_template_file(str(input_template_path))
            parser.set_generated_file(input_file_list[0])
            # Modify wing parameters
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
            parser.transfer_var('\"' + input_file_list[1].replace('\\', '/') + '\"', 0, 3)
            parser.mark_anchor("airfoil_1_file")
            parser.transfer_var('\"' + input_file_list[1].replace('\\', '/') + '\"', 0, 3)
            parser.mark_anchor("airfoil_2_file")
            parser.transfer_var('\"' + input_file_list[1].replace('\\', '/') + '\"', 0, 3)
            parser.mark_anchor("csv_file")
            csv_name = output_file_list[0]
            parser.transfer_var('\"' + csv_name.replace('\\', '/') + '\"', 0, 3)
            parser.generate()

        # STEP 4/XX - RUN BATCH TO GENERATE GEOMETRY .CSV FILE #########################################################
        ################################################################################################################

        self.options["external_output_files"] = output_file_list
        super().compute(inputs, outputs)

        # STEP 5/XX - DEFINE NEW INPUT/OUTPUT FILES LIST AND CREATE BATCH FOR VLM COMPUTATION ##########################
        ################################################################################################################

        input_file_list = output_file_list
        input_file_list.append(input_file_list[0].replace('.csv', '.vspaero'))
        output_file_list = [input_file_list[0].replace('.csv', '.lod'), input_file_list[0].replace('.csv', '.polar')]
        self.options["external_input_files"] = input_file_list
        self.options["external_output_files"] = output_file_list
        self.options["command"] = [pth.join(target_directory, 'vspaero.bat')]
        batch_file = open(self.options["command"][0], "w+")
        batch_file.write("@echo off\n")
        command = pth.join(target_directory, VSPAERO_EXE_NAME) + ' ' \
                  + input_file_list[1].replace('.vspaero', '') + ' >nul 2>nul\n'
        batch_file.write(command)
        batch_file.close()

        # STEP 6/XX - OPEN THE TEMPLATE VSPAERO FOR COMPUTATION, MODIFY VALUES AND SAVE TO WORKDIR #####################
        ################################################################################################################

        parser = InputFileGenerator()
        template_file = pth.split(input_file_list[1])[1]
        with path(local_resources, template_file) as input_template_path:
            parser.set_template_file(str(input_template_path))
            parser.set_generated_file(input_file_list[1])
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
            parser.transfer_var(float(aoa_angle), 0, 3)
            parser.mark_anchor("Vinf")
            parser.transfer_var(float(v_inf), 0, 3)
            parser.mark_anchor("Rho")
            parser.transfer_var(float(rho), 0, 3)
            parser.mark_anchor("ReCref")
            parser.transfer_var(float(reynolds), 0, 3)
            parser.generate()

        # STEP 7/XX - RUN BATCH TO GENERATE AERO OUTPUT FILES (.lod, .polar...) ########################################
        ################################################################################################################

        super().compute(inputs, outputs)

        # STEP 8/XX - READ FILES, RETURN RESULTS (AND CLEAR TEMPORARY WORKDIR) #########################################
        ################################################################################################################

        # Open .lod file and extract data
        wing_y_vect = []
        wing_cl_vect = []
        wing_cd_vect = []
        wing_cm_vect = []
        with open(output_file_list[0], 'r') as lf:
            data = lf.readlines()
            for i in range(len(data)):
                line = data[i].split()
                line.append('**')
                if line[0] == '1':
                    wing_y_vect.append(float(line[2]))
                    wing_cl_vect.append(float(line[5]))
                    wing_cd_vect.append(float(line[6]))
                    wing_cm_vect.append(float(line[12]))
                if line[0] == 'Comp':
                    cl_wing = float(data[i + 1].split()[5]) + float(data[i + 2].split()[5])  # sum CL left/right
                    cdi_wing = float(data[i + 1].split()[6]) + float(data[i + 2].split()[6])  # sum CDi left/right
                    cm_wing = float(data[i + 1].split()[12]) + float(data[i + 2].split()[12])  # sum CM left/right
                    break
        # Open .polar file and extract data
        with open(output_file_list[1], 'r') as lf:
            data = lf.readlines()
            wing_e = float(data[1].split()[10])
        # Delete temporary directory
        if not (self.options["openvsp_exe_path"]):
            # noinspection PyUnboundLocalVariable
            tmp_directory.cleanup()
        # Return values
        wing = {'y_vector': wing_y_vect,
                'cl_vector': wing_cl_vect,
                'cd_vector': wing_cd_vect,
                'cm_vector': wing_cm_vect,
                'cl': cl_wing,
                'cdi': cdi_wing,
                'cm': cm_wing,
                'coef_e': wing_e}
        return wing

    def compute_isolated_htp(self, inputs, outputs, altitude, mach, aoa_angle):
        """
        Function that computes in OpenVSP environment the HTP alone and returns the different aerodynamic parameters.

        @param inputs: inputs parameters defined within FAST-OAD-GA
        @param outputs: outputs parameters defined within FAST-OAD-GA
        @param altitude: altitude for aerodynamic calculation in meters
        @param mach: air speed expressed in mach
        @param aoa_angle: air speed angle of attack with respect to htp (degree)
        @return: htp dictionary including aero parameters as keys: y_vector, cl_vector, cd_vector, cm_vector, cl
        cdi, cm, coef_e
        """

        # STEP 1/XX - DEFINE OR CALCULATE INPUT DATA FOR AERODYNAMIC EVALUATION ########################################
        ################################################################################################################

        # Get inputs (and calculate missing ones)
        sref_htp = float(inputs['data:geometry:horizontal_tail:area'])
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        sweep_25_htp = inputs["data:geometry:horizontal_tail:sweep_25"]
        semi_span_htp = inputs["data:geometry:horizontal_tail:span"] / 2.0
        root_chord_htp = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord_htp = inputs["data:geometry:horizontal_tail:tip:chord"]
        lp_htp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        l0_htp = inputs["data:geometry:horizontal_tail:MAC:length"]
        x0_htp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:local"]
        height_htp = inputs["data:geometry:horizontal_tail:z:from_wingMAC25"]
        # Compute remaining inputs
        atm = Atmosphere(altitude, altitude_in_feet=False)
        x_htp = fa_length + lp_htp - x0_htp - 0.25 * l0_htp
        z_htp = -(height_max - 0.12 * l0_htp) * 0.5 - height_htp
        rho = atm.density
        v_inf = max(atm.speed_of_sound * mach, 0.01)  # avoid V=0 m/s crashes
        reynolds = v_inf * l0_htp / atm.kinematic_viscosity

        # STEP 2/XX - DEFINE WORK DIRECTORY, COPY RESOURCES AND CREATE COMMAND BATCH ###################################
        ################################################################################################################

        # If a folder path is specified for openvsp .exe, it becomes working directory (target), if not temporary folder
        # is created
        if self.options["openvsp_exe_path"]:
            target_directory = pth.abspath(self.options["openvsp_exe_path"])
        else:
            tmp_directory = self._create_tmp_directory()
            target_directory = tmp_directory.name
        # Define the list of necessary input files: geometry script and foil file for both wing/HTP
        input_file_list = [pth.join(target_directory, INPUT_HTP_SCRIPT),
                           pth.join(target_directory, self.options['htp_airfoil_file'])]
        self.options["external_input_files"] = input_file_list
        # Define standard error file by default to avoid error code return
        self.stderr = pth.join(target_directory, STDERR_FILE_NAME)
        # Copy resource in working (target) directory
        # noinspection PyTypeChecker
        copy_resource_folder(openvsp3201, target_directory)
        # noinspection PyTypeChecker
        copy_resource(resources, self.options['htp_airfoil_file'], target_directory)
        # Create corresponding .bat files (one for each geometry configuration)
        self.options["command"] = [pth.join(target_directory, 'vspscript.bat')]
        batch_file = open(self.options["command"][0], "w+")
        batch_file.write("@echo off\n")
        command = pth.join(target_directory, VSPSCRIPT_EXE_NAME) + ' -script ' \
                  + pth.join(target_directory, INPUT_HTP_SCRIPT) + ' >nul 2>nul\n'
        batch_file.write(command)
        batch_file.close()

        # STEP 3/XX - OPEN THE TEMPLATE SCRIPT FOR GEOMETRY GENERATION, MODIFY VALUES AND SAVE TO WORKDIR ##############
        ################################################################################################################

        output_file_list = [pth.join(target_directory, INPUT_HTP_SCRIPT.replace('.vspscript', '_DegenGeom.csv'))]
        parser = InputFileGenerator()
        with path(local_resources, INPUT_HTP_SCRIPT) as input_template_path:
            parser.set_template_file(str(input_template_path))
            parser.set_generated_file(input_file_list[0])
            # Modify htp parameters
            parser.mark_anchor("x_htp")
            parser.transfer_var(float(x_htp), 0, 5)
            parser.mark_anchor("z_htp")
            parser.transfer_var(float(z_htp), 0, 5)
            parser.mark_anchor("semi_span_htp")
            parser.transfer_var(float(semi_span_htp), 0, 5)
            parser.mark_anchor("root_chord_htp")
            parser.transfer_var(float(root_chord_htp), 0, 5)
            parser.mark_anchor("tip_chord_htp")
            parser.transfer_var(float(tip_chord_htp), 0, 5)
            parser.mark_anchor("sweep_25_htp")
            parser.transfer_var(float(sweep_25_htp), 0, 5)
            parser.mark_anchor("airfoil_0_file")
            parser.transfer_var('\"' + input_file_list[1].replace('\\', '/') + '\"', 0, 3)
            parser.mark_anchor("airfoil_1_file")
            parser.transfer_var('\"' + input_file_list[1].replace('\\', '/') + '\"', 0, 3)
            parser.mark_anchor("csv_file")
            csv_name = output_file_list[0]
            parser.transfer_var('\"' + csv_name.replace('\\', '/') + '\"', 0, 3)
            parser.generate()

        # STEP 4/XX - RUN BATCH TO GENERATE GEOMETRY .CSV FILE #########################################################
        ################################################################################################################

        self.options["external_output_files"] = output_file_list
        super().compute(inputs, outputs)

        # STEP 5/XX - DEFINE NEW INPUT/OUTPUT FILES LIST AND CREATE BATCH FOR VLM COMPUTATION ##########################
        ################################################################################################################

        input_file_list = output_file_list
        input_file_list.append(input_file_list[0].replace('.csv', '.vspaero'))
        output_file_list = [input_file_list[0].replace('.csv', '.lod'), input_file_list[0].replace('.csv', '.polar')]
        self.options["external_input_files"] = input_file_list
        self.options["external_output_files"] = output_file_list
        self.options["command"] = [pth.join(target_directory, 'vspaero.bat')]
        batch_file = open(self.options["command"][0], "w+")
        batch_file.write("@echo off\n")
        command = pth.join(target_directory, VSPAERO_EXE_NAME) + ' ' \
                  + input_file_list[1].replace('.vspaero', '') + ' >nul 2>nul\n'
        batch_file.write(command)
        batch_file.close()

        # STEP 6/XX - OPEN THE TEMPLATE VSPAERO FOR COMPUTATION, MODIFY VALUES AND SAVE TO WORKDIR #####################
        ################################################################################################################

        parser = InputFileGenerator()
        template_file = pth.split(input_file_list[1])[1]
        with path(local_resources, template_file) as input_template_path:
            parser.set_template_file(str(input_template_path))
            parser.set_generated_file(input_file_list[1])
            parser.reset_anchor()
            parser.mark_anchor("Sref")
            parser.transfer_var(float(sref_htp), 0, 3)
            parser.mark_anchor("Cref")
            parser.transfer_var(float(l0_htp), 0, 3)
            parser.mark_anchor("Bref")
            parser.transfer_var(float(2. * semi_span_htp), 0, 3)
            parser.mark_anchor("X_cg")
            parser.transfer_var(float(fa_length + lp_htp), 0, 3)
            parser.mark_anchor("Mach")
            parser.transfer_var(float(mach), 0, 3)
            parser.mark_anchor("AOA")
            parser.transfer_var(float(aoa_angle), 0, 3)
            parser.mark_anchor("Vinf")
            parser.transfer_var(float(v_inf), 0, 3)
            parser.mark_anchor("Rho")
            parser.transfer_var(float(rho), 0, 3)
            parser.mark_anchor("ReCref")
            parser.transfer_var(float(reynolds), 0, 3)
            parser.generate()

        # STEP 7/XX - RUN BATCH TO GENERATE AERO OUTPUT FILES (.lod, .polar...) ########################################
        ################################################################################################################

        super().compute(inputs, outputs)

        # STEP 8/XX - READ FILES, RETURN RESULTS (AND CLEAR TEMPORARY WORKDIR) #########################################
        ################################################################################################################

        # Open .lod file and extract data
        htp_y_vect = []
        htp_cl_vect = []
        htp_cd_vect = []
        htp_cm_vect = []
        with open(output_file_list[0], 'r') as lf:
            data = lf.readlines()
            for i in range(len(data)):
                line = data[i].split()
                line.append('**')
                if line[0] == '1':
                    htp_y_vect.append(float(line[2]))
                    htp_cl_vect.append(float(line[5]))
                    htp_cd_vect.append(float(line[6]))
                    htp_cm_vect.append(float(line[12]))
                if line[0] == 'Comp':
                    cl_htp = float(data[i + 1].split()[5]) + float(data[i + 2].split()[5])  # sum CL left/right
                    cdi_htp = float(data[i + 1].split()[6]) + float(data[i + 2].split()[6])  # sum CDi left/right
                    cm_htp = float(data[i + 1].split()[12]) + float(data[i + 2].split()[12])  # sum CM left/right
                    break
        # Open .polar file and extract data
        with open(output_file_list[1], 'r') as lf:
            data = lf.readlines()
            htp_e = float(data[1].split()[10])
        # Delete temporary directory
        if not (self.options["openvsp_exe_path"]):
            # noinspection PyUnboundLocalVariable
            tmp_directory.cleanup()
        # Return values
        htp = {'y_vector': htp_y_vect,
                'cl_vector': htp_cl_vect,
                'cd_vector': htp_cd_vect,
                'cm_vector': htp_cm_vect,
                'cl': cl_htp,
                'cdi': cdi_htp,
                'cm': cm_htp,
                'coef_e': htp_e}
        return htp


    def compute_aircraft(self, inputs, outputs, altitude, mach, aoa_angle):
        """
        Function that computes in OpenVSP environment the complete aircraft (considering wing and horizontal tail plan)
        and returns the different aerodynamic parameters. The downwash is done by OpenVSP considering far field.

        @param inputs: inputs parameters defined within FAST-OAD-GA
        @param outputs: outputs parameters defined within FAST-OAD-GA
        @param altitude: altitude for aerodynamic calculation in meters
        @param mach: air speed expressed in mach
        @param aoa_angle: air speed angle of attack with respect to aircraft
        @return: wing/htp and aircraft dictionaries including their respective aerodynamic coefficients
        """

        # STEP 1/XX - DEFINE OR CALCULATE INPUT DATA FOR AERODYNAMIC EVALUATION ########################################
        ################################################################################################################

        # Get inputs (and calculate missing ones)
        sref_wing = float(inputs['data:geometry:wing:area'])
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
        atm = Atmosphere(altitude, altitude_in_feet=False)
        x_wing = fa_length-x0_wing-0.25*l0_wing
        z_wing = -(height_max - 0.12*l2_wing)*0.5
        span2_wing = y4_wing - y2_wing
        distance_htp = fa_length + lp_htp - 0.25 * l0_htp - x0_htp
        rho = atm.density
        v_inf = max(atm.speed_of_sound * mach, 0.01)  # avoid V=0 m/s crashes
        reynolds = v_inf * l0_wing / atm.kinematic_viscosity

        # STEP 2/XX - DEFINE WORK DIRECTORY, COPY RESOURCES AND CREATE COMMAND BATCH ###################################
        ################################################################################################################

        # If a folder path is specified for openvsp .exe, it becomes working directory (target), if not temporary folder
        # is created
        if self.options["openvsp_exe_path"]:
            target_directory = pth.abspath(self.options["openvsp_exe_path"])
        else:
            tmp_directory = self._create_tmp_directory()
            target_directory = tmp_directory.name
        # Define the list of necessary input files: geometry script and foil file for both wing/HTP
        input_file_list = [pth.join(target_directory, INPUT_AIRCRAFT_SCRIPT),
                           pth.join(target_directory, self.options['wing_airfoil_file']),
                           pth.join(target_directory, self.options['htp_airfoil_file'])]
        self.options["external_input_files"] = input_file_list
        # Define standard error file by default to avoid error code return
        self.stderr = pth.join(target_directory, STDERR_FILE_NAME)
        # Copy resource in working (target) directory
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
        command = pth.join(target_directory, VSPSCRIPT_EXE_NAME) + ' -script ' \
                  + pth.join(target_directory, INPUT_AIRCRAFT_SCRIPT) + ' >nul 2>nul\n'
        batch_file.write(command)
        batch_file.close()

        # STEP 3/XX - OPEN THE TEMPLATE SCRIPT FOR GEOMETRY GENERATION, MODIFY VALUES AND SAVE TO WORKDIR ##############
        ################################################################################################################

        output_file_list = [pth.join(target_directory, INPUT_AIRCRAFT_SCRIPT.replace('.vspscript', '_DegenGeom.csv'))]
        parser = InputFileGenerator()
        with path(local_resources, INPUT_AIRCRAFT_SCRIPT) as input_template_path:
            parser.set_template_file(str(input_template_path))
            parser.set_generated_file(input_file_list[0])
            # Modify wing parameters
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
            # Modify HTP parameters
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
            csv_name = output_file_list[0]
            parser.transfer_var('\"' + csv_name.replace('\\', '/') + '\"',  0, 3)
            parser.generate()

        # STEP 4/XX - RUN BATCH TO GENERATE GEOMETRY .CSV FILE #########################################################
        ################################################################################################################

        self.options["external_output_files"] = output_file_list
        super().compute(inputs, outputs)

        # STEP 5/XX - DEFINE NEW INPUT/OUTPUT FILES LIST AND CREATE BATCH FOR VLM COMPUTATION ##########################
        ################################################################################################################

        input_file_list = output_file_list
        input_file_list.append(input_file_list[0].replace('csv', 'vspaero'))
        output_file_list = [input_file_list[0].replace('csv', 'lod'), input_file_list[0].replace('csv', 'polar')]
        self.options["external_input_files"] = input_file_list
        self.options["external_output_files"] = output_file_list
        self.options["command"] = [pth.join(target_directory, 'vspaero.bat')]
        batch_file = open(self.options["command"][0], "w+")
        batch_file.write("@echo off\n")
        command = pth.join(target_directory, VSPAERO_EXE_NAME) + ' ' \
                  + input_file_list[1].replace('.vspaero', '') + ' >nul 2>nul\n'
        batch_file.write(command)
        batch_file.close()

        # STEP 6/XX - OPEN THE TEMPLATE VSPAERO FOR COMPUTATION, MODIFY VALUES AND SAVE TO WORKDIR #####################
        ################################################################################################################

        parser = InputFileGenerator()
        template_file = pth.split(input_file_list[1])[1]
        with path(local_resources, template_file) as input_template_path:
            parser.set_template_file(str(input_template_path))
            parser.set_generated_file(input_file_list[1])
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
            parser.transfer_var(float(aoa_angle), 0, 3)
            parser.mark_anchor("Vinf")
            parser.transfer_var(float(v_inf), 0, 3)
            parser.mark_anchor("Rho")
            parser.transfer_var(float(rho), 0, 3)
            parser.mark_anchor("ReCref")
            parser.transfer_var(float(reynolds), 0, 3)
            parser.generate()

        # STEP 7/XX - RUN BATCH TO GENERATE AERO OUTPUT FILES (.lod, .polar...) ########################################
        ################################################################################################################

        super().compute(inputs, outputs)

        # STEP 8/XX - READ FILES, RETURN RESULTS (AND CLEAR TEMPORARY WORKDIR) #########################################
        ################################################################################################################

        # Open .lod file and extract data
        wing_y_vect = []
        wing_cl_vect = []
        wing_cd_vect = []
        wing_cm_vect = []
        htp_y_vect = []
        htp_cl_vect = []
        htp_cd_vect = []
        htp_cm_vect = []
        with open(output_file_list[0], 'r') as lf:
            data = lf.readlines()
            for i in range(len(data)):
                line = data[i].split()
                line.append('**')
                if line[0] == '1':
                    wing_y_vect.append(float(line[2]))
                    wing_cl_vect.append(float(line[5]))
                    wing_cd_vect.append(float(line[6]))
                    wing_cm_vect.append(float(line[12]))
                elif line[0] == '3':
                    htp_y_vect.append(float(line[2]))
                    htp_cl_vect.append(float(line[5]))
                    htp_cd_vect.append(float(line[6]))
                    htp_cm_vect.append(float(line[12]))
                if line[0] == 'Comp':
                    cl_wing = float(data[i + 1].split()[5]) + float(data[i + 2].split()[5])  # sum CL left/right
                    cdi_wing = float(data[i + 1].split()[6]) + float(data[i + 2].split()[6])  # sum CDi left/right
                    cm_wing = float(data[i + 1].split()[12]) + float(data[i + 2].split()[12])  # sum CM left/right
                    cl_htp = float(data[i + 3].split()[5]) + float(data[i + 4].split()[5])  # sum CL left/right
                    cdi_htp = float(data[i + 3].split()[6]) + float(data[i + 4].split()[6])  # sum CDi left/right
                    cm_htp = float(data[i + 3].split()[12]) + float(data[i + 4].split()[12])  # sum CM left/right
                    break
        # Open .polar file and extract data
        with open(output_file_list[1], 'r') as lf:
            data = lf.readlines()
            aircraft_cl = float(data[1].split()[4])
            aircraft_cd0 = float(data[1].split()[5])
            aircraft_cdi = float(data[1].split()[6])
            aircraft_e = float(data[1].split()[10])
        # Delete temporary directory
        if not(self.options["openvsp_exe_path"]):
            # noinspection PyUnboundLocalVariable
            tmp_directory.cleanup()
        # Return values
        wing = {'y_vector': wing_y_vect,
                'cl_vector': wing_cl_vect,
                'cd_vector': wing_cd_vect,
                'cm_vector': wing_cm_vect,
                'cl': cl_wing,
                'cdi': cdi_wing,
                'cm': cm_wing}
        htp = {'y_vector': htp_y_vect,
               'cl_vector': htp_cl_vect,
               'cd_vector': htp_cd_vect,
               'cm_vector': htp_cm_vect,
               'cl': cl_htp,
               'cdi': cdi_htp,
               'cm': cm_htp}
        aircraft = {'cl': aircraft_cl,
                    'cd0': aircraft_cd0,
                    'cdi': aircraft_cdi,
                    'coef_e': aircraft_e}
        return wing, htp, aircraft


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
    def search_results(result_folder_path, geometry_set):

        if os.path.exists(result_folder_path):
            geometry_set_labels = ["sweep25_wing", "taper_ratio_wing", "aspect_ratio_wing", "sweep25_htp",
                                   "taper_ratio_htp", "aspect_ratio_htp", "mach", "area_ratio"]
            # If some results already stored search for corresponding geometry
            if pth.exists(pth.join(result_folder_path, "geometry_0.csv")):
                idx = 0
                while pth.exists(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv")):
                    if pth.exists(pth.join(result_folder_path, "openvsp_" + str(idx) + ".csv")):
                        data = pd.read_csv(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv"))
                        values = data.to_numpy()[:, 1].tolist()
                        labels = data.to_numpy()[:, 0].tolist()
                        data = pd.DataFrame(values, index=labels)
                        # noinspection PyBroadException
                        try:
                            if np.size(data.loc[geometry_set_labels[0:-1], 0].to_numpy()) == 7:
                                saved_set = np.around(data.loc[geometry_set_labels[0:-1], 0].to_numpy(), decimals=6)
                                if np.sum(saved_set == geometry_set[0:-1]) == 7:
                                    result_file_path = pth.join(result_folder_path, "openvsp_" + str(idx) + ".csv")
                                    saved_area_ratio = data.loc["area_ratio", 0]
                                    return result_file_path, saved_area_ratio
                        except:
                            break
                    idx += 1

        return None, 1.0


    @staticmethod
    def save_geometry(result_folder_path, geometry_set):

        # Save geometry if not already computed by finding first available index
        geometry_set_labels = ["sweep25_wing", "taper_ratio_wing", "aspect_ratio_wing", "sweep25_htp",
                               "taper_ratio_htp", "aspect_ratio_htp", "mach", "area_ratio"]
        data = pd.DataFrame(geometry_set, index=geometry_set_labels)
        idx = 0
        while pth.exists(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv")):
            idx += 1
        data.to_csv(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv"))
        result_file_path = pth.join(result_folder_path, "openvsp_" + str(idx) + ".csv")

        return result_file_path

    @staticmethod
    def save_results(result_file_path, results):

        labels = ["cl_0_wing", "cl_alpha_wing", "cm_0_wing", "y_vector_wing", "cl_vector_wing", "coef_k_wing",
                  "cl_0_htp", "cl_X_htp", "cl_alpha_htp", "cl_alpha_htp_isolated", "y_vector_htp", "cl_vector_htp",
                  "coef_k_htp"]
        data = pd.DataFrame(results, index=labels)
        data.to_csv(result_file_path)

    @staticmethod
    def read_results(result_file_path):

        data = pd.read_csv(result_file_path)
        values = data.to_numpy()[:, 1].tolist()
        labels = data.to_numpy()[:, 0].tolist()

        return pd.DataFrame(values, index=labels)
