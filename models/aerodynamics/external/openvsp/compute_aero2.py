"""
    Estimation of aero coefficients using OPENVSP
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

import math
import numpy as np
import pandas as pd
import os
import os.path as pth
import warnings
from openmdao.core.group import Group

from .openvsp import OPENVSPSimpleGeometry, DEFAULT_WING_AIRFOIL, DEFAULT_HTP_AIRFOIL
from ...constants import SPAN_MESH_POINT_OPENVSP
from ...components.compute_reynolds import ComputeUnitReynolds

INPUT_AOA = 10.0  # only one value given since calculation is done by default around 0.0!


class ComputeAEROopenvsp(Group):
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare('wing_airfoil_file', default=DEFAULT_WING_AIRFOIL, types=str, allow_none=True)
        self.options.declare('htp_airfoil_file', default=DEFAULT_HTP_AIRFOIL, types=str, allow_none=True)

    def setup(self):
        self.add_subsystem("comp_unit_reynolds", ComputeUnitReynolds(low_speed_aero=self.options["low_speed_aero"]),
                           promotes=["*"])
        self.add_subsystem("aero_openvsp",
                           _ComputeAEROopenvsp(
                               low_speed_aero=self.options["low_speed_aero"],
                               result_folder_path=self.options["result_folder_path"],
                               wing_airfoil_file=self.options["wing_airfoil_file"],
                               htp_airfoil_file=self.options["htp_airfoil_file"],
                           ), promotes=["*"])


class _ComputeAEROopenvsp(OPENVSPSimpleGeometry):

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare("openvsp_exe_path", default="", types=str, allow_none=True)
        self.options.declare("wing_airfoil_file", default=DEFAULT_WING_AIRFOIL, types=str, allow_none=True)
        self.options.declare("htp_airfoil_file", default=DEFAULT_HTP_AIRFOIL, types=str, allow_none=True)
        
    def setup(self):
        super().setup()
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
        else:
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units='m')

        if self.options["low_speed_aero"]:
            self.add_output("data:aerodynamics:wing:low_speed:CL0_clean")
            self.add_output("data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1")
            self.add_output("data:aerodynamics:wing:low_speed:CM0_clean")
            self.add_output("data:aerodynamics:wing:low_speed:Y_vector", shape=SPAN_MESH_POINT_OPENVSP, units="m")
            self.add_output("data:aerodynamics:wing:low_speed:CL_vector", shape=SPAN_MESH_POINT_OPENVSP)
            self.add_output("data:aerodynamics:wing:low_speed:induced_drag_coefficient")
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:CL0")
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1")
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient")
        else:
            self.add_output("data:aerodynamics:wing:cruise:CL0_clean")
            self.add_output("data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1")
            self.add_output("data:aerodynamics:wing:cruise:CM0_clean")
            self.add_output("data:aerodynamics:wing:cruise:CM_alpha", units="rad**-1")
            self.add_output("data:aerodynamics:wing:cruise:induced_drag_coefficient")
            self.add_output("data:aerodynamics:horizontal_tail:cruise:CL0")
            self.add_output("data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1")
            self.add_output("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient")
        
        self.declare_partials("*", "*", method="fd")

    def check_config(self, logger):
        # let void to avoid logger error on "The command cannot be empty"
        pass
    
    def compute(self, inputs, outputs):

        # Check AOA input is float
        if not (type(INPUT_AOA) == float):
            raise TypeError('INPUT_AOA should be a float!')

        # Get inputs necessary to define global geometry
        sref_wing = float(inputs['data:geometry:wing:area'])
        sref_htp = float(inputs['data:geometry:horizontal_tail:area'])
        area_ratio = sref_htp / sref_wing
        if self.options["low_speed_aero"]:
            altitude = 0.0
            mach = round(float(inputs["data:aerodynamics:low_speed:mach"])*1e6)/1e6
        else:
            altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
            mach = round(float(inputs["data:aerodynamics:cruise:mach"])*1e6)/1e6
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
            wing_X = self.compute_wing(inputs, outputs, altitude, mach, INPUT_AOA)

            # Post-process wing data -----------------------------------------------------------------------------------
            width_max = inputs["data:geometry:fuselage:maximum_width"]
            span_wing = inputs['data:geometry:wing:span']
            k_fus = 1 + 0.025 * width_max / span_wing - 0.025 * (width_max / span_wing) ** 2
            cl_0_wing = float(wing_0["cl"] * k_fus)
            cl_X_wing = float(wing_X["cl"] * k_fus)
            cm_0_wing = float(wing_0["cm"] * k_fus)
            cl_alpha_wing = (cl_X_wing - cl_0_wing) / (INPUT_AOA * math.pi / 180)
            y_vector = wing_0["y_vector"]
            cl_vector = (np.array(wing_0["cl_vector"]) * k_fus).tolist()
            k_fus = 1 - 2 * (width_max / span_wing) ** 2  # Fuselage correction
            # Full aircraft correction: Wing lift is 105% of total lift, so: CDi = (CL*1.05)^2/(piAe) -> e' = e/1.05^2
            coef_e = float(wing_X["coef_e"] * k_fus / 1.05 ** 2)
            coef_k_wing = float(1. / (math.pi * span_wing ** 2 / sref_wing * coef_e))
            # Post-process HTP data
            _, htp_0, aircraft_0 = self.compute_aircraft(inputs, outputs, altitude, mach, 0.0)
            _, htp_X, _ = self.compute_aircraft(inputs, outputs, altitude, mach, INPUT_AOA)
            cl_0_htp = float(htp_0["cl"])
            cl_X_htp = float(htp_X["cl"])
            cl_alpha_htp = float((cl_X_htp - cl_0_htp) / (INPUT_AOA * math.pi / 180))
            coef_k_htp = float(htp_0["cdi"]) / cl_0_htp ** 2

            # Resize vectors -------------------------------------------------------------------------------------------
            if SPAN_MESH_POINT_OPENVSP < len(y_vector):
                y_interp = np.linspace(y_vector[0], y_vector[-1], SPAN_MESH_POINT_OPENVSP)
                cl_vector = np.interp(y_interp, y_vector, cl_vector)
                y_vector = y_interp
                warnings.warn("Defined maximum span mesh in fast aerodynamics\\constants.py exceeded!")
            else:
                additional_zeros = list(np.zeros(SPAN_MESH_POINT_OPENVSP-len(y_vector)))
                y_vector.extend(additional_zeros)
                cl_vector.extend(additional_zeros)

            # Save results to defined path -----------------------------------------------------------------------------
            if self.options["result_folder_path"] != "":
                results = [cl_0_wing, cl_alpha_wing, cm_0_wing, y_vector, cl_vector, cl_0_htp, cl_alpha_htp,
                           coef_k_wing, coef_k_htp]
                self.save_results(result_file_path, results)

        # Else retrieved results are used, eventually adapted with new area ratio
        else:

            # Read values from result file -----------------------------------------------------------------------------
            data = self.read_results(result_file_path)
            cl_0_wing = float(data.loc["cl_0_wing", 0])
            cl_alpha_wing = float(data.loc["cl_alpha_wing", 0])
            cm_0_wing = float(data.loc["cm_0_wing", 0])
            y_vector = np.array([float(i) for i in data.loc["y_vector", 0][1:-2].split(',')])
            cl_vector = np.array([float(i) for i in data.loc["cl_vector", 0][1:-2].split(',')])
            coef_k_wing = float(data.loc["coef_k_wing", 0])
            cl_0_htp = float(data.loc["cl_0_htp", 0]) * (area_ratio/saved_area_ratio)
            cl_alpha_htp = float(data.loc["cl_alpha_htp", 0]) * (area_ratio/saved_area_ratio)
            coef_k_htp = float(data.loc["coef_k_htp", 0]) * (area_ratio/saved_area_ratio)

        # Defining outputs -----------------------------------------------------------------------------
        if self.options["low_speed_aero"]:
            outputs['data:aerodynamics:wing:low_speed:CL0_clean'] = cl_0_wing
            outputs['data:aerodynamics:wing:low_speed:CL_alpha'] = cl_alpha_wing
            outputs['data:aerodynamics:wing:low_speed:CM0_clean'] = cm_0_wing
            outputs['data:aerodynamics:wing:low_speed:Y_vector'] = y_vector
            outputs['data:aerodynamics:wing:low_speed:CL_vector'] = cl_vector
            outputs["data:aerodynamics:wing:low_speed:induced_drag_coefficient"] = coef_k_wing
            outputs['data:aerodynamics:horizontal_tail:low_speed:CL0'] = cl_0_htp
            outputs['data:aerodynamics:horizontal_tail:low_speed:CL_alpha'] = cl_alpha_htp
            outputs["data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"] = coef_k_htp
        else:
            outputs['data:aerodynamics:wing:cruise:CL0_clean'] = cl_0_wing
            outputs['data:aerodynamics:wing:cruise:CL_alpha'] = cl_alpha_wing
            outputs['data:aerodynamics:wing:cruise:CM0_clean'] = cm_0_wing
            outputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"] = coef_k_wing
            outputs['data:aerodynamics:horizontal_tail:cruise:CL0'] = cl_0_htp
            outputs['data:aerodynamics:horizontal_tail:cruise:CL_alpha'] = cl_alpha_htp
            outputs["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"] = coef_k_htp


    @staticmethod
    def search_results(result_folder_path, geometry_set):

        if os.path.exists(result_folder_path):
            if result_folder_path != "":
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

        labels = ["cl_0_wing", "cl_alpha_wing", "cm_0_wing", "y_vector", "cl_vector", "cl_0_htp", "cl_alpha_htp",
                  "coef_k_wing", "coef_k_htp"]
        data = pd.DataFrame(results, index=labels)
        data.to_csv(result_file_path)


    @staticmethod
    def read_results(result_file_path):

        data = pd.read_csv(result_file_path)
        values = data.to_numpy()[:, 1].tolist()
        labels = data.to_numpy()[:, 0].tolist()

        return pd.DataFrame(values, index=labels)
