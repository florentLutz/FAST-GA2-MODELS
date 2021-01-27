"""
    Estimation of cl/cm/oswald aero coefficients using VLM
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

import numpy as np
import math
import os
import os.path as pth
import pandas as pd
import warnings
import logging
from openmdao.core.group import Group
from openmdao.core.explicitcomponent import ExplicitComponent

from fastoad.utils.physics import Atmosphere

from .vlm import VLM
from ...constants import SPAN_MESH_POINT_OPENVSP
from ...constants import POLAR_POINT_COUNT
from ..xfoil import XfoilPolar
from ...components.compute_reynolds import ComputeUnitReynolds

DEFAULT_WING_AIRFOIL = "naca23012.af"
DEFAULT_HTP_AIRFOIL = "naca0012.af"
INPUT_AOA = 10.0  # only one value given since calculation is done by default around 0.0!
_LOGGER = logging.getLogger(__name__)


class ComputeAEROvlm(Group):
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare('wing_airfoil_file', default=DEFAULT_WING_AIRFOIL, types=str, allow_none=True)
        self.options.declare('htp_airfoil_file', default=DEFAULT_HTP_AIRFOIL, types=str, allow_none=True)

    def setup(self):
        self.add_subsystem("comp_unit_reynolds", ComputeUnitReynolds(low_speed_aero=self.options["low_speed_aero"]),
                           promotes=["*"])
        self.add_subsystem("comp_local_reynolds", ComputeLocalReynolds(low_speed_aero=self.options["low_speed_aero"]),
                           promotes=["*"])
        if self.options["low_speed_aero"]:
            self.add_subsystem("wing_polar_ls",
                               XfoilPolar(
                                   airfoil_file=self.options["wing_airfoil_file"]
                               ), promotes=[])
            self.add_subsystem("htp_polar_ls",
                               XfoilPolar(
                                   airfoil_file=self.options["htp_airfoil_file"]
                               ), promotes=[])
        else:
            self.add_subsystem("wing_polar_hs",
                               XfoilPolar(
                                   airfoil_file=self.options["wing_airfoil_file"]
                               ), promotes=[])
            self.add_subsystem("htp_polar_hs",
                               XfoilPolar(
                                   airfoil_file=self.options["htp_airfoil_file"]
                               ), promotes=[])
        self.add_subsystem("aero_vlm",
                           _ComputeAEROvlm(
                               low_speed_aero=self.options["low_speed_aero"],
                               result_folder_path=self.options["result_folder_path"],
                               wing_airfoil_file=self.options["wing_airfoil_file"],
                               htp_airfoil_file=self.options["htp_airfoil_file"],
                           ), promotes=["*"])

        if self.options["low_speed_aero"]:
            self.connect("data:aerodynamics:low_speed:mach", "wing_polar_ls.xfoil:mach")
            self.connect("data:aerodynamics:wing:low_speed:reynolds", "wing_polar_ls.xfoil:reynolds")
            self.connect("wing_polar_ls.xfoil:CL", "data:aerodynamics:wing:low_speed:CL")
            self.connect("wing_polar_ls.xfoil:CDp", "data:aerodynamics:wing:low_speed:CDp")
            self.connect("data:aerodynamics:low_speed:mach", "htp_polar_ls.xfoil:mach")
            self.connect("data:aerodynamics:horizontal_tail:low_speed:reynolds", "htp_polar_ls.xfoil:reynolds")
            self.connect("htp_polar_ls.xfoil:CL", "data:aerodynamics:horizontal_tail:low_speed:CL")
            self.connect("htp_polar_ls.xfoil:CDp", "data:aerodynamics:horizontal_tail:low_speed:CDp")
        else:
            self.connect("data:aerodynamics:cruise:mach", "wing_polar_hs.xfoil:mach")
            self.connect("data:aerodynamics:wing:cruise:reynolds", "wing_polar_hs.xfoil:reynolds")
            self.connect("wing_polar_hs.xfoil:CL", "data:aerodynamics:wing:cruise:CL")
            self.connect("wing_polar_hs.xfoil:CDp", "data:aerodynamics:wing:cruise:CDp")
            self.connect("data:aerodynamics:cruise:mach", "htp_polar_hs.xfoil:mach")
            self.connect("data:aerodynamics:horizontal_tail:cruise:reynolds", "htp_polar_hs.xfoil:reynolds")
            self.connect("htp_polar_hs.xfoil:CL", "data:aerodynamics:horizontal_tail:cruise:CL")
            self.connect("htp_polar_hs.xfoil:CDp", "data:aerodynamics:horizontal_tail:cruise:CDp")


class ComputeLocalReynolds(ExplicitComponent):
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:unit_reynolds", val=np.nan, units="m**-1")
        else:
            self.add_input("data:aerodynamics:cruise:unit_reynolds", val=np.nan, units="m**-1")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")

        if self.options["low_speed_aero"]:
            self.add_output("data:aerodynamics:wing:low_speed:reynolds")
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:reynolds")
        else:
            self.add_output("data:aerodynamics:wing:cruise:reynolds")
            self.add_output("data:aerodynamics:horizontal_tail:cruise:reynolds")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:wing:low_speed:reynolds"] = (
                inputs["data:aerodynamics:low_speed:unit_reynolds"]
                * inputs["data:geometry:wing:MAC:length"]
            )
            outputs["data:aerodynamics:horizontal_tail:low_speed:reynolds"] = (
                    inputs["data:aerodynamics:low_speed:unit_reynolds"]
                    * inputs["data:geometry:horizontal_tail:MAC:length"]
            )
        else:
            outputs["data:aerodynamics:wing:cruise:reynolds"] = (
                    inputs["data:aerodynamics:cruise:unit_reynolds"]
                    * inputs["data:geometry:wing:MAC:length"]
            )
            outputs["data:aerodynamics:horizontal_tail:cruise:reynolds"] = (
                    inputs["data:aerodynamics:cruise:unit_reynolds"]
                    * inputs["data:geometry:horizontal_tail:MAC:length"]
            )


class _ComputeAEROvlm(VLM):

    def initialize(self):
        super().initialize()
        self.options.declare("low_speed_aero", default=False, types=bool)
        self.options.declare("result_folder_path", default="", types=str)
        
    def setup(self):
        
        super().setup()
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        nans_array = np.full(POLAR_POINT_COUNT, np.nan)
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:CL", val=nans_array)
            self.add_input("data:aerodynamics:wing:low_speed:CDp", val=nans_array)
            self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL", val=nans_array)
            self.add_input("data:aerodynamics:horizontal_tail:low_speed:CDp", val=nans_array)
        else:
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units='ft')
            self.add_input("data:aerodynamics:wing:cruise:CL", val=nans_array)
            self.add_input("data:aerodynamics:wing:cruise:CDp", val=nans_array)
            self.add_input("data:aerodynamics:horizontal_tail:cruise:CL", val=nans_array)
            self.add_input("data:aerodynamics:horizontal_tail:cruise:CDp", val=nans_array)

        if self.options["low_speed_aero"]:
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
    
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # Check AOA input length to be =2
        if not (type(INPUT_AOA) == float):
            raise TypeError('INPUT_AOA should be a float!')
        else:
            INPUT_AOAList = [0.0, INPUT_AOA]

        # Get inputs
        sref_wing = float(inputs['data:geometry:wing:area'])
        sref_htp = float(inputs['data:geometry:horizontal_tail:area'])
        area_ratio = sref_htp / sref_wing
        if self.options["low_speed_aero"]:
            altitude = 0.0
            atm = Atmosphere(altitude)
            mach = round(float(inputs["data:aerodynamics:low_speed:mach"])*1e6)/1e6
        else:
            altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
            atm = Atmosphere(altitude)
            mach = round(float(inputs["data:aerodynamics:cruise:mach"])*1e6)/1e6

        # Create result folder first (if it must fail, let it fail as soon as possible)
        result_folder_path = self.options["result_folder_path"]
        if result_folder_path != "":
            if not os.path.exists(result_folder_path):
                os.makedirs(pth.join(result_folder_path), exist_ok=True)

        # Get the primary form factors for wing/htp and store it to geometryX.csv file (to avoid re-computation)
        # or save result file path if geometry already computed (to load results afterwards)
        already_computed = False
        result_file_path = None
        saved_area_ratio = 1.0
        if result_folder_path != "":
            sweep25_wing = float(inputs["data:geometry:wing:sweep_25"])
            taper_ratio_wing = float(inputs["data:geometry:wing:taper_ratio"])
            aspect_ratio_wing = float(inputs["data:geometry:wing:aspect_ratio"])
            sweep25_htp = float(inputs["data:geometry:horizontal_tail:sweep_25"])
            aspect_ratio_htp = float(inputs["data:geometry:horizontal_tail:aspect_ratio"])
            taper_ratio_htp = float(inputs["data:geometry:horizontal_tail:taper_ratio"])
            geometry_set = np.around(np.array(
                [sweep25_wing, taper_ratio_wing, aspect_ratio_wing, sweep25_htp, taper_ratio_htp,
                 aspect_ratio_htp, mach, area_ratio]), decimals=6)
            geometry_set_labels = ["sweep25_wing", "taper_ratio_wing", "aspect_ratio_wing", "sweep25_htp",
                                   "taper_ratio_htp", "aspect_ratio_htp", "mach", "area_ratio"]
            # If some results already stored search for corresponding geometry
            if pth.exists(pth.join(result_folder_path, "geometry_0.csv")):
                idx = 0
                while pth.exists(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv")):
                    if pth.exists(pth.join(result_folder_path, "vlm_" + str(idx) + ".csv")):
                        data = pd.read_csv(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv"))
                        values = data.to_numpy()[:, 1].tolist()
                        labels = data.to_numpy()[:, 0].tolist()
                        data = pd.DataFrame(values, index=labels)
                        # noinspection PyBroadException
                        try:
                            if np.size(data.loc[geometry_set_labels[0:-1], 0].to_numpy()) == 7:
                                saved_set = np.around(data.loc[geometry_set_labels[0:-1], 0].to_numpy(), decimals=6)
                                if np.sum(saved_set == geometry_set[0:-1]) == 7:
                                    result_file_path = pth.join(result_folder_path, "vlm_" + str(idx) + ".csv")
                                    saved_area_ratio = data.loc["area_ratio", 0]
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
                result_file_path = pth.join(result_folder_path, "vlm_" + str(idx) + ".csv")


        if not already_computed:

            # Get inputs (and calculate missing ones)
            aspect_ratio_wing = float(inputs["data:geometry:wing:aspect_ratio"])
            aspect_ratio_htp = float(inputs["data:geometry:horizontal_tail:aspect_ratio"])
            wing_span = inputs['data:geometry:wing:span']
            width_max = inputs['data:geometry:fuselage:maximum_width']
            htp_span = inputs['data:geometry:horizontal_tail:span']
            v_inf = max(atm.speed_of_sound * mach, 0.01)  # avoid V=0 m/s crashes
            if self.options["low_speed_aero"]:
                cl_wing_airfoil = inputs["data:aerodynamics:wing:low_speed:CL"]
                cdp_wing_airfoil = inputs["data:aerodynamics:wing:low_speed:CDp"]
                cl_htp_airfoil = inputs["data:aerodynamics:horizontal_tail:low_speed:CL"]
                cdp_htp_airfoil = inputs["data:aerodynamics:horizontal_tail:low_speed:CDp"]
            else:
                cl_wing_airfoil = inputs["data:aerodynamics:wing:cruise:CL"]
                cdp_wing_airfoil = inputs["data:aerodynamics:wing:cruise:CDp"]
                cl_htp_airfoil = inputs["data:aerodynamics:horizontal_tail:cruise:CL"]
                cdp_htp_airfoil = inputs["data:aerodynamics:horizontal_tail:cruise:CDp"]

            # Initialize
            super()._run(inputs)

            # (Post)-processing: STEP1/2 - wing coefficients -------------------------------------------
            cl_wing_vect, _, oswald, cm_wing_vect = super().compute_wing(inputs, INPUT_AOAList, v_inf, flaps_angle=0.0,
                                                                         use_airfoil=True)
            y_vector, cl_vector = super().get_cl_curve(INPUT_AOAList[0], v_inf)
            k_fus = 1 - 2 * (width_max / wing_span) ** 2
            beta = math.sqrt(1 - mach ** 2)  # Prandtl-Glauert
            cl_0_wing = float(cl_wing_vect[0] * k_fus / beta)
            cl_1_wing = float(cl_wing_vect[1] * k_fus / beta)
            cm_0_wing = float(cm_wing_vect[0] * k_fus / beta)
            cm_1_wing = float(cm_wing_vect[1] * k_fus / beta)
            cl_vector = (np.array(cl_vector) * k_fus).tolist()
            # Calculate derivative
            cl_alpha_wing = (cl_1_wing - cl_0_wing) / (INPUT_AOAList[1] * math.pi / 180)
            cm_alpha_wing = (cm_1_wing - cm_0_wing) / (INPUT_AOAList[1] * math.pi / 180)
            if SPAN_MESH_POINT_OPENVSP < len(y_vector):
                y_interp = np.linspace(y_vector[0], y_vector[-1], SPAN_MESH_POINT_OPENVSP)
                cl_vector = np.interp(y_interp, y_vector, cl_vector)
                y_vector = y_interp
                warnings.warn("Defined maximum span mesh in fast aerodynamics\\constants.py exceeded!")
            else:
                additional_zeros = list(np.zeros(SPAN_MESH_POINT_OPENVSP-len(y_vector)))
                y_vector.extend(additional_zeros)
                cl_vector.extend(additional_zeros)
            # Calculate oswald
            oswald = oswald[1] * k_fus  # Fuselage correction
            if mach > 0.4:
                oswald = oswald * (-0.001521 * ((mach - 0.05) / 0.3 - 1) ** 10.82 + 1)  # Mach correction
            cdp_foil = self._interpolate_cdp(cl_wing_airfoil, cdp_wing_airfoil, cl_1_wing)
            cdi = (1.05 * cl_1_wing) ** 2 / (
                        math.pi * aspect_ratio_wing * oswald) + cdp_foil  # Aircraft cor.: Wing = 105% total lift.
            coef_e = cl_1_wing ** 2 / (math.pi * aspect_ratio_wing * cdi)
            coef_k_wing = float(1. / (math.pi * wing_span ** 2 / sref_wing * coef_e))

            # (Post)-processing: STEP2/2 - HTP coefficients --------------------------------------------
            # Calculate downwash angle based on Gudmundsson model (p.467)

            downwash_angle = 2.0 * np.array(cl_wing_vect)/beta * 180.0 / (aspect_ratio_wing * np.pi**2)
            HTP_AOAList = list(np.array(INPUT_AOAList) - downwash_angle)
            cl_htp_vect, cdi_htp_vect, oswald, cm_htp_vect = super().compute_htp(inputs, HTP_AOAList, v_inf, use_airfoil=True)
            # Write value with wing Sref
            cl_htp_vect = np.array(cl_htp_vect) / beta * area_ratio
            cm_htp_vect = np.array(cm_htp_vect) / beta * area_ratio
            cl_0_htp = float(cl_htp_vect[0])
            cl_1_htp = float(cl_htp_vect[1]) / area_ratio
            cdi_1_htp = float(cdi_htp_vect[1])
            cm_0_htp = float(cm_htp_vect[0])
            # Calculate derivative
            cl_alpha_htp = float((cl_htp_vect[1] - cl_htp_vect[0]) / (INPUT_AOAList[1] * math.pi / 180))
            cm_alpha_htp = float((cm_htp_vect[1] - cm_htp_vect[0]) / (INPUT_AOAList[1] * math.pi / 180))
            # Calculate oswald
            oswald = oswald[1]
            if mach > 0.4:
                oswald = oswald * (-0.001521 * ((mach - 0.05) / 0.3 - 1) ** 10.82 + 1)  # Mach correction
            cdp_foil = self._interpolate_cdp(cl_htp_airfoil, cdp_htp_airfoil, cl_1_htp)
            cdi = cdi_1_htp + cdp_foil
            coef_e = cl_1_htp ** 2 / (math.pi * aspect_ratio_htp * cdi)
            coef_k_htp = float(1. / (math.pi * htp_span ** 2 / sref_htp * coef_e) / area_ratio)

            # Save results to defined path -------------------------------------------------------------
            if self.options["result_folder_path"] != "":
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
            coef_k_wing = float(data.loc["coef_k_wing", 0])
            cl_0_htp = float(data.loc["cl_0_htp", 0]) * (area_ratio / saved_area_ratio)
            cl_alpha_htp = float(data.loc["cl_alpha_htp", 0]) * (area_ratio / saved_area_ratio)
            cm_0_htp = float(data.loc["cm_0_htp", 0]) * (area_ratio / saved_area_ratio)
            cm_alpha_htp = float(data.loc["cm_alpha_htp", 0]) * (area_ratio / saved_area_ratio)
            coef_k_htp = float(data.loc["coef_k_htp", 0]) * (area_ratio / saved_area_ratio)

        # Defining outputs -----------------------------------------------------------------------------
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
    def _interpolate_cdp(lift_coeff: np.ndarray, drag_coeff: np.ndarray, ojective: float) -> float:
        """

        :param lift_coeff: CL array
        :param drag_coeff: CDp array
        :param ojective: CL_ref objective value
        :return: CD_ref if CL_ref encountered, or default value otherwise
        """
        # Reduce vectors for interpolation
        for idx in range(len(lift_coeff)):
            if np.sum(lift_coeff[idx:len(lift_coeff)] == 0) == (len(lift_coeff) - idx):
                lift_coeff = lift_coeff[0:idx]
                drag_coeff = drag_coeff[0:idx]
                break

        # Interpolate value if within the interpolation range
        if min(lift_coeff) <= ojective <= max(lift_coeff):
            idx_max = int(float(np.where(lift_coeff == max(lift_coeff))[0]))
            return np.interp(ojective, lift_coeff[0:idx_max + 1], drag_coeff[0:idx_max + 1])
        elif ojective < lift_coeff[0]:
            cdp = drag_coeff[0] + (ojective - lift_coeff[0]) * (drag_coeff[1] - drag_coeff[0]) \
                  / (lift_coeff[1] - lift_coeff[0])
        else:
            cdp = drag_coeff[-1] + (ojective - lift_coeff[-1]) * (drag_coeff[-1] - drag_coeff[-2]) \
                  / (lift_coeff[-1] - lift_coeff[-2])
        _LOGGER.warning("CL not in range. Linear extrapolation of CDp value (%s)", cdp)
        return cdp
