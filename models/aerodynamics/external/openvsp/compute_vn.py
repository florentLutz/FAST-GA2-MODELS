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
from scipy.constants import g
import scipy.optimize as optimize
from openmdao.core.group import Group

from .openvsp import OPENVSPSimpleGeometry, DEFAULT_WING_AIRFOIL, DEFAULT_HTP_AIRFOIL
from ...constants import SPAN_MESH_POINT_OPENVSP
from ....propulsion.fuel_propulsion.base import FuelEngineSet
from ...components.compute_reynolds import ComputeUnitReynolds

from fastoad import BundleLoader
from fastoad.base.flight_point import FlightPoint
from fastoad.constants import EngineSetting
from fastoad.utils.physics import Atmosphere

INPUT_AOA = 10.0  # only one value given since calculation is done by default around 0.0!


class ComputeVNopenvsp(OPENVSPSimpleGeometry):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None
        self.kts_to_ms = 0.5144  # Converting from knots to meters per seconds
        self.ft_to_m = 0.3048  # Converting from feet to meters
        self.lbf_to_N = 4.4482  # Converting from pound force to Newtons

    def initialize(self):
        self.options.declare("openvsp_exe_path", default="", types=str, allow_none=True)
        self.options.declare("wing_airfoil_file", default=DEFAULT_WING_AIRFOIL, types=str, allow_none=True)
        self.options.declare("htp_airfoil_file", default=DEFAULT_HTP_AIRFOIL, types=str, allow_none=True)
        self.options.declare("propulsion_id", default="", types=str)
        
    def setup(self):
        super().setup()
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

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
        v_tas = inputs["data:aerodynamics:cruise:mach"]
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        design_mass = inputs["data:weight:aircraft:DW"]

        design_vc = Atmosphere(cruise_altitude, altitude_in_feet=False).get_equivalent_airspeed(v_tas)
        velocity_array, load_factor_array, conditions = self.flight_domain(inputs, outputs, design_mass,
                                                                           cruise_altitude, design_vc,
                                                                           design_n_ps=0.0, design_n_ng=0.0)



    def flight_domain(self, inputs, outputs, mass, altitude, design_vc, design_n_ps=0.0, design_n_ng=0.0):

        # Get necessary inputs
        wing_area = inputs["data:geometry:wing:area"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        category = inputs["data:TLAR:category"]  # Aerobatic = 1.0, Utility = 2.0, Normal = 3.0, Commuter = 4.0
        level = inputs["data:TLAR:level"]
        Vh = inputs["data:TLAR:v_max_sl"]
        root_chord = inputs["data:geometry:wing:root:chord"]
        tip_chord = inputs["data:geometry:wing:tip:chord"]
        cl_max_flaps = inputs["data:weight:aircraft:MTOW"]
        cl_max = inputs["data:weight:aircraft:MTOW"]
        cl_min = inputs["data:weight:aircraft:MTOW"]
        cl_alpha_LS = inputs["data:geometry:wing:tip:chord"]
        mean_chord = (root_chord + tip_chord) / 2.0
        atm_0 = Atmosphere(0.0)
        atm = Atmosphere(altitude, altitude_in_feet=False)

        # Initialise the lists in which we will store the data
        velocity_array = []
        load_factor_array = []

        # For some of the correlation presented in the regulation, we need to convert the data
        # of the airplane to imperial units
        weight_lbf = (mass * g) / self.lbf_to_N
        mtow_lbf = (mtow * g) / self.lbf_to_N
        wing_area_sft = wing_area / (self.ft_to_m ** 2.0)
        mtow_loading_psf = mtow_lbf / wing_area_sft  # [lbf/ft**2]

        # We can now start computing the values of the different air-speeds given in the regulation
        # as well as the load factors. We will here make the choice to stick with the limits given
        # in the certifications even though they sometimes allow to choose design speeds and loads
        # over the values written in the documents.

        # Lets start by computing the 1g stall speeds using the usual formulations
        Vs_1g_ps = math.sqrt((2. * mass * g) / (atm_0.density * wing_area * cl_max)) / self.kts_to_ms  # [KEAS]
        Vs_1g_ng = math.sqrt((2. * mass * g) / (atm_0.density * wing_area * abs(cl_min))) / self.kts_to_ms  # [KEAS]
        velocity_array.append(Vs_1g_ps)
        velocity_array.append(Vs_1g_ng)

        # We will now establish the minimum limit maneuvering load factors outside of gust load
        # factors. Th designer can take higher load factor if he so wish. As will later be done for the
        # the cruising speed, we will simply ensure that the designer choice agrees with certifications
        # The limit load factor can be found in section CS 23.337 (a) and (b)
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm

        if category == 1.0:
            n_lim_1 = 6.0  # For aerobatic GA aircraft
        else:
            n_lim_1 = 3.80  # For non aerobatic GA aircraft
        n_lim_2 = 2.1 + 24000. / (mtow_lbf + 10000.)  # CS 23.337 (a)
        n_lim_ps_min = min(n_lim_1, n_lim_2)  # CS 23.337 (a)
        n_lim_ps = max(n_lim_ps_min, design_n_ps)

        if category == 1.0:
            n_lim_ng_max = - 0.5 * n_lim_ps  # CS 23.337 (b)
        else:
            n_lim_ng_max = - 0.4 * n_lim_ps  # CS 23.337 (b)
        n_lim_ng = min(n_lim_ng_max, design_n_ng)

        load_factor_array.append(n_lim_ps)
        load_factor_array.append(n_lim_ng)

        # We can now go back to the computation of the maneuvering speeds, we will first compute it
        # "traditionally" and should we find out that the line limited by the Cl max is under the gust
        # line, we will adjust it (see Step 10. of section 16.4.1 of (1)). As for the traditional
        # computation they can be found in CS 23.335 (c)
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm

        Va = Vs_1g_ps * math.sqrt(n_lim_ps)  # [KEAS]
        Vg = Vs_1g_ng * math.sqrt(abs(n_lim_ng))  # [KEAS]

        # Starting from there, we need to compute the gust lines as it can have an impact on the choice
        # of the maneuvering speed. We will also compute the maximum intensity gust line for later
        # use but keep in mind that this is specific for commuter or level 4 aircraft
        # The values used to compute the gust lines can be found in CS 23.341
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf

        # We first compute the gust velocities as presented in CS 23.333 (c), for now, we don't take into account
        # the case of the commuter nor do we implement the reduction of gust intensity with the location
        # of the gust center

        if altitude < 20000.0:
            U_de_Vc = 50.  # [ft/s]
            U_de_Vd = 25.  # [ft/s]
            U_de_Vmg = 66.  # [ft/s]
        elif 20000.0 < altitude < 50000.0:
            U_de_Vc = 66.7 - 0.000833 * altitude  # [ft/s]
            U_de_Vd = 33.4 - 0.000417 * altitude  # [ft/s]
            U_de_Vmg = 84.7 - 0.000933 * altitude  # [ft/s]
        else:
            U_de_Vc = 25.  # [ft/s]
            U_de_Vd = 12.5  # [ft/s]
            U_de_Vmg = 38.  # [ft/s]

        # Let us define aeroplane mass ratio formula and alleviation factor formula
        mu_g = lambda x: (2.0 * mass * g / wing_area) / (atm.density * mean_chord * x * g)  # [x = cl_alpha]
        K_g = lambda x: (0.88 * x) / (5.3 + x)  # [x = mu_g]



        # We now need to check if we are in the aforementioned case (usually happens for low design wing
        # loading aircraft and/or mission wing loading)

        coef_Va_gust_line = (K_g(mu_g(cl_alpha_LS)) * U_de_Vc * cl_alpha_LS) \
                            / (498. * weight_lbf / wing_area_sft)  # [1./KEAS]
        na_gust = 1. + coef_Va_gust_line * Va

        if na_gust > n_lim_ps:
            # In case the gust line load factor is above the maneuvering load factor, we need to solve
            # a polynomial equation to find the new maneuvering speed and load factor

            stall_line_coeff = (0.5 * atm_0.density * self.kts_to_ms ** 2.0 * wing_area * cl_max) / (mass * g)
            Va = self.maneuver_velocity(stall_line_coeff, coef_Va_gust_line, atm.speed_of_sound, n_lim_ps)  # [KEAS]
            n_Va = 1. + coef_Va_gust_line * Va  # [-]
        else:
            n_Va = n_lim_ps  # [-]

        velocity_array.append(Va)
        load_factor_array.append(n_Va)

        # We now need to do the same thing for the negative maneuvering speed

        coef_Vg_gust_line = (K_g(mu_g(cl_alpha_LS)) * U_de_Vc * cl_alpha_LS) \
                            / (498. * weight_lbf / wing_area_sft)  # [1./KEAS]
        ng_gust = 1. - coef_Vg_gust_line * Vg  # [-]

        if ng_gust < n_lim_ng:
            # In case the gust line load factor is below the maneuvering load factor, we need to solve
            # a polynomial equation to find the new maneuvering speed and load factor
            stall_line_coeff = (0.5 * atm_0.density * self.kts_to_ms ** 2.0 * wing_area * cl_min) / (mass * g)
            Vg = self.maneuver_velocity(stall_line_coeff, coef_Vg_gust_line, atm.speed_of_sound, n_lim_ng, -1)  # [KEAS]
            n_Vg = 1. - coef_Vg_gust_line * Vg  # [-]
        else:
            n_Vg = n_lim_ng  # [-]

        velocity_array.append(Vg)
        load_factor_array.append(n_Vg)

        # For the cruise velocity, things will be different since it is an entry choice. As such we will
        # simply check that it complies with the values given in the certification papers and re-adjust
        # it if necessary. For airplane certified for aerobatics, the coefficient in front of the wing
        # loading in psf is slightly different than for normal aircraft but for either case it becomes
        # 28.6 at wing loading superior to 100 psf
        # Values and methodology used can be found in CS 23.335 (a)
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm

        if category == 1.0:
            if mtow_loading_psf < 20.0:
                k_c = 36.0
            elif mtow_loading_psf < 100.:
                # Linear variation from 33.0 to 28.6
                k_c = 36.0 + (mtow_loading_psf - 20.0) * (28.6 - 36.0) / (100.0 - 20.0)
            else:
                k_c = 28.6
        else:
            if mtow_loading_psf < 20.0:
                k_c = 33.0

            elif mtow_loading_psf < 100.:
                # Linear variation from 33.0 to 28.6
                k_c = 33.0 + (mtow_loading_psf - 20.0) * (28.6 - 33.0) / (100.0 - 20.0)

            else:
                k_c = 28.6

        Vc_min_1 = k_c * math.sqrt(weight_lbf / wing_area_sft)  # [KEAS]
        # This second constraint rather refers to the paragraph on maneuvering speeds, which needs to be chosen
        # so that they are smaller than cruising speeds
        Vc_min_2 = Va  # [KEAS]
        Vc_min = max(Vc_min_1, Vc_min_2)  # [KEAS]

        # Depending on whether or not the maximum Sea Level flight velocity was already demonstrated we either
        # take the value from flight experiments or we compute it using a method which finds for which speed
        # the power required for flight is equal to the power available

        if np.isnan(Vh):
            design_Vc_ms = design_vc * self.kts_to_ms  # [m/s]
            Vh = self.max_speed(inputs, mass, altitude, design_Vc_ms) / self.kts_to_ms  # [KEAS]

        Vc_threshold = 0.9 * Vh  # [KEAS]

        # The certifications specifies that Vc need not be more than 0.9 Vh so we will simply take the
        # minimum value between the Vc_min and this value

        Vc_min_fin = min(Vc_min, Vc_threshold)  # [KEAS]

        # The constraint regarding the maximum velocity for cruise does not appear in the certifications but
        # from a physics point of view we can easily infer that the cruise speed will never be greater than
        # the maximum level velocity at sea level hence

        Vc = max(min(design_vc, Vh), Vc_min_fin)  # [KEAS]
        velocity_array.append(Vc)

        # Lets now look at the load factors associated with the Vc, since it is here that the greatest
        # load factors can appear

        mach = Vc * math.sqrt(atm_0.density / atm.density) * self.kts_to_ms / atm.speed_of_sound
        cl_alpha_local = self.compute_cl_alpha_wing(inputs, outputs, altitude, mach, INPUT_AOA)
        coef_Vc_gust_line = (K_g(mu_g(cl_alpha_local)) * U_de_Vc * cl_alpha_local) \
                            / (498. * weight_lbf / wing_area_sft)  # [1./KEAS]

        n_Vc_gust_ps = 1. + coef_Vc_gust_line * Vc  # [-]
        n_Vc_ps = max(n_Vc_gust_ps, n_lim_ps)  # [-]

        n_Vc_gust_ng = 1. - coef_Vc_gust_line * Vc  # [-]
        n_Vc_ng = min(n_Vc_gust_ng, n_lim_ng)  # [-]

        load_factor_array.append(n_Vc_ps)
        load_factor_array.append(n_Vc_ng)

        # We now compute the diving speed, methods are described in CS 23.335 (b). We will take the minimum
        # diving speed allowable as our design diving speed. We need to keep in mind that this speed could
        # be greater if the designer was willing to show that the structure holds for the wanted Vd. For
        # airplane that needs to be certified for aerobatics use, the factor between Vd_min and Vc_min is
        # slightly different, but they both become 1.35 for wing loading higher than 100 psf
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm

        Vd_min_1 = 1.25 * Vc  # [KEAS]

        if category == 1.0:
            if mtow_loading_psf < 20.0:
                k_d = 1.55
            elif mtow_loading_psf < 100.:
                # Linear variation from 1.55 to 1.35
                k_d = 1.55 + (mtow_loading_psf - 20.0) * (1.35 - 1.55) / (100.0 - 20.0)
            else:
                k_d = 1.35
        elif category == 2.0:
            if mtow_loading_psf < 20.0:
                k_d = 1.50
            elif mtow_loading_psf < 100.:
                # Linear variation from 1.5 to 1.35
                k_d = 1.50 + (mtow_loading_psf - 20.0) * (1.35 - 1.50) / (100.0 - 20.0)
            else:
                k_d = 1.35
        else:
            if mtow_loading_psf < 20.0:
                k_d = 1.4
            elif mtow_loading_psf < 100.:
                # Linear variation from 1.4 to 1.35
                k_d = 1.4 + (mtow_loading_psf - 20.0) * (1.35 - 1.4) / (100.0 - 20.0)
            else:
                k_d = 1.35

        Vd_min_2 = k_d * Vc_min_fin  # [KEAS]
        Vd = max(Vd_min_1, Vd_min_2)  # [KEAS]

        velocity_array.append(Vd)

        # Similarly to what was done for the design cruising speed we will explore the load factors
        # associated with the diving speed since gusts are likely to broaden the flight domain around
        # these points

        mach = Vd * math.sqrt(atm_0.density / atm.density) * self.kts_to_ms / atm.speed_of_sound
        cl_alpha_local = self.compute_cl_alpha_wing(inputs, outputs, altitude, mach, INPUT_AOA)
        coef_Vd_gust_line = (K_g(mu_g(cl_alpha_local)) * U_de_Vd * cl_alpha_local) \
                            / (498. * weight_lbf / wing_area_sft)  # [1./KEAS]

        n_Vd_gust_ps = 1. + coef_Vd_gust_line * Vd  # [-]
        n_Vd_ps = max(n_Vd_gust_ps, n_lim_ps)  # [-]

        # For the negative load factor at the diving speed, it seems that for non_aerobatic airplanes, it is
        # always sized according to the gust lines, regardless of the negative design load factor. For aerobatic
        # airplanes however, it seems as if it is sized for a greater value (more negative) but it does not look
        # to be equal to the negative diving factor as can be seen in figure 16-13 of (1). No information was
        # found for the location of this precises point, so the choice was made to take it as the negative
        # design load factor or the load factor given by the gust, whichever is the greatest (most negative).
        # This way, for non aerobatic airplane, we ensure to be conservative.

        if category == 1.0:
            n_Vd_gust_ng = 1. - coef_Vd_gust_line * Vd  # [-]
            n_Vd_ng = min(n_Vd_gust_ng, n_lim_ng)  # [-]
        else:
            n_Vd_gust_ng = 1. - coef_Vd_gust_line * Vd  # [-]
            n_Vd_ng = n_Vd_gust_ng  # [-]

        load_factor_array.append(n_Vd_ps)
        load_factor_array.append(n_Vd_ng)

        # We have now calculated all the velocities need to plot the flight domain. For the sake of
        # thoroughness we will also compute the maximal structural cruising speed and cruise never-exceed
        # speed. The computation for these two can be found in CS 23.1505
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm

        # Let us start, as presented in the certifications papers with the never-exceed speed
        # Since we made the choice to take the Vd as the minimum value allowed by certifications, the V_ne
        # will have a fixed value and not a range as one would have expect. Indeed if Vd = Vd_min and since
        # V_ne has to be greater or equal to 0.9 x Vd_min and smaller or equal to 0.9 x Vd, V_ne will be equal
        # to 0.9 Vd. For future implementations, it should be noted that this section will need to be rewritten
        # should the Vd become a design parameter like what was made on Vc. Additionally the effect of
        # buffeting which serves as an additional upper limit is not included but should be taken into
        # account in detailed analysis phases

        V_ne = 0.9 * Vd  # [KEAS]

        velocity_array.append(V_ne)

        V_no_min = Vc_min  # [KEAS]
        V_no_max = min(Vc, 0.89 * V_ne)  # [KEAS]

        # Again we need to make a choice for this speed : what value would be retained. We will take the
        # highest speed acceptable for certification, i.e

        V_no = V_no_max  # [KEAS]

        velocity_array.append(V_no)

        # One additional velocity needs to be computed if we are talking about commuter aircraft. It is
        # the maximum gust intensity velocity. Due to the way we are returning the values, even if we are not
        # ivestigating a commuter aircraft we need to return a value for Vb so we will put it to 0.0. If we
        # are invetigating a commuter aircraft, we will compute it according ot the guidleines from CS 23.335 (d)
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm
        # We decided to put this computation here as we may need the gust load factor in cruise conditions for
        # one of the possible candidates for the Vb. While writing this program, the writer realized they were
        # no paragraph that impeach the Vc from being at a value such that one one of the conditions for the
        # minimum speed was above the Vc creating a problem with point (2). This case may however never appear
        # in practice as it would suppose that the Vc chosen is above the stall line which is more than certainly
        # avoided by the correlation between Vc_min and W/S in CS 23.335 (a)

        if (level == 4.0) or (category == 4.0):

            # We first need to compute the intersection of the stall line with the gust line given by the
            # gust of maximum intensity. Similar calculation were already done in case the maneuvering speed
            # is dictated by the Vc gust line so the computation will be very similar

            n_Vc_gust = 1. + coef_Vc_gust_line * self.compute_cl_alpha_wing(inputs, outputs, altitude, mach, INPUT_AOA) \
                        / cl_alpha * Vc  # [-]

            coef_Vb_gust_line = (K_g * U_de_Vb * cl_alpha) / (498. * weight_lbf / wing_area_sft)  # [1./KEAS]
            coeff_stall_line = (0.5 * atm_0.density * 0.5144 ** 2.0 * wing_area * cl_max) / (mass * g)
            Vb_min_1 = self.maneuver_velocity(coeff_stall_line, coef_Vb_gust_line, atm.speed_of_sound, n_Vc_gust)

            # The second candidate for the Vb is given by the stall speed and the load factor at the cruise
            # speed

            Vb_min_2 = Vs_1g_ps * math.sqrt(n_Vc_gust)  # [KEAS]
            Vb = min(Vb_min_1, Vb_min_2)  # [KEAS]

            # As for the computation of the associated load factor, no source were found for any formula or
            # hint as to its computation. It can however be guessed that depending on the minimum value found
            # above, it will either be on the stall line or at the maximum design load factor

            if Vb == Vb_min_1:  # On the gust line
                n_Vb = (0.5 * atm_0.density * wing_area * cl_max) / (mass * g) * (Vb * self.kts_to_ms) ** 2.0  # [-]
            else:
                n_Vb = n_Vc_ps  # [-]

        else:
            Vb = 0.0  # [KEAS]
            n_Vb = 0.0

        velocity_array.append(Vb)
        load_factor_array.append(n_Vb)

        # Let us now look at the flight domain in the flap extended configuration. For the computation of these
        # speeds and load factors, we will use the formula provided in CS 23.1511
        # https://www.easa.europa.eu/sites/default/files/dfu/CS-23%20Amendment%204.pdf
        # https://www.astm.org/Standards/F3116.htm

        # For the computation of the Vfe CS 23.1511, refers to CS 23.345 but there only seems to be a
        # requirement for the lowest the Vfe can be, hence we will take this speed as the Vfe. As for the
        # load factors that are prescribed we will use the guidelines provided in CS 23.345 (b)

        # Let us start by computing the Vfe
        Vsfe_1g_ps = math.sqrt((2. * mass * g) / (atm_0.density * wing_area * cl_max_flaps)) / self.kts_to_ms  # [KEAS]
        Vfe_min_1 = 1.4 * Vs_1g_ps  # [KEAS]
        Vfe_min_2 = 1.8 * Vsfe_1g_ps  # [KEAS]
        Vfe_min = max(Vfe_min_1, Vfe_min_2)  # [KEAS]
        Vfe = Vfe_min  # [KEAS]

        velocity_array.append(Vsfe_1g_ps)
        velocity_array.append(Vfe)

        # We can then move on to the computation of the load limitation of the flapped flight domain, which
        # must be equal to either a constant load factor of 2 or a load factor dictated by a gust of 25 fps.
        # Also since the use of flaps is limited to take-off, approach and landing, we will use the SL density
        # and a constant gust velocity

        U_de_fe = 25.  # [ft/s]

        # Here is the aeroplane mass ratio for the particular load case
        mu_g = (2. * mass * g / wing_area) / (atm.density * mean_chord * cl_alpha * g)  # [-]

        # We can now compute the gust alleviation factor
        K_g = (0.88 * mu_g) / (5.3 + mu_g)  # [-]

        # Finally, we compute the gust line coefficients
        coef_Vfe_gust_line = (K_g * U_de_fe * cl_alpha) / (498. * weight_lbf / wing_area_sft)  # [1./KEAS]

        n_lim_ps_fe = 2.0

        n_Vfe_max_1 = n_lim_ps_fe
        n_Vfe_max_2 = 1. + coef_Vfe_gust_line * Vfe
        n_Vfe = max(n_Vfe_max_1, n_Vfe_max_2)

        load_factor_array.append(n_Vfe)

        # We also store the conditions in which the values were computed so that we can easily access
        # them when drawing the flight domains

        conditions = [mass, altitude]

        return velocity_array, load_factor_array, conditions


    def max_speed(self, inputs, altitude, mass, v_init):

        # noinspection PyTypeChecker
        roots = optimize.fsolve(
            self.delta_axial_load,
            v_init,
            args=(inputs, altitude, mass)
        )[0]

        return np.max(roots[roots > 0.0])


    def delta_axial_load(self, inputs, air_speed, altitude, mass):

        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:count"]
        )
        wing_area = inputs["data:geometry:wing:area"]
        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        coef_k = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]

        # Get the available thrust from propulsion system
        atm = Atmosphere(altitude, altitude_in_feet=False)
        flight_point = FlightPoint(
            mach=air_speed/atm.speed_of_sound, altitude=altitude, engine_setting=EngineSetting.TAKEOFF,
            thrust_rate=1.0
        )
        propulsion_model.compute_flight_points(flight_point)
        thrust = float(flight_point.thrust)

        # Get the necessary thrust to overcome
        cl = (mass * g) / (0.5 * atm.density * wing_area * air_speed**2.0)
        cd = cd0 + coef_k * cl ** 2.0
        drag = 0.5 * atm.density * wing_area * cd * air_speed**2.0

        return thrust - drag

    def maneuver_velocity(self, stall_line_coeff, uncor_gust_coeff, sos, design_n, sign=1.):

        # Be sure to give Vc in m/s
        initial_speed = math.sqrt(design_n / stall_line_coeff)  # [KEAS]

        # We use here the fsolve function from scipy optimize, which solve f(x) = 0.0, here since our
        # function computes the difference between gust load factor and stall load factor, the value which
        # solve the equation will be the maneuvering speed
        # noinspection PyTypeChecker
        roots = optimize.fsolve(
            self.maneuver_load_factor_diff,
            initial_speed,
            args=(stall_line_coeff, uncor_gust_coeff, sos, sign, self)
        )[0]

        return np.max(roots[roots > 0.0])

    @staticmethod
    def maneuver_load_factor_diff(va, stall_line_coeff, uncor_gust_coeff, sos, sign, self):

        n_Va_stall_line = stall_line_coeff * va ** 2.0
        mach = va * 0.5144 / sos

        Cl_alpha_tmp = 5.0
        # No importance since we only want the corrective factor
        cor_gust_coeff = uncor_gust_coeff * self.adjust_Cl_alpha(Cl_alpha_tmp, mach) / Cl_alpha_tmp
        n_Va_gust_line = 1. + sign * cor_gust_coeff * va

        return n_Va_stall_line - n_Va_gust_line
