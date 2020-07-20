"""
Estimation of vertical tail area
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
import openmdao.api as om
from fastoad.utils.physics import Atmosphere


class ComputeVTArea(om.ExplicitComponent):
    """
    Computes area of vertical tail plane

    Area is computed to fulfill lateral stability requirement and engine failure compensation
    for dual-engine aircraft.
    """

    def setup(self):
        
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan, units="kn")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:CG:aft:MAC_position", val=np.nan, units="m")
        self.add_input("data:aerodynamics:fuselage:cruise:CnBeta", val=np.nan)
        self.add_input("data:aerodynamics:vertical_tail:cruise:CnBeta", val=np.nan)
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="kn")
        self.add_input("data:TLAR:v_approach", val=np.nan, units="kn")
        self.add_input("data:mission:sizing:cruise:altitude", val=np.nan, units="ft")
        self.add_input("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:diameter", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:y", val=np.nan, units="m")
        
        self.add_output("data:geometry:vertical_tail:wetted_area", units="m**2", ref=100.0)
        self.add_output("data:geometry:vertical_tail:area", units="m**2", ref=50.0)
        self.add_output("data:aerodynamics:vertical_tail:cruise:CnBeta", units="m**2")

        self.declare_partials(
                "data:geometry:vertical_tail:wetted_area",
                [
                        "data:geometry:wing:area",
                        "data:geometry:wing:span",
                        "data:geometry:wing:MAC:length",
                        "data:weight:aircraft:CG:aft:MAC_position",
                        "data:aerodynamics:fuselage:cruise:CnBeta",
                        "data:aerodynamics:vertical_tail:cruise:CnBeta",
                        "data:TLAR:v_cruise",
                        "data:TLAR:v_approach",
                        "data:mission:sizing:cruise:altitude",
                        "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
                        "data:geometry:propulsion:nacelle:diameter",
                        "data:geometry:propulsion:nacelle:y",
                ],
                method="fd",
        )
        
        self.declare_partials(
                "data:geometry:vertical_tail:area",
                [
                        "data:geometry:wing:area",
                        "data:geometry:wing:span",
                        "data:geometry:wing:MAC:length",
                        "data:weight:aircraft:CG:aft:MAC_position",
                        "data:aerodynamics:fuselage:cruise:CnBeta",
                        "data:aerodynamics:vertical_tail:cruise:CnBeta",
                        "data:TLAR:v_cruise",
                        "data:TLAR:v_approach",
                        "data:mission:sizing:cruise:altitude",
                        "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
                        "data:geometry:propulsion:nacelle:diameter",
                        "data:geometry:propulsion:nacelle:y",
                ],
                method="fd",
        )
        
        self.declare_partials(
                "data:geometry:vertical_tail:area",
                [
                        "data:aerodynamics:fuselage:cruise:CnBeta",
                        "data:TLAR:v_cruise",
                        "data:mission:sizing:cruise:altitude",
                ],
                method="fd",
        )
        
        

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Sizing constraints for the vertical tail.
        # Limiting cases: rotating torque objective (cn_beta_goal) during cruise, and  
        # compensation of engine failure induced torque at approach speed/altitude. 
        # Returns maximum area.

        engine_number = inputs["data:geometry:propulsion:engine:count"]
        wing_area = inputs["data:geometry:wing:area"]
        span = inputs["data:geometry:wing:span"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        cg_mac_position = inputs["data:weight:aircraft:CG:aft:MAC_position"]
        cn_beta_fuselage = inputs["data:aerodynamics:fuselage:cruise:CnBeta"]
        cn_beta_vt = inputs["data:aerodynamics:vertical_tail:cruise:CnBeta"]
        cruise_speed = inputs["data:TLAR:v_cruise"] * 0.514444 # converted to m/s
        approach_speed = inputs["data:TLAR:v_approach"] * 0.514444 # converted to m/s
        cruise_altitude = inputs["data:mission:sizing:cruise:altitude"] 
        wing_htp_distance = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        nac_diam = inputs["data:geometry:propulsion:nacelle:diameter"]
        y_nacelle = inputs["data:geometry:propulsion:nacelle:y"]
        
        
        # CASE1: OBJECTIVE TORQUE @ CRUISE ####################################
        
        atm = Atmosphere(cruise_altitude)
        speed_of_sound = atm.speed_of_sound
        cruise_mach = cruise_speed / speed_of_sound
        # Matches suggested goal by Raymer, Fig 16.20
        cn_beta_goal = 0.0569 - 0.01694 * cruise_mach + 0.15904 * cruise_mach ** 2

        required_cnbeta_vtp = cn_beta_goal - cn_beta_fuselage
        distance_to_cg = wing_htp_distance + 0.25 * l0_wing - cg_mac_position * l0_wing
        vt_area_1 = required_cnbeta_vtp / (distance_to_cg / wing_area / span * cn_beta_vt)
        
        # CASE2: ENGINE FAILURE COMPENSATION###################################
        
        failure_altitude = 5000 # CS23 for Twin engine - at 5000ft
        atm = Atmosphere(failure_altitude)
        speed_of_sound = atm.speed_of_sound
        pressure = atm.pressure
        if engine_number == 2.0:
            stall_speed = approach_speed / 1.3
            MC_speed = 1.2 * stall_speed # Flights mechanics from GA - Serge Bonnet CS23
            MC_mach = MC_speed / speed_of_sound
            # Calculation of engine power for given conditions
            engine_power = 1200 # FIXME: should get engine compute_manual function or max power @ 5000ft
            # Calculation of engine thrust and nacelle drag (failed one) 
            Tmot = engine_power / MC_speed
            Dnac = 0.07 * math.pi * (nac_diam / 2) **2
            # Torque compensation
            vt_area_2 = 2 * (y_nacelle / wing_htp_distance) * (Tmot + Dnac) \
                        / (pressure * MC_mach**2 * 0.9 * 0.42 * 10)
        else:
            vt_area_2 = 0.0
        
        vt_area = max(vt_area_1, vt_area_2)
        wet_vt_area = 2.1 * vt_area

        outputs["data:geometry:vertical_tail:wetted_area"] = wet_vt_area
        outputs["data:geometry:vertical_tail:area"] = vt_area
        outputs["data:aerodynamics:vertical_tail:cruise:CnBeta"] = required_cnbeta_vtp
        