"""
Estimation of horizontal tail area
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
from scipy.constants import g
from fastoad.models.aerodynamics.aerodynamics_functions import HorizontalTailAero
from fastoad.utils.physics import Atmosphere

class ComputeHTArea(om.ExplicitComponent):
    """
    Computes area of horizontal tail plane

    Area is computed to fulfill aircraft balance requirement at rotation speed
    """

    def setup(self):
        
        self.add_input("data:geometry:has_T_tail", val=np.nan)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Sizing constraints for the horizontal tail (methods from Torenbeek).
        # Limiting cases: Rotating power at takeoff/landing, with the most 
        # forward CG position. Returns maximum area.

        n_engines = inputs["data:geometry:propulsion:engine:count"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        cl_max_takeoff = inputs["data:aerodynamics:aircraft:landing:CL_max"]
        wing_area = inputs["data:geometry:wing:area"]
        x_wing_aero_center = inputs["data:geometry:wing:MAC:at25percent:x"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        wing_mac = inputs["data:geometry:wing:MAC:length"]
        cg_range = inputs["settings:weight:aircraft:CG:range"]
        x_cg_aft = inputs["data:weight:aircraft:CG:aft:x"]
        thrust_height = inputs["data:geometry:propulsion:engine:z:from_aeroCenter"]
        cl_max_landing = inputs["data:aerodynamics:aircraft:landing:CL_max"]
        mlw = inputs["data:weight:aircraft:MLW"]
        x_lg = inputs["data:weight:airframe:landing_gear:main:CG:x"]
        z_eng = inputs["data:geometry:propulsion:engine:z:from_wingMAC25"]
        max_thrust = inputs["data:propulsion:MTO_thrust"]
        landing_t_rate = inputs["data:mission:operational:landing:thrust_rate"]
        takeoff_t_rate = inputs["data:mission:operational:takeoff:thrust_rate"]
        
        # Conditions for calculation
        atm = Atmosphere(0.0)
        rho = atm.density
        speed_of_sound = atm.speed_of_sound
        g = 9.81
        ang_vel = 12 * math.pi/180 #12 deg/s (typical for light aircraft)
        
        # CASE1: TAKE-OFF######################################################
        # method extracted from Torenbeek 1982 p325
        
        # Calculation of take-off minimum speed
        weight = mtow * g
        vs1 = math.sqrt(weight / (0.5*rho*wing_area*cl_max_takeoff))
        # Rotation speed requirement from FAR 23.51 (depends on number of engines)
        if n_engines==1:
            v_r = vs1 * 1.0
        else:
            v_r = vs1 * 1.1
        
        # Definition of max forward gravity center position
        x_cg = x_cg_aft - cg_range * wing_mac
        # Definition of horizontal tail global position
        x_ht = x_wing_aero_center + lp_ht
        # Calculation of wheel factor
        thrust = max_thrust * takeoff_t_rate
        fact_wheel = (x_lg - x_cg - z_eng*thrust/weight) / wing_mac \
                    * (vs1/v_r)**2
        # Compute aerodynamic coefficients for takeoff
        cl_ht, cl_alpha_ht, _ = HorizontalTailAero(aoa=0.0, flaps_angle=10.0, elevator_angle=-25.0, mach=v_r/speed_of_sound)
        cl_ht *= wing_area/ht_area
        cl_alpha_ht *= wing_area/ht_area
        cm_wing = 3.0 # FIXME: find an efficient way to connect complete aero evaluation (1 function call: maybe take-off?)
        # Calculation of correction coefficient n_h and n_q            
        n_h = (x_ht-x_lg) / l_h * 0.9 # 0.9=(v_h/v_r)²: dynamic pressure reduction at tail (typical value)  
        n_q = 1 + cl_alpha_ht/cl_ht * ang_vel * (x_ht - x_lg) / v_r
                    
        # Calculation of volume coefficient based on Torenbeek formula
        coef_vol = cl_max/(n_h * n_q * cl_ht) * (cm_wing/cl_max - fact_wheel) \
                    + cl_r/cl_ht *(x_lg - ac_wing)/wing_mac
        # Calulation of equivalent area
        area_1 = coef_vol * wing_area * wing_mac / lp_ht
        
        # CASE2: LANDING#######################################################
        
        # Calculation of take-off minimum speed
        weight = mlw * g
        vs1 = math.sqrt(weight / (0.5*rho*wing_area*cl_max_landing))
        # Rotation speed correction
        v_r = vs1 * 1.3
        # Calculation of wheel factor
        thrust = max_thrust * landing_t_rate
        fact_wheel = (x_lg - x_cg - z_eng*thrust/weight) / wing_mac \
                    * (vs1/v_r)**2
        # Evaluate aircraft overall angle (aoa)
        cl_r = weight / (0.5 * rho * v_r**2 * wing_area)
        cl_flaps = DeltaClHighLift(flaps_angle=30.0, slats_anle=0.0, mach=v_r/speed_of_sound)
        alpha = (cl_r - cl_0 - cl_flaps) / cl_alpha * 180/math.pi
        # Compute aerodynamic coefficients for landing
        cl_ht, cl_alpha_ht, _ = HorizontalTailAero(aoa=alpha, flaps_angle=30, elevator_angle=-25.0, mach=v_r/speed_of_sound)
        cl_ht *= wing_area/ht_area
        cl_alpha_ht *= wing_area/ht_area
        cm_wing = 3.0 # FIXME: find an efficient way to connect complete aero evaluation (1 function call: maybe take-off?)
        # Calculation of correction coefficient n_h and n_q            
        n_h = (x_ht-x_lg) / l_h * 0.9 # 0.9=(v_h/v_r)²: dynamic pressure reduction at tail (typical value)  
        n_q = 1 + cl_alpha_ht/cl_ht * ang_vel * (x_ht - x_lg) / v_r
        
        # Calculation of volume coefficient based on Torenbeek formula
        coef_vol = cl_max/(n_h * n_q * cl_ht) * (cm_wing/cl_max - fact_wheel) \
                    + cl_r/cl_ht *(x_lg - ac_wing)/wing_mac
        # Calulation of equivalent area
        area_2 = coef_vol * wing_area * wing_mac / lp_ht
        
        outputs["data:geometry:horizontal_tail:area"] = max(area_1, area_2)
