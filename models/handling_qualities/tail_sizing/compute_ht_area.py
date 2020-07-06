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
from fastoad.models.aerodynamics.components.cl_ht import ClHorizontalTail


class ComputeHTArea(om.ExplicitComponent):
    """
    Computes area of horizontal tail plane

    Area is computed to fulfill aircraft balance requirement at rotation speed
    """

    def setup(self):
        
        # TODO: complete setup

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Sizing constraints for the horizontal tail (methods from Torenbeek).
        # Limiting cases: Rotating power at takeoff/landing, with the most 
        # forward CG position. Returns maximum area.

        n_engines = inputs["data:geometry:propulsion:engine:count"]
        tail_type = np.round(inputs["data:geometry:has_T_tail"])
        mtow = inputs["data:weight:aircraft:MTOW"]
        cl_max_takeoff = inputs["data:aerodynamics:aircraft:landing:CL_max"]
        wing_area = inputs["data:geometry:wing:area"]
        x_wing_aero_center = inputs["data:geometry:wing:MAC:at25percent:x"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        wing_mac = inputs["data:geometry:wing:MAC:length"]
        cg_range = inputs["settings:weight:aircraft:CG:range"]
        x_cg_aft = inputs["data:weight:aircraft:CG:aft:x"]
        x_wing_aero_center = inputs["data:geometry:wing:MAC:at25percent:x"]
        thrust_height = inputs["data:geometry:propulsion:engine:z:from_aeroCenter"]
        cl_max_landing = inputs["data:aerodynamics:aircraft:landing:CL_max"]
        mlw = inputs["data:weight:aircraft:MLW"]
        
        atm = Atmosphere(0.0)
        rho = atm.density
        
        # CASE1: TAKE-OFF######################################################
        
        # Calculation of landing minimum speed
        vspeed = math.sqrt(mtow * g / (0.5*rho*wing_area*cl_max_takeoff))
        # Speed requirement from FAR 23.51 (depends on number of engines)
        if n_engines==1:
            vspeed = vspeed * 1.0
        else:
            vspeed = vspeed * 1.1
        
        # Calculation of gravity center position
        x_cg = x_cg_aft - cg_range * wing_mac
        
        # Calculation of horizontal tail lift coefficient and dynamic pressure
        cl_ht, _ = ClHorizontalTail(speed=vspeed, flaps_angle=10.0, elevator_angle = -25.0) # TODO: to be completed with necessary data
        pdyn = (0.5 * rho * vspeed ** 2) * 0.9 # NOTE: typical dynamic pressure reduction at tail

        # Calculation of thrust
        thrust = 0.0 # !!!: neglected for the moment
        
        # Moment equilibrium calculation
        T_ht = ((x_cg-x_wing_aero_center) * mtow * g + thrust * thrust_height)
        htp_area_1 = T_ht / lp_ht / (cl_ht*pdyn)
        
        # CASE2: LANDING#######################################################
        
        # Calculation of landing minimum speed
        vspeed = math.sqrt(mlw * g / (0.5*rho*wing_area*cl_max_landing))
        # TODOC: Speed requirement ...
        vspeed = vspeed * 1.3
        
        # Calculation of horizontal tail lift coefficient and dynamic pressure
        cl_ht, _ = ClHorizontalTail(speed=vspeed, flaps_angle=30.0, elevator_angle = -25.0) # TODO: to be completed with necessary data
        pdyn = (0.5 * rho * vspeed ** 2)
        
        # Calculation of thrust
        thrust = 0.0 # !!!: neglected for the moment
        
        # Moment equilibrium calculation
        T_ht = ((x_cg-x_wing_aero_center) * mtow * g + thrust * thrust_height)
        htp_area_2 = T_ht / lp_ht / (cl_ht*pdyn)
        
        # EVALUATION OF MAXIMUM AREA ##########################################
        
        htp_area = max(htp_area_1, htp_area_2)

        if tail_type == 1:
            wet_area_coeff = 2.0 * 1.05 #k_b coef from Gudmunnson p.707
        elif tail_type == 0:
            wet_area_coeff = 1.6 * 1.05 #k_b coef from Gudmunnson p.707
        else:
            raise ValueError("Value of data:geometry:has_T_tail can only be 0 or 1")
        wet_area_htp = wet_area_coeff * htp_area
        
        outputs["data:geometry:horizontal_tail:wetted_area"] = wet_area_htp
        outputs["data:geometry:horizontal_tail:area"] = htp_area
