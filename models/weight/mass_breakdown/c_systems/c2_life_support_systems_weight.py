"""
Estimation of life support systems weight
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
from openmdao.core.explicitcomponent import ExplicitComponent
from fastoad.utils.physics import Atmosphere


class LifeSupportSystemsWeight(ExplicitComponent):
    """
    Weight estimation for life support systems

    This includes only air conditioning / pressurization.
    
    Insulation, de-icing, internal lighting system, fixed oxygen, permanent security kits are neglected.
    Seats and installation of crew are already evaluated within d2_passenger_seats_weight

    # TODO: Based on :????????????
    """

    def setup(self):
        
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:TLAR:NPAX", val=np.nan)
        self.add_input("data:weight:systems:navigation:mass", val=np.nan, units="lb")
        self.add_input("data:TLAR:v_limit", val=np.nan, units="m/s")
       
        self.add_output("data:weight:systems:life_support:air_conditioning:mass", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        mtow = inputs["data:weight:aircraft:MTOW"]
        npax = inputs["data:TLAR:NPAX"]
        m_iae = inputs["data:weight:systems:navigation:mass"]
        limit_speed = inputs["data:TLAR:v_limit"]

        atm = Atmosphere(0.0)
        limit_speed = limit_speed/atm.speed_of_sound # converted to mach
        c22 = 0.261*mtow**.52*npax**0.68*m_iae**0.17*limit_speed**0.08 # mass formula in lb
       
        outputs["data:weight:systems:life_support:air_conditioning:mass"] = c22
        