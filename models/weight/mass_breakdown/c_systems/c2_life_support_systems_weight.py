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


class LifeSupportSystemsWeight(ExplicitComponent):
    """
    Weight estimation for life support systems

    This includes only air conditioning / pressurization.
    
    Insulation, de-icing, internal lighting system, fixed oxygen, permanent security kits are neglected.
    Seats and installation of crew are already evaluated within d2_passenger_seats_weight

    # TODO: Based on :????????????
    """

    def setup(self):
        
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:TLAR:NPAX", val=np.nan)
        self.add_input("data:weight:systems:navigation:mass", val=np.nan, units="kg")
        self.add_input("data:TLAR:limit_speed", val=np.nan, units="kn") # TODO: replace vne, to be added to xml variables
        self.add_output("data:weight:systems:life_support:insulation:mass", units="kg") # old weight_C21
        self.add_output("data:weight:systems:life_support:air_conditioning:mass", units="kg") # old weight_C22
        self.add_output("data:weight:systems:life_support:de-icing:mass", units="kg") # old weight_C23
        self.add_output("data:weight:systems:life_support:cabin_lighting:mass", units="kg") # old weight_C24
        self.add_output(
            "data:weight:systems:life_support:seats_crew_accommodation:mass", units="kg"
        ) # old weight_C25
        self.add_output("data:weight:systems:life_support:oxygen:mass", units="kg") # old weight_C26
        self.add_output("data:weight:systems:life_support:safety_equipment:mass", units="kg") # old weight_C27

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        mtow = inputs["data:weight:aircraft:MTOW"]*2.20462 # Takeoff weight in lb
        npax = inputs["data:TLAR:NPAX"]
        m_iae = inputs["data:weight:systems:navigation:mass"]*2.20462 # converted to lb
        limit_speed = inputs["data:TLAR:limit_speed"]/666.739 # converted to mach
        c22 = 0.261*mtow**.52*npax**0.68*m_iae**0.17*limit_speed**0.08 # mass in lb
       
        outputs["data:weight:systems:life_support:insulation:mass"] = 0.0
        outputs["data:weight:systems:life_support:air_conditioning:mass"] = c22/ 2.20462 # converted to kg
        outputs["data:weight:systems:life_support:de-icing:mass"] = 0.0
        outputs["data:weight:systems:life_support:cabin_lighting:mass"] =0.0
        outputs["data:weight:systems:life_support:seats_crew_accommodation:mass"] = 0.0
        outputs["data:weight:systems:life_support:oxygen:mass"] = 0.0
        outputs["data:weight:systems:life_support:safety_equipment:mass"] = 0.0
        