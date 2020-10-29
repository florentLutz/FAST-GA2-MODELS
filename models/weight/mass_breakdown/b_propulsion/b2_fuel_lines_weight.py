"""
Estimation of fuel lines weight
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


class ComputeFuelLinesWeight(ExplicitComponent):
    """
    Weight estimation for fuel lines
    
    # TODO: Based on :????????????
    """

    def setup(self):
        
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)
        self.add_input("data:weight:aircraft:MFW", val=np.nan, units="lb")
        
        self.add_output("data:weight:propulsion:fuel_lines:mass", units="lb")

        self.declare_partials(
            "data:weight:propulsion:fuel_lines:mass", "data:weight:aircraft:MFW", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        tank_nb = 2  # Number of fuel tanks is assumed to be two, 1 per semi-wing
        engine_nb = inputs["data:geometry:propulsion:engine:count"]
        fuel_mass = inputs["data:weight:aircraft:MFW"]
        
        b2 = 2.49*(
                (fuel_mass/5.87)**0.6 * 0.5**0.3
                * tank_nb**0.2 * engine_nb**0.13
        )**1.21  # mass formula in lb
        
        outputs["data:weight:propulsion:fuel_lines:mass"] = b2
