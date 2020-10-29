"""
Estimation of navigation systems weight
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


class ComputeNavigationSystemsWeight(ExplicitComponent):
    """
    Weight estimation for navigation systems

    # TODO: Based on :????????????
    """

    def setup(self):
        
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        
        self.add_output("data:weight:systems:navigation:mass", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        mtow = inputs["data:weight:aircraft:MTOW"]
        
        c3 = 40 + 0.008 * mtow  # mass formula in lb

        outputs["data:weight:systems:navigation:mass"] = c3
