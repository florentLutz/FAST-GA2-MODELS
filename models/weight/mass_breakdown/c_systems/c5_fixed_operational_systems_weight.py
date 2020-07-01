"""
Estimation of fixed operational systems weight
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

import openmdao.api as om


class FixedOperationalSystemsWeight(om.IndepVarComp):
    """
    Weight for fixed operational systems (weather radar, flight recorder, ...) is neglected for GA

    """

    def setup(self):
        
        self.add_output("data:weight:systems:operational:radar:mass", units="kg")
        self.add_output("data:weight:systems:operational:cargo_hold:mass", units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        outputs["data:weight:systems:operational:radar:mass"] = 0.0
        outputs["data:weight:systems:operational:cargo_hold:mass"] = 0.0
