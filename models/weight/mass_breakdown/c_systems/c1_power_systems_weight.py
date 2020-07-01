"""
Estimation of power systems weight
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


class PowerSystemsWeight(ExplicitComponent):
    """
    Weight estimation for power systems (generation and distribution)

    # TODO: Based on :????????????
    """

    def setup(self):
        
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:weight:propulsion:fuel_lines:mass", val=np.nan, units="kg")
        self.add_input("data:weight:systems:navigation:mass", val=np.nan, units="kg")
        self.add_output("data:weight:systems:power:auxiliary_power_unit:mass", units="kg") # old weight_C11
        self.add_output("data:weight:systems:power:electric_systems:mass", units="kg") # old weight_C12
        self.add_output("data:weight:systems:power:hydraulic_systems:mass", units="kg") # old weight_C13

        self.declare_partials("*", "*", method="fd")

    # pylint: disable=too-many-locals
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        mtow = inputs["data:weight:aircraft:MTOW"]*2.20462 # Takeoff weight in lb
        m_fuel_lines = inputs["data:weight:propulsion:fuel_lines:mass"]*2.20462 # converted to lb
        m_iae = inputs["data:weight:systems:navigation:mass"]*2.20462 # converted to lb
        c12 = 426*((m_fuel_lines+m_iae)/1000)**0.51 # mass in lb
        c13 = 0.007*mtow # mass in lb
        
        outputs["data:weight:systems:power:auxiliary_power_unit:mass"] = 0.0 # no APU on general aircraft
        outputs["data:weight:systems:power:electric_systems:mass"] = c12/ 2.20462 # converted to kg
        outputs["data:weight:systems:power:hydraulic_systems:mass"] = c13/ 2.20462 # converted to kg
