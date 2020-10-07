"""
Estimation of engine weight
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
from ....propulsion.fuel_propulsion.basicIC_engine.exceptions import FastBasicICEngineInconsistentInputParametersError


# FIXME:  the weight estimation of the engine should be defined within the engine model (handle hybrid architecture)
class ComputeEngineWeight(ExplicitComponent):
    """
    Engine weight estimation

    """

    def setup(self):
        
        self.add_input("data:propulsion:engine:power_SL", val=np.nan, units="hp")
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)
        
        self.add_output("data:weight:propulsion:engine:mass", units="lb")

        self.declare_partials("data:weight:propulsion:engine:mass",
            ["data:propulsion:engine:power_SL"],
            method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        power_sl = inputs["data:propulsion:engine:power_SL"] * (746/735.5) # conversion to european hp
        n_engines = inputs["data:geometry:propulsion:engine:count"]
        
        b1 = ((power_sl - 21.55)/0.5515)*n_engines
            
        outputs["data:weight:propulsion:engine:mass"] = b1
