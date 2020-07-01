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
from fastoad.models.propulsion.fuel_engine.basicIC_engine import FastBasicICEngineInconsistentInputParametersError


# FIXME:  the weight estimation of the engine should be defined within the engine model (handle hybrid architecture)
class EngineWeight(ExplicitComponent):
    """
    Engine weight estimation

    """

    def setup(self):
        
        self.add_input("data:propulsion:engine:power_SL", val=np.nan, units="W") # Power @ see level
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)
        self.add_input("data:propulsion:engine:fuel_type", val=np.nan)
        self.add_input("data:propulsion:engine:n_strokes", val=np.nan)
        self.add_output("data:weight:propulsion:engine:mass", units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        power_sl = inputs["data:propulsion:engine:power_SL"]
        n_engines = inputs["data:geometry:propulsion:engine:count"]
        fuel_type = inputs["data:propulsion:engine:fuel_type"]
        n_strokes = inputs["data:propulsion:engine:n_strokes"]
        
        if (fuel_type == 1.0) and (n_strokes == 2.0):
            b1 = 0.593*power_sl + 8.0199
        elif (fuel_type == 1.0) and (n_strokes == 4.0):
            b1 = (0.7258*power_sl/n_engines + 32.223)*n_engines
        elif (fuel_type == 2.0) and (n_strokes == 2.0):
            b1 = 1.0053*power_sl + 24.363
        elif (fuel_type == 2.0) and (n_strokes == 4.0):
            b1 = (0.8755*power_sl/n_engines + 46.469)*n_engines
        else:
            raise FastBasicICEngineInconsistentInputParametersError(
                "Bad motor configuration: only 2 or 4-strokes and fuel type 1/2 available."
            )
            
        outputs["data:weight:propulsion:engine:mass"] = b1
