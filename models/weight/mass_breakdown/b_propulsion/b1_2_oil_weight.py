"""
Estimation of engine and associated component weight
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
from ....propulsion.fuel_propulsion.base import FuelEngineSet
from fastoad import BundleLoader


class ComputeOilWeight(ExplicitComponent):
    """
    Weight estimation for motor oil

    Based on : Wells, Douglas P., Bryce L. Horvath, and Linwood A. McCullers. "The Flight Optimization System Weights
    Estimation Method." (2017). Equation 123

    Not used since already included in the engine installed weight but left there in case
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:propulsion:count", val=np.nan)

        self.add_output("data:weight:propulsion:engine_oil:mass", units="lb")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propulsion_model = FuelEngineSet(self._engine_wrapper.get_model(inputs),
                                         inputs["data:geometry:propulsion:count"])

        # This should give the UNINSTALLED weight
        sl_thrust_newton = propulsion_model.compute_sl_thrust()
        sl_thrust_lbs = sl_thrust_newton * 0.224809

        b1_2 = 0.082 * inputs["data:geometry:propulsion:count"] * sl_thrust_lbs ** 0.65

        outputs["data:weight:propulsion:engine_oil:mass"] = b1_2
