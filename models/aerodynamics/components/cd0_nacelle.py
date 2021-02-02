"""
    FAST - Copyright (c) 2016 ONERA ISAE
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

import math

import numpy as np
import warnings
from openmdao.core.explicitcomponent import ExplicitComponent
from ...propulsion.fuel_propulsion.base import FuelEngineSet
from fastoad import BundleLoader


class Cd0Nacelle(ExplicitComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)
        self.options.declare("propulsion_id", default="", types=str)
        
    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:propulsion:count", val=np.nan)
        # self.add_input("data:geometry:propulsion:layout", val=np.nan), in the engine
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
            self.add_input("data:aerodynamics:low_speed:unit_reynolds", val=np.nan, units="m**-1")
            self.add_output("data:aerodynamics:nacelles:low_speed:CD0")
        else:
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:aerodynamics:cruise:unit_reynolds", val=np.nan, units="m**-1")
            self.add_output("data:aerodynamics:nacelles:cruise:CD0")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propulsion_model = FuelEngineSet(self._engine_wrapper.get_model(inputs), 1.0)
        engine_number = inputs["data:geometry:propulsion:count"]
        prop_layout = inputs["data:geometry:propulsion:layout"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        wing_area = inputs["data:geometry:wing:area"]
        if self.options["low_speed_aero"]:
            mach = inputs["data:aerodynamics:low_speed:mach"]
            unit_reynolds = inputs["data:aerodynamics:low_speed:unit_reynolds"]
        else:
            mach = inputs["data:aerodynamics:cruise:mach"]
            unit_reynolds = inputs["data:aerodynamics:cruise:unit_reynolds"]

        drag_force = propulsion_model.compute_drag(mach, unit_reynolds, l0_wing)

        if (prop_layout == 1.0) or (prop_layout == 2.0):
            cd0 = drag_force / wing_area * engine_number
        elif prop_layout == 3.0:
            cd0 = 0.0
        else:
            cd0 = 0.0
            warnings.warn('Propulsion layout {} not implemented in model, replaced by layout 1!'.format(prop_layout))

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:nacelles:low_speed:CD0"] = cd0
        else:
            outputs["data:aerodynamics:nacelles:cruise:CD0"] = cd0
