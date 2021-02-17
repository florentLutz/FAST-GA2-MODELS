"""OpenMDAO wrapping of basic IC engine."""
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
from openmdao.core.component import Component

from fastoad.models.propulsion.propulsion import IOMPropulsionWrapper
from fastoad.module_management.service_registry import RegisterPropulsion
from fastoad.openmdao.validity_checker import ValidityDomainChecker

from ...propulsion import IPropulsion, BaseOMPropulsionComponent
from .basicIC_engine import BasicICEngine


@RegisterPropulsion(
    "fastga.wrapper.propulsion.basicIC_engine",
    desc="""
Parametric ICE engine-propeller model as OpenMDAO component.

Implementation of basic scaled power propeller-engine model with fixed efficiency.
For more information, see BasicICEngine class in FAST-OAD developer documentation.
""",
)
class OMBasicICEngineWrapper(IOMPropulsionWrapper):
    """
    Wrapper class of for basic IC engine model.
    It is made to allow a direct call to :class:`~.basicIC_engine.BasicICEngine` in an OpenMDAO
    component.
    Example of usage of this class::
        import openmdao.api as om
        class MyComponent(om.ExplicitComponent):
            def initialize():
                self._engine_wrapper = OMRubberEngineWrapper()
            def setup():
                # Adds OpenMDAO variables that define the engine
                self._engine_wrapper.setup(self)
                # Do the normal setup
                self.add_input("my_input")
                [finish the setup...]
            def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
                [do something]
                # Get the engine instance, with parameters defined from OpenMDAO inputs
                engine = self._engine_wrapper.get_model(inputs)
                # Run the engine model. This is a pure Python call. You have to define
                # its inputs before, and to use its outputs according to your needs
                sfc, thrust_rate, thrust = engine.compute_flight_points(
                    mach,
                    altitude,
                    engine_setting,
                    thrust_is_regulated,
                    thrust_rate,
                    thrust
                    )
                [do something else]
        )
    """

    def setup(self, component: Component):
        component.add_input("data:propulsion:IC_engine:max_power", np.nan, units="W")
        component.add_input("data:propulsion:IC_engine:fuel_type", np.nan)
        component.add_input("data:propulsion:IC_engine:strokes_nb", np.nan)
        component.add_input("data:TLAR:v_cruise", np.nan, units="m/s")
        component.add_input("data:mission:sizing:main_route:cruise:altitude", np.nan, units="m")
        component.add_input("data:geometry:propulsion:layout", np.nan)

    @staticmethod
    def get_model(inputs) -> IPropulsion:
        """
        :param inputs: input parameters that define the engine
        :return: an :class:`BasicICEngine` instance
        """
        engine_params = {
            "max_power": inputs["data:propulsion:IC_engine:max_power"],
            "design_altitude": inputs["data:mission:sizing:main_route:cruise:altitude"],
            "design_speed": inputs["data:TLAR:v_cruise"],
            "fuel_type": inputs["data:propulsion:IC_engine:fuel_type"],
            "strokes_nb": inputs["data:propulsion:IC_engine:strokes_nb"],
            "prop_layout": inputs["data:geometry:propulsion:layout"]
        }

        return BasicICEngine(**engine_params)


@ValidityDomainChecker(
    {
        "data:propulsion:IC_engine:max_power": (50000, 250000),  # power range validity
        "data:propulsion:IC_engine:fuel_type": [1.0, 2.0],  # fuel list
        "data:propulsion:IC_engine:strokes_nb": [2.0, 4.0],  # architecture list
        "data:geometry:propulsion:layout": [1.0, 3.0],  # propulsion position (3.0=Nose, 1.0=Wing)
    }
)
class OMBasicICEngineComponent(BaseOMPropulsionComponent):
    """
    Parametric engine model as OpenMDAO component
    See :class:`BasicICEngine` for more information.
    """

    def setup(self):
        super().setup()
        self.get_wrapper().setup(self)

    @staticmethod
    def get_wrapper() -> OMBasicICEngineWrapper:
        return OMBasicICEngineWrapper()
