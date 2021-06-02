"""Base module for propulsion models."""
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA/ISAE
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

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import openmdao.api as om
import pandas as pd

from fastoad.base.dict import AddKeyAttributes
from fastoad.base.flight_point import FlightPoint
from fastoad.models.propulsion.propulsion import IOMPropulsionWrapper


class IPropulsion(ABC):
    """
    Interface that should be implemented by propulsion models.

    Using this class allows to delegate to the propulsion model the management of
    propulsion-related data when computing performances.

    The performance model calls :meth:`compute_flight_points` by providing one or
    several flight points. The method will feed these flight points with results
    of the model (e.g. thrust, SFC, ..).

    The performance model will then be able to call :meth:`get_consumed_mass` to
    know the mass consumption for each flight point.

    Note::

        If the propulsion model needs fields that are not among defined fields
        of the :class`FlightPoint class`, these fields can be made authorized by
        :class`FlightPoint class` by putting such command before defining the
        class::

            >>> # Simply adds the fields:
            >>> AddKeyAttributes(["ion_drive_power", "warp"])(FlightPoint)
            >>> # Adds the fields with associated default values:
            >>> AddKeyAttributes({"ion_drive_power":110., "warp":9.0})(FlightPoint)
    """

    @abstractmethod
    def compute_flight_points(self, flight_points: Union[FlightPoint, pd.DataFrame]):
        """
        Computes Specific Fuel Consumption according to provided conditions.

        See :class:`FlightPoint` for available fields that may be used for computation.
        If a DataFrame instance is provided, it is expected that its columns match
        field names of FlightPoint (actually, the DataFrame instance should be
        generated from a list of FlightPoint instances).

        .. note:: **About thrust_is_regulated, thrust_rate and thrust**

            :code:`thrust_is_regulated` tells if a flight point should be computed using
            :code:`thrust_rate` (when False) or :code:`thrust` (when True) as input. This way,
            the method can be used in a vectorized mode, where each point can be set to respect
            a **thrust** order or a **thrust rate** order.

            - if :code:`thrust_is_regulated` is not defined, the considered input will be the
              defined one between :code:`thrust_rate` and :code:`thrust` (if both are provided,
              :code:`thrust_rate` will be used)

            - if :code:`thrust_is_regulated` is :code:`True` or :code:`False` (i.e., not a sequence),
              the considered input will be taken accordingly, and should of course be defined.

            - if there are several flight points, :code:`thrust_is_regulated` is a sequence or array,
              :code:`thrust_rate` and :code:`thrust` should be provided and have the same shape as
              :code:`thrust_is_regulated:code:`. The method will consider for each element which input
              will be used according to :code:`thrust_is_regulated`.


        :param flight_points: FlightPoint or DataFrame instance
        :return: None (inputs are updated in-place)
        """

    @abstractmethod
    def compute_weight(self) -> float:
        """
        Computes total propulsion mass.

        :return: the total uninstalled mass in kg
        """

    @abstractmethod
    def compute_max_power(self, flight_points: Union[FlightPoint, pd.DataFrame]):
        """
        Computes max available power on one engine.

        :return: the maximum available power in W
        """

    @abstractmethod
    def compute_dimensions(self) -> (float, float, float, float):
        """
        Computes propulsion dimensions.

        :return: (height, width, length, wet area) of nacelle in m or m²
        """

    @abstractmethod
    def compute_drag(
            self,
            mach: Union[float, np.array],
            unit_reynolds: Union[float, np.array],
            wing_mac: float,
    ) -> Union[float, np.array]:
        """
        Computes nacelle drag force for out of fuselage engine.

        :param mach: mach at which drag should be calculated
        :param unit_reynolds: unitary Reynolds for calculation
        :param wing_mac: wing MAC length in m
        :return: drag force cd0*wing_area
        """

    @abstractmethod
    def get_consumed_mass(self, flight_point: FlightPoint, time_step: float) -> float:
        """
        Computes consumed mass for provided flight point and time step.

        This method should rely on FlightPoint fields that are generated by
        :meth: `compute_flight_points`.

        :param flight_point:
        :param time_step:
        :return: the consumed mass in kg
        """


class BaseOMPropulsionComponent(om.ExplicitComponent, ABC):
    """
    Base class for OpenMDAO wrapping of subclasses of :class:`IEngineForOpenMDAO`.

    Classes that implements this interface should add their own inputs in setup()
    and implement :meth:`get_wrapper`.
    """

    def initialize(self):
        self.options.declare("flight_point_count", 1, types=(int, tuple))

    def setup(self):
        shape = self.options["flight_point_count"]
        self.add_input("data:propulsion:mach", np.nan, shape=shape)
        self.add_input("data:propulsion:altitude", np.nan, shape=shape, units="m")
        self.add_input("data:propulsion:engine_setting", np.nan, shape=shape)
        self.add_input("data:propulsion:use_thrust_rate", np.nan, shape=shape)
        self.add_input("data:propulsion:required_thrust_rate", np.nan, shape=shape)
        self.add_input("data:propulsion:required_thrust", np.nan, shape=shape, units="N")

        self.add_output("data:propulsion:SFC", shape=shape, units="kg/s/N")
        self.add_output("data:propulsion:thrust_rate", shape=shape)
        self.add_output("data:propulsion:thrust", shape=shape, units="N")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        wrapper = self.get_wrapper().get_model(inputs)
        flight_point = FlightPoint(
            mach=inputs["data:propulsion:mach"],
            altitude=inputs["data:propulsion:altitude"],
            engine_setting=inputs["data:propulsion:engine_setting"],
            thrust_is_regulated=np.logical_not(
                inputs["data:propulsion:use_thrust_rate"].astype(int)
            ),
            thrust_rate=inputs["data:propulsion:required_thrust_rate"],
            thrust=inputs["data:propulsion:required_thrust"],
        )
        wrapper.compute_flight_points(flight_point)
        outputs["data:propulsion:SFC"] = flight_point.sfc
        outputs["data:propulsion:thrust_rate"] = flight_point.thrust_rate
        outputs["data:propulsion:thrust"] = flight_point.thrust

    @staticmethod
    @abstractmethod
    def get_wrapper() -> IOMPropulsionWrapper:
        """
        This method defines the used :class:`IOMPropulsionWrapper` instance.

        :return: an instance of OpenMDAO wrapper for propulsion model
        """
