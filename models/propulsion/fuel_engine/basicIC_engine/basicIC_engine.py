"""Parametric propeller IC engine."""
# -*- coding: utf-8 -*-
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

import logging
from typing import Union, Sequence, Tuple, Optional

import numpy as np
import pandas as pd

from fastoad.base.flight_point import FlightPoint
from fastoad.constants import EngineSetting
from fastoad.exceptions import FastUnknownEngineSettingError
from fastoad.models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from ..basicIC_engine.exceptions import FastBasicICEngineInconsistentInputParametersError
from fastoad.utils.physics import Atmosphere

# Logger for this module
_LOGGER = logging.getLogger(__name__)

class BasicICEngine(AbstractFuelPropulsion):   

    def __init__(
        self,  
        power_sea_level: float, 
        fuel_type: float, 
        strokes_nb: float,
        delta_t4_climb: float = -50,
        delta_t4_cruise: float = -100,
    ):
        """
        Parametric turboprop engine.

        It computes engine characteristics using analytical model

        :param power_sea_level: (unit=kW)
        :param fuel_type: 1.0 for gasoline and 2.0 for gasoil engine
        :param strokes_nb: can be either 2-strockes (=2.0) or 4-strockes (=4.0) 
        """
        
        if (fuel_type!=1.0) and (fuel_type!=2.0):
            raise FastBasicICEngineInconsistentInputParametersError(
                "Bad engine configuration: fuel type {0:d} does not exist.".format(fuel_type)
            )
        
        if (strokes_nb!=2.0) and (strokes_nb!=4.0):
            raise FastBasicICEngineInconsistentInputParametersError(
                "Bad engine configuration: {0:d}-strokes does not exist.".format(strokes_nb)
            )
        
        self.power_sea_level=power_sea_level
        self.altitude = None
        self.mach = None
        self.fuel_type = fuel_type
        self.strokes_nb = strokes_nb
        
        ## This dictionary is expected to have a dT4 value for all EngineSetting values
        self.dt4_values = {
            EngineSetting.TAKEOFF: 0.0,
            EngineSetting.CLIMB: delta_t4_climb,
            EngineSetting.CRUISE: delta_t4_cruise,
            EngineSetting.IDLE: delta_t4_cruise,
        }

        # ... so check that all EngineSetting values are in dict
        unknown_keys = [key for key in EngineSetting if key not in self.dt4_values.keys()]
        if unknown_keys:
            raise FastUnknownEngineSettingError("Unknown flight phases: %s", unknown_keys)
        
    def compute_flight_points(self, flight_points: Union[FlightPoint, pd.DataFrame]):
        # pylint: disable=too-many-arguments  # they define the trajectory
        sfc, thrust_rate, thrust = self.compute_flight_points_from_dt4(
            flight_points.mach,
            flight_points.altitude,
            self._get_delta_t4(flight_points.engine_setting),
            flight_points.thrust_is_regulated,
            flight_points.thrust_rate,
            flight_points.thrust,
        )
        flight_points.sfc = sfc
        flight_points.thrust_rate = thrust_rate
        flight_points.thrust = thrust
        
    def compute_flight_points_from_dt4(
        self,
        mach: Union[float, Sequence],
        altitude: Union[float, Sequence],
        delta_t4: Union[float, Sequence],
        use_thrust_rate: Optional[Union[bool, Sequence]] = None,
        thrust_rate: Optional[Union[float, Sequence]] = None,
        thrust: Optional[Union[float, Sequence]] = None,
    ) -> Tuple[Union[float, Sequence], Union[float, Sequence], Union[float, Sequence]]:
        """
        Same as :meth:`compute_flight_points` except that delta_t4 is used directly
        instead of specifying flight phase.

        :param mach: Mach number
        :param altitude: (unit=m) altitude w.r.t. to sea level
        :param delta_t4: (unit=K) difference between operational and design values of
                         turbine inlet temperature in K
        :param use_thrust_rate: tells if thrust_rate or thrust should be used (works element-wise)
        :param thrust_rate: thrust rate (unit=none)
        :param thrust: required thrust (unit=N)
        :return: SFC (in kg/s/N), thrust rate, thrust (in N)
        """
        mach = np.asarray(mach)
        altitude = np.asarray(altitude)
        delta_t4 = np.asarray(delta_t4)

        if use_thrust_rate is not None:
            use_thrust_rate = np.asarray(np.round(use_thrust_rate, 0), dtype=bool)

        use_thrust_rate, thrust_rate, thrust = self._check_thrust_inputs(
            use_thrust_rate, thrust_rate, thrust
        )

        use_thrust_rate = np.asarray(np.round(use_thrust_rate, 0), dtype=bool)
        thrust_rate = np.asarray(thrust_rate)
        thrust = np.asarray(thrust)

        atmosphere = Atmosphere(altitude, delta_t4, altitude_in_feet=False)
        speed = mach * atmosphere.speed_of_sound

        max_power = self.max_power(atmosphere) * 1000 # Transformation kW -> W
        max_thrust = max_power / speed # FIXME: define max_thrust function for low speed!
        
        # We compute thrust values from thrust rates when needed
        idx = use_thrust_rate
        if np.size(max_thrust) == 1:
            maximum_thrust = max_thrust
            out_thrust_rate = thrust_rate
            out_thrust = thrust
        else:
            out_thrust_rate = (
                np.full(np.shape(max_thrust), thrust_rate.item())
                if np.size(thrust_rate) == 1
                else thrust_rate
            )
            out_thrust = (
                np.full(np.shape(max_thrust), thrust.item()) if np.size(thrust) == 1 else thrust
            )

            maximum_thrust = max_thrust[idx]

        out_thrust[idx] = out_thrust_rate[idx] * maximum_thrust

        # thrust_rate is obtained from entire thrust vector (could be optimized if needed,
        # as some thrust rates that are computed may have been provided as input)
        out_thrust_rate = out_thrust / max_thrust
        
        # Now SFC can be computed
        sfc_0 = self.sfc_at_max_thrust(atmosphere, mach)
        sfc = sfc_0 * self.sfc_ratio(altitude, out_thrust_rate)
        
        return sfc, out_thrust_rate, out_thrust
        
        
    @staticmethod
    def _check_thrust_inputs(
        use_thrust_rate: Optional[Union[float, Sequence]],
        thrust_rate: Optional[Union[float, Sequence]],
        thrust: Optional[Union[float, Sequence]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Checks that inputs are consistent and return them in proper shape.

        Some of the inputs can be None, but outputs will be proper numpy arrays.

        :param use_thrust_rate:
        :param thrust_rate:
        :param thrust:
        :return: the inputs, but transformed in numpy arrays.
        """
        # Ensure they are numpy array
        if use_thrust_rate is not None:
            # As OpenMDAO may provide floats that could be slightly different
            # from 0. or 1., a rounding operation is needed before converting
            # to booleans
            use_thrust_rate = np.asarray(np.round(use_thrust_rate, 0), dtype=bool)
        if thrust_rate is not None:
            thrust_rate = np.asarray(thrust_rate)
        if thrust is not None:
            thrust = np.asarray(thrust)

        # Check inputs: if use_thrust_rate is None, we will use the provided input between
        # thrust_rate and thrust
        if use_thrust_rate is None:
            if thrust_rate is not None:
                use_thrust_rate = True
                thrust = np.empty_like(thrust_rate)
            elif thrust is not None:
                use_thrust_rate = False
                thrust_rate = np.empty_like(thrust)
            else:
                raise FastBasicICEngineInconsistentInputParametersError(
                    "When use_thrust_rate is None, either thrust_rate or thrust should be provided."
                )

        elif np.size(use_thrust_rate) == 1:
            # Check inputs: if use_thrust_rate is a scalar, the matching input(thrust_rate or
            # thrust) must be provided.
            if use_thrust_rate:
                if thrust_rate is None:
                    raise FastBasicICEngineInconsistentInputParametersError(
                        "When use_thrust_rate is True, thrust_rate should be provided."
                    )
                thrust = np.empty_like(thrust_rate)
            else:
                if thrust is None:
                    raise FastBasicICEngineInconsistentInputParametersError(
                        "When use_thrust_rate is False, thrust should be provided."
                    )
                thrust_rate = np.empty_like(thrust)

        else:
            # Check inputs: if use_thrust_rate is not a scalar, both thrust_rate and thrust must be
            # provided and have the same shape as use_thrust_rate
            if thrust_rate is None or thrust is None:
                raise FastBasicICEngineInconsistentInputParametersError(
                    "When use_thrust_rate is a sequence, both thrust_rate and thrust should be "
                    "provided."
                )
            if np.shape(thrust_rate) != np.shape(use_thrust_rate) or np.shape(thrust) != np.shape(
                use_thrust_rate
            ):
                raise FastBasicICEngineInconsistentInputParametersError(
                    "When use_thrust_rate is a sequence, both thrust_rate and thrust should have "
                    "same shape as use_thrust_rate"
                )

        return use_thrust_rate, thrust_rate, thrust

    def sfc_at_max_thrust(
        self, 
        atmosphere: Atmosphere, 
        mach: Union[float, Sequence[float]],
    ) -> np.ndarray:
        """
        Computation of Specific Fuel Consumption at maximum thrust.

        :param atmosphere: Atmosphere instance at intended altitude
        :return: SFC (in kg/s/N)
        """

        altitude = atmosphere.get_altitude(False)
        max_power = self.max_engine_power(altitude)
        # Model of Specific-Fuel-Consumption factor psfc (kg/kW/h)
        if self.fuel_type == 1.0: # Gasoline engine
            if self.strokes_nb == 2.0:
                psfc_max = 1125.9 * max_power ** (-0.2441)
            elif self.strokes_nb == 4.0: 
                psfc_max = -0.0011 * max_power ** 2 + 0.5905 * max_power + 228.58
        elif self.fuel_type == 2.:
            if self.strokes_nb == 2.:  # Diesel engine 2 strokes
                psfc_max = -0.765 * max_power + 334.94
            elif self.strokes_nb == 4.:  # Diesel engine 4 strokes
                psfc_max = -0.964 * max_power + 231.91
        sfc_max = psfc_max*max_power/3600 # change units from kg/h to kg/s
        
        return sfc_max
    
    def sfc_ratio(
        self,
        altitude: Union[float, Sequence[float]],
        thrust_rate: Union[float, Sequence[float]],
        mach: Union[float, Sequence[float]] = 0.8,
    ) -> np.ndarray:
        """
        Computation of ratio :math:`\\frac{SFC(F)}{SFC(Fmax)}`, given 
        thrust_rate :math:`\\frac{F}{Fmax}`.

        :param thrust_rate:
        :return: SFC ratio
        """

        thrust_rate = np.asarray(thrust_rate)

        sfc_ratio = (-0.9976 * thrust_rate ** 2 + 1.9964 * thrust_rate)

        return sfc_ratio
    
    def max_engine_power(
        self, 
        atmosphere: Atmosphere,
    ) -> Union[float, Sequence]:
        """
        Computes maximum power at given atmosphere conditions (altitude, delta_t4)
        
        :param atmosphere: 
        :return: maximum power (in kW)
        """
        
        atmosphere_design = Atmosphere(0.0, altitude_in_feet=False)
        
        sigma = atmosphere.density/atmosphere_design.density
        max_engine_power = self.power_sea_level*(sigma-(1-sigma)/7.55)

        return max_engine_power
    
    def _get_delta_t4(
        self, phase: Union[EngineSetting, Sequence[EngineSetting]]
    ) -> Union[float, Sequence[float]]:
        """
        :param phase:
        :return: DeltaT4 according to engine_setting
        """

        if np.shape(phase) == ():  # engine_setting is a scalar
            return self.dt4_values[phase]

        # Here engine_setting is a sequence. Ensure now it is a numpy array
        phase_array = np.asarray(phase)

        delta_t4 = np.empty(phase_array.shape)
        for phase_value, dt4_value in self.dt4_values.items():
            delta_t4[phase_array == phase_value] = dt4_value

        return delta_t4