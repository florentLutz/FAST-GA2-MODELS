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
from ..basicIC_engine.exceptions import FastBasicICEngineInconsistentInputParametersError
from fastoad.utils.physics import Atmosphere

# Logger for this module
_LOGGER = logging.getLogger(__name__)



class BasicICEngine():   

    def __init__(
        self,  
        volume: float, 
        fuel_type: float, 
        strokes_nb: float,
    ):
        """
        Parametric Internal Combustion engine.

        It computes engine characteristics using fuel type and motor architecture (volume, strokes)

        :param volume: (unit=cm**3)
        :param fuel_type: 1.0 for gasoline and 2.0 for gasoil engine
        :param strokes_nb: can be either 2-strockes (=2.0) or 4-strockes (=4.0)
        :param max_altitude: (unit=m) maximum operating altitude
        """
        
        if (fuel_type!=1.0) and (fuel_type!=2.0):
            raise FastBasicICEngineInconsistentInputParametersError(
                "Bad engine configuration: fuel type {0:d} model does not exist.".format(fuel_type)
            )
        
        if (strokes_nb!=4.0):
            raise FastBasicICEngineInconsistentInputParametersError(
                "Bad engine configuration: {0:d}-strokes model does not exist.".format(strokes_nb)
            )
        
        if (strokes_nb == 4.0) and (fuel_type == 1.0):
            self = gasoline_4S()
            self.volume = fuel_type
            self.fuel_type = fuel_type
            self.strokes_nb = strokes_nb
        
        
    def compute_max_torque_curve(self):
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
        
class gasoline_4S()
    """
    Engine performance @ sea level
    """
    def __init__(self):
        fuel_density = 0.75 #kg/L
        fuel_PCI = 13.139 # Wh/kg
        self.N = [1000, 1225, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250]
        self.PME_max = [10, 10, 11, 14, 14.7, 14.7, 15, 15, 15, 15, 15, 15, 15.8, 15.8, 15, 14.7, 14, 13, 13, 12, 11.4, 11.4]
        self.pme = [0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11.4, 12, 13, 14, 14.7, 15, 15.8]
        engine.conso_spe = [
        [2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000,  2000],
        [1088,  1071,  1034,  1051,  1071,  1088,  1098,  1117,  1142,  1183,  1200,  1136,  1221,  1235,  1283,  1288,  1340,  1379,  1418,  1488,  1491,  1721],
        [630,   632,   629,   633,   635,   645,   653,   657,   679,   688,   702,   687,   713,   730,   746,   761,   789,   824,   840,   875,   885,   1010],
        [418,   415,   420,   418,   420,   424,   430,   433,   441,   448,   455,   456,   463,   471,   480,   488,   502,   518,   531,   552,   565,   631],
        [349,   346,   347,   349,   352,   354,   359,   359,   366,   370,   375,   378,   382,   387,   394,   403,   413,   424,   433,   442,   458,   523],
        [314,   312,   313,   315,   316,   319,   321,   323,   327,   331,   337,   340,   340,   346,   352,   359,   368,   376,   384,   393,   402,   464],
        [295,   292,   292,   294,   296,   298,   299,   301,   306,   309,   314,   318,   318,   323,   328,   334,   339,   347,   353,   356,   365,   422],
        [280,   279,   279,   280,   281,   283,   286,   287,   291,   294,   298,   301,   304,   306,   311,   314,   319,   325,   329,   334,   341,   392],
        [271,   269,   268,   269,   271,   274,   275,   277,   280,   283,   288,   290,   291,   293,   296,   299,   302,   308,   312,   315,   329,   368],
        [268,   263,   262,   262,   263,   265,   267,   269,   271,   275,   277,   280,   280,   282,   284,   287,   291,   296,   302,   328,   343,   359],
        [266,   261,   259,   258,   258,   260,   262,   263,   264,   266,   269,   270,   272,   273,   277,   280,   283,   288,   299,   326,   340,   359],
        [265,   260,   257,   256,   254,   255,   257,   258,   259,   261,   264,   265,   266,   268,   272,   275,   279,   294,   305,   329,   350,   382],
        [9999,  303,   260,   259,   252,   255,   256,   256,   256,   259,   262,   263,   264,   266,   270,   273,   291,   298,   315,   334,   356,   437],
        [9999,  9999,  261,   260,   252,   255,   257,   256,   256,   258,   262,   263,   265,   266,   273,   281,   293,   302,   318,   344,   375,   461],
        [9999,  9999,  263,   261,   253,   256,   258,   256,   255,   257,   261,   264,   266,   266,   277,   291,   296,   308,   323,   356,   398,   9999],
        [9999,  9999,  266,   262,   259,   260,   262,   263,   255,   258,   263,   268,   266,   275,   286,   294,   306,   324,   354,   368,   9999,  9999],
        [9999,  9999,  287,   281,   283,   271,   270,   273,   261,   266,   271,   277,   284,   296,   297,   309,   325,   351,   9999,  9999,  9999,  9999],
        [9999,  9999,  9999,  312,   303,   281,   276,   277,   265,   278,   285,   281,   291,   297,   311,   321,   343,   9999,  9999,  9999,  9999,  9999],
        [9999,  9999,  9999,  9999,  9999,  288,   281,   282,   267,   285,   295,   283,   297,   298,   321,   329,   9999,  9999,  9999,  9999,  9999,  9999],
        [9999,  9999,  9999,  9999,  9999,  9999,  9999,  9999,  9999,  9999,  9999,  297,   313,   321,   325,   9999,  9999,  9999,  9999,  9999,  9999,  9999]
        ]
    