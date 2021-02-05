"""
Simple implementation of International Standard Atmosphere.
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

from fastoad.utils.physics import Atmosphere as Atmos
import numpy as np


class Atmosphere(Atmos):
    """Same as :class:`Atmosphere` except that get_true_airspeed and get_equivalent_airspeed have been corrected to
    integrate the compressibility effect.

    """

    def get_true_airspeed(self, equivalent_airspeed):
        """
        Computes true airspeed (TAS) from equivalent airspeed (EAS).

        :param equivalent_airspeed: in m/s
        :return: true airspeed in m/s
        """
        sea_level = Atmosphere(0)
        current_level = Atmosphere(self._altitude, altitude_in_feet=False)
        gamma = 1.4

        impact_pressure = (
                sea_level.pressure
                * (((np.asarray(equivalent_airspeed) / sea_level.speed_of_sound) ** 2.0 / 5.0 + 1.0) ** 3.5 - 1.0)
        )

        total_pressure = current_level.pressure + impact_pressure
        sigma_0 = total_pressure / current_level.pressure

        mach = (2. / (gamma - 1.0) * (sigma_0 ** ((gamma - 1.0) / gamma) - 1.0)) ** 0.5

        true_airspeed = mach * current_level.speed_of_sound

        return self._return_value(true_airspeed)

    def get_equivalent_airspeed(self, true_airspeed):
        """
        Computes equivalent airspeed (EAS) from true airspeed (TAS).

        :param true_airspeed: in m/s
        :return: equivalent airspeed in m/s
        """
        sea_level = Atmosphere(0)
        current_level = Atmosphere(self._altitude, altitude_in_feet=False)
        gamma = 1.4

        mach = np.asarray(true_airspeed) / current_level.speed_of_sound

        sigma_0 = (1.0 + (gamma - 1.0) / 2.0 * mach ** 2.0) ** (gamma / (gamma - 1.0))
        total_pressure = sigma_0 * current_level.pressure
        impact_pressure = total_pressure - current_level.pressure

        equivalent_airspeed = (
                sea_level.speed_of_sound
                * (5.0 * ((impact_pressure / sea_level.pressure + 1.0) ** (1.0 / 3.5) - 1.0)) ** 0.5
        )

        return self._return_value(equivalent_airspeed)
