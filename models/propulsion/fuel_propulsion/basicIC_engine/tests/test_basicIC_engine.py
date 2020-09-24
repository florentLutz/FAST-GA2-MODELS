"""
Test module for basicIC_engine.py
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
import pandas as pd
import pytest
from fastoad.base.flight_point import FlightPoint
from fastoad.constants import EngineSetting
from fastoad.utils.physics import Atmosphere


from ..basicIC_engine import BasicICEngine


def test_compute_flight_points():
    engine = BasicICEngine(30000.0, 1.0, 4.0)  # load a 4-strokes 30kW gasoline engine

    # Test with scalars
    flight_point = FlightPoint(
        mach=0, altitude=0, engine_setting=EngineSetting.TAKEOFF, thrust_rate=0.8
    )  # with engine_setting as EngineSetting
    engine.compute_flight_points(flight_point)
    np.testing.assert_allclose(flight_point.thrust, 24000 * 0.8, rtol=1e-2)
    np.testing.assert_allclose(flight_point.sfc, 6.99160433e-08, rtol=1e-2)

    flight_point = FlightPoint(
        mach=0.3, altitude=0, engine_setting=EngineSetting.CLIMB.value, thrust=117.545
    )  # with engine_setting as int
    engine.compute_flight_points(flight_point)
    np.testing.assert_allclose(flight_point.thrust_rate, 0.5, rtol=1e-2)
    np.testing.assert_allclose(flight_point.sfc, 6.51113702e-06, rtol=1e-2)

    # Test full arrays
    # 2D arrays are used, where first line is for thrust rates, and second line
    # is for thrust values.
    # As thrust rates and thrust values match, thrust rate results are 2 equal
    # lines and so are thrust value results.
    machs = [0, 0.3, 0.3, 0.8, 0.8]
    altitudes = [0, 0, 0, 10000, 13000]
    thrust_rates = [0.8, 0.5, 0.5, 0.4, 0.7]
    thrusts = [1.92000000e+04, 1.17545000e+02, 1.17545000e+02, 2.57198313e+01, 4.05992947e+01]
    engine_settings = [
        EngineSetting.TAKEOFF,
        EngineSetting.TAKEOFF,
        EngineSetting.CLIMB,
        EngineSetting.IDLE,
        EngineSetting.CRUISE.value,
    ]  # mix EngineSetting with integers
    expected_sfc = [6.99160433e-08, 6.51113702e-06, 6.51113702e-06, 1.40206418e-05, 1.96148075e-05]

    flight_points = pd.DataFrame()
    flight_points["mach"] = machs + machs
    flight_points["altitude"] = altitudes + altitudes
    flight_points["engine_setting"] = engine_settings + engine_settings
    flight_points["thrust_is_regulated"] = [False] * 5 + [True] * 5
    flight_points["thrust_rate"] = thrust_rates + [0.0] * 5
    flight_points["thrust"] = [0.0] * 5 + thrusts
    engine.compute_flight_points(flight_points)
    np.testing.assert_allclose(flight_points.sfc, expected_sfc + expected_sfc, rtol=1e-4)
    np.testing.assert_allclose(flight_points.thrust_rate, thrust_rates + thrust_rates, rtol=1e-4)
    np.testing.assert_allclose(flight_points.thrust, thrusts + thrusts, rtol=1e-4)


def test_engine_weight():
    name1 = BasicICEngine(21000.0, 1.0, 4.0)
    np.testing.assert_allclose(name1.engine_weight(), 12, atol=1)
    name2 = BasicICEngine(75000.0, 1.0, 4.0)
    np.testing.assert_allclose(name2.engine_weight(), 146, atol=1)


def test_engine_dim():
    name1 = BasicICEngine(21000.0, 1.0, 4.0)
    np.testing.assert_allclose(name1.engine_dim(), [1.23, 0.89, 0.67], atol=1e-2)
    name2 = BasicICEngine(75000.0, 1.0, 4.0)
    np.testing.assert_allclose(name2.engine_dim(), [1.88, 1.37, 1.03], atol=1e-2)


def test_sfc_at_max_thrust():
    """
    Checks model against values from :...

    .. bibliography:: ../refs.bib
    """

    # Check with arrays
    name1 = BasicICEngine(21000.0, 1.0, 4.0)
    atm = Atmosphere([0, 10668, 13000], altitude_in_feet=False)
    sfc = name1.sfc_at_max_power(atm)
    # Note: value for alt==10668 is different from PhD report
    #       alt=13000 is here just for testing in stratosphere
    np.testing.assert_allclose(sfc, [6.68042778e-08, 6.57954095e-08, 6.56038524e-08], rtol=1e-4)

    # Check with scalars
    name2 = BasicICEngine(75000.0, 1.0, 4.0)
    atm = Atmosphere(0, altitude_in_feet=False)
    sfc = name2.sfc_at_max_power(atm)
    np.testing.assert_allclose(sfc, 7.407777777777777e-08, rtol=1e-4)


def test_sfc_ratio():
    """    Checks SFC ratio model    """
    engine = BasicICEngine(75000.0, 1.0, 4.0)

    # Test different altitude with constant thrust rate/power ratio
    altitudes = np.array([-2370, -1564, -1562.5, -1560, -846, 678, 2202, 3726])
    ratio, _ = engine.sfc_ratio(altitudes, 0.8)
    assert ratio == pytest.approx(
        [0.958656, 0.958656, 0.958656, 0.958656, 0.958656, 0.958656, 0.958656, 0.958656], rel=1e-3
    )

    # Because there some code differs when we have scalars:
    ratio, _ = engine.sfc_ratio(1562.5, 0.6)
    assert ratio == pytest.approx(0.839, rel=1e-3)
