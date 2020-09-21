"""
Test module for OpenMDAO versions of basicICEngine
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
import openmdao.api as om
from fastoad.constants import EngineSetting

from tests.testing_utilities import run_system
from ..openmdao import OMBasicICEngineComponent


def test_OMBasicICEngineComponent():
    """ Tests ManualBasicICEngine component """
    # Same test as in test_basicIC_engine.test_compute_flight_points
    engine = OMBasicICEngineComponent(flight_point_count=(2, 5))

    machs = [0, 0.3, 0.3, 0.8, 0.8]
    altitudes = [0, 0, 0, 10000, 13000]
    thrust_rates = [0.8, 0.5, 0.5, 0.4, 0.7]
    thrusts = [1.92000000e+04, 1.17545000e+02, 1.17545000e+02, 2.57198313e+01, 4.05992947e+01]
    phases = [
        EngineSetting.TAKEOFF,
        EngineSetting.TAKEOFF,
        EngineSetting.CLIMB,
        EngineSetting.IDLE,
        EngineSetting.CRUISE.value,
    ]  # mix EngineSetting with integers
    expected_sfc = [6.99160433e-08, 6.51113702e-06, 6.51113702e-06, 1.40206418e-05, 1.96148075e-05]

    ivc = om.IndepVarComp()
    ivc.add_output("data:propulsion:IC_engine:max_power", 30000, units="W")
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4)

    ivc.add_output("data:propulsion:mach", [machs, machs])
    ivc.add_output("data:propulsion:altitude", [altitudes, altitudes], units="m")
    ivc.add_output("data:propulsion:engine_setting", [phases, phases])
    ivc.add_output("data:propulsion:use_thrust_rate", [[True] * 5, [False] * 5])
    ivc.add_output("data:propulsion:required_thrust_rate", [thrust_rates, [0] * 5])
    ivc.add_output("data:propulsion:required_thrust", [[0] * 5, thrusts], units="N")

    problem = run_system(engine, ivc)

    np.testing.assert_allclose(
        problem["data:propulsion:SFC"], [expected_sfc, expected_sfc], rtol=1e-2
    )
    np.testing.assert_allclose(
        problem["data:propulsion:thrust_rate"], [thrust_rates, thrust_rates], rtol=1e-2
    )
    np.testing.assert_allclose(problem["data:propulsion:thrust"], [thrusts, thrusts], rtol=1e-2)
