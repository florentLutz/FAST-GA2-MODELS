"""
Test module for Overall Aircraft Design process
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

import os
import os.path as pth
from shutil import rmtree

import openmdao.api as om
import pytest
from numpy.testing import assert_allclose

from fastoad.io.configuration.configuration import FASTOADProblemConfigurator

DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
PATH = pth.dirname(__file__).split(os.path.sep)
NOTEBOOKS_PATH = PATH[0] + os.path.sep
for folder in PATH[1:len(PATH) - 3]:
    NOTEBOOKS_PATH = pth.join(NOTEBOOKS_PATH, folder)
NOTEBOOKS_PATH = pth.join(NOTEBOOKS_PATH, "notebooks")
# XML_FILE_NAME = "beechcraft_76.xml"
# XML_FILE_NAME = "cirrus_sr22.xml"
XML_FILE_NAME = "beechcraft_76_wing_pos.xml"
# XML_FILE_NAME = "cirrus_sr22_wing_pos.xml"
PROCESS_FILE_NAME = "oad_process_be76.yml"
# PROCESS_FILE_NAME = "oad_process_sr22.yml"


@pytest.fixture(scope="module")
def cleanup():
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
    rmtree("D:/tmp", ignore_errors=True)


def _test_oad_process_no_wing_positioning(cleanup):
    """
    Test the overall aircraft design process without wing positioning.
    """

    configurator = FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, PROCESS_FILE_NAME))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, XML_FILE_NAME)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()
    problem.run_model()
    problem.write_outputs()

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)
    om.view_connections(
        problem, outfile=pth.join(RESULTS_FOLDER_PATH, "connections.html"), show_browser=False
    )
    om.n2(problem, outfile=pth.join(RESULTS_FOLDER_PATH, "n2.html"), show_browser=False)

    # Check that weight-performances loop correctly converged
    _check_weight_performance_loop(problem)

    if XML_FILE_NAME == "cirrus_sr22.xml":
        # noinspection PyTypeChecker
        assert_allclose(problem.get_val("data:mission:sizing:fuel", units="kg"), 258.831, atol=1)
        # noinspection PyTypeChecker
        assert_allclose(problem["data:handling_qualities:stick_fixed_static_margin"], 0.0728, atol=1e-2)
        # noinspection PyTypeChecker
        assert_allclose(problem.get_val("data:weight:aircraft:MTOW", units="kg"), 1629.7406025, atol=1)
        # noinspection PyTypeChecker
        assert_allclose(problem.get_val("data:weight:aircraft:OWE", units="kg"), 1030.9167, atol=1)
    else:
        # noinspection PyTypeChecker
        assert_allclose(problem.get_val("data:mission:sizing:fuel", units="kg"), 228.624, atol=1)
        # noinspection PyTypeChecker
        assert_allclose(problem["data:handling_qualities:stick_fixed_static_margin"], 0.0252, atol=1e-2)
        # noinspection PyTypeChecker
        assert_allclose(problem.get_val("data:weight:aircraft:MTOW", units="kg"), 1678.863295, atol=1)
        # noinspection PyTypeChecker
        assert_allclose(problem.get_val("data:weight:aircraft:OWE", units="kg"), 1090.2524, atol=1)


def test_oad_process_wing_positioning(cleanup):
    """
    Test the overall aircraft design process with wing positioning.
    """

    configurator = FASTOADProblemConfigurator(pth.join(DATA_FOLDER_PATH, PROCESS_FILE_NAME))

    # Create inputs
    ref_inputs = pth.join(DATA_FOLDER_PATH, XML_FILE_NAME)
    configurator.write_needed_inputs(ref_inputs)

    # Create problems with inputs
    problem = configurator.get_problem(read_inputs=True)
    problem.setup()
    problem.run_model()
    problem.write_outputs()

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)
    om.view_connections(
        problem, outfile=pth.join(RESULTS_FOLDER_PATH, "connections.html"), show_browser=False
    )
    om.n2(problem, outfile=pth.join(RESULTS_FOLDER_PATH, "n2.html"), show_browser=False)

    # Check that weight-performances loop correctly converged
    _check_weight_performance_loop(problem)

    if XML_FILE_NAME == "cirrus_sr22_wing_pos.xml":
        assert_allclose(problem.get_val("data:mission:sizing:fuel", units="kg"), 250.285, atol=1)
        # noinspection PyTypeChecker
        assert_allclose(problem["data:handling_qualities:stick_fixed_static_margin"], 0.1000, atol=1e-2)
        # noinspection PyTypeChecker
        assert_allclose(problem.get_val("data:weight:aircraft:MTOW", units="kg"), 1648.162803, atol=1)
        # noinspection PyTypeChecker
        assert_allclose(problem.get_val("data:weight:aircraft:OWE", units="kg"), 1042.877, atol=1)
    else:
        assert_allclose(problem.get_val("data:mission:sizing:fuel", units="kg"), 255.0, atol=1)
        # noinspection PyTypeChecker
        assert_allclose(problem["data:handling_qualities:stick_fixed_static_margin"], 0.1500, atol=1e-2)
        # noinspection PyTypeChecker
        assert_allclose(problem.get_val("data:weight:aircraft:MTOW", units="kg"), 1777.71776501, atol=1)
        # noinspection PyTypeChecker
        assert_allclose(problem.get_val("data:weight:aircraft:OWE", units="kg"), 1132.541, atol=1)


def _check_weight_performance_loop(problem):
    assert_allclose(
        problem["data:weight:aircraft:OWE"],
        problem["data:weight:airframe:mass"]
        + problem["data:weight:propulsion:mass"]
        + problem["data:weight:systems:mass"]
        + problem["data:weight:furniture:mass"],
        rtol=5e-2,
    )
    assert_allclose(
        problem["data:weight:aircraft:MZFW"],
        problem["data:weight:aircraft:OWE"] + problem["data:weight:aircraft:max_payload"],
        rtol=5e-2,
    )
    assert_allclose(
        problem["data:weight:aircraft:MTOW"],
        problem["data:weight:aircraft:OWE"]
        + problem["data:weight:aircraft:payload"]
        + problem["data:mission:sizing:fuel"],
        rtol=5e-2,
    )
