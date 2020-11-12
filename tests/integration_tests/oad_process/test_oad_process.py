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
from fastoad.io.configuration.configuration import FASTOADProblemConfigurator
from fastoad.io.xml import VariableXmlStandardFormatter
from fastoad.openmdao.utils import get_problem_after_setup
from numpy.testing import assert_allclose


DATA_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")


@pytest.fixture(scope="module")
def cleanup():
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)


def test_oad_process(cleanup):
    """
    Test for the overall aircraft design process.
    """

    problem = FASTOADProblemConfigurator(
        pth.join(DATA_FOLDER_PATH, "oad_process.toml")
    ).get_problem()

    ref_inputs = pth.join(DATA_FOLDER_PATH, "beechcraft_76.xml")
    get_problem_after_setup(problem).write_needed_inputs(ref_inputs, VariableXmlStandardFormatter())

    recorder = om.SqliteRecorder("track_solving_process")
    problem.driver.add_recorder(recorder)
    problem.recording_options["record_inputs"] = True
    problem.recording_options["record_residuals"] = True
    problem.recording_options["record_responses"] = True
    problem.read_inputs()
    print('\n')
    problem.setup(check=True)
    problem.set_solver_print(level=2)
    problem.run_model()
    problem.write_outputs()

    cr = om.CaseReader("track_solving_process")

    if not pth.exists(RESULTS_FOLDER_PATH):
        os.mkdir(RESULTS_FOLDER_PATH)
    om.view_connections(
        problem, outfile=pth.join(RESULTS_FOLDER_PATH, "connections.html"), show_browser=False
    )
    om.n2(problem, outfile=pth.join(RESULTS_FOLDER_PATH, "n2.html"), show_browser=False)

    # Check that weight-performances loop correctly converged
    assert_allclose(
        problem["data:weight:aircraft:OWE"],
        problem["data:weight:airframe:mass"]
        + problem["data:weight:propulsion:mass"]
        + problem["data:weight:systems:mass"]
        + problem["data:weight:furniture:mass"],
        atol=1,
    )
    assert_allclose(
        problem["data:weight:aircraft:MZFW"],
        problem["data:weight:aircraft:OWE"] + problem["data:weight:aircraft:max_payload"],
        atol=1,
    )
    assert_allclose(
        problem["data:weight:aircraft:MTOW"],
        problem["data:weight:aircraft:OWE"]
        + problem["data:weight:aircraft:max_payload"]
        + problem["data:mission:sizing:fuel"],
        atol=1,
    )
