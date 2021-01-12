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
import numpy as np
from shutil import rmtree, copy
import time

from command import api as _api
from fastoad import api
import openmdao.api as om
import pytest
from numpy.testing import assert_allclose

from fastoad.io.configuration.configuration import FASTOADProblemConfigurator
from fastoad.io.xml import VariableXmlStandardFormatter
from fastoad.openmdao.utils import get_problem_after_setup
from fastoad.utils.postprocessing import VariableViewer
from fastoad.utils.postprocessing.analysis_and_plots import wing_geometry_plot, aircraft_geometry_plot, \
    mass_breakdown_sun_plot, mass_breakdown_bar_plot

INPUT_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
PATH = pth.dirname(__file__).split(os.path.sep)
NOTEBOOKS_PATH = PATH[0] + os.path.sep
for folder in PATH[1:len(PATH)-3]:
    NOTEBOOKS_PATH = pth.join(NOTEBOOKS_PATH, folder)
NOTEBOOKS_PATH = pth.join(NOTEBOOKS_PATH, "notebooks")


@pytest.fixture(scope="module")
def cleanup():
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)


def test_oad_process(cleanup):
    """
    Test for the overall aircraft design process.
    """

    problem = FASTOADProblemConfigurator(pth.join(INPUT_FOLDER_PATH, "oad_process.toml")).get_problem()

    ref_inputs = pth.join(INPUT_FOLDER_PATH, "beechcraft_76.xml")
    get_problem_after_setup(problem).write_needed_inputs(ref_inputs, VariableXmlStandardFormatter())
    problem.read_inputs()
    print('\n')
    problem.setup(check=True)
    problem.set_solver_print(level=2)
    with Timer(name="Mass-performance loop:"):
        problem.run_model()
    problem.write_outputs()

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
        + problem["data:weight:aircraft:max_payload"]
        + problem["data:mission:sizing:fuel"],
        rtol=5e-2,
    )

    assert_allclose(problem["data:handling_qualities:static_margin"], 0.12, atol=1e-2)
    # noinspection PyTypeChecker
    assert_allclose(problem.get_val("data:weight:aircraft:MTOW", units="kg"), 1503, atol=1)
    # noinspection PyTypeChecker
    assert_allclose(problem.get_val("data:mission:sizing:fuel", units="kg"), 185, atol=1)


def est_api(cleanup):

    # Generation of inputs ----------------------------------------------------
    # We get the same inputs as in tutorial notebook
    DATA_FOLDER_PATH = pth.join(NOTEBOOKS_PATH, "tutorial", "data")
    WORK_FOLDER_PATH = pth.join(NOTEBOOKS_PATH, "tutorial", "workdir")
    configuration_file = pth.join(WORK_FOLDER_PATH, "oad_process.toml")
    SOURCE_FILE = pth.join(DATA_FOLDER_PATH, "beechcraft_76.xml")
    _api.generate_configuration_file(configuration_file, overwrite=True)


    # Run 800NM model ---------------------------------------------------------
    api.generate_inputs(configuration_file, SOURCE_FILE, overwrite=True)
    api.evaluate_problem(configuration_file, True)

    OUTPUT_FILE = pth.join(WORK_FOLDER_PATH, 'problem_outputs.xml')
    Beechcraft_800nm_OUTPUT_FILE = pth.join(WORK_FOLDER_PATH, 'problem_outputs_Beechcraft_800nm_mda.xml')
    copy(OUTPUT_FILE, Beechcraft_800nm_OUTPUT_FILE)

    # Run 1000NM model --------------------------------------------------------
    viewer = VariableViewer()
    viewer.load(SOURCE_FILE)
    index = np.where(viewer.dataframe['Name'] == 'data:TLAR:range')[0][0]
    viewer.dataframe.at[index, 'Value'] = 1000.0
    viewer.save(SOURCE_FILE)

    api.generate_inputs(configuration_file, SOURCE_FILE, overwrite=True)
    api.evaluate_problem(configuration_file, True)

    OUTPUT_FILE = pth.join(WORK_FOLDER_PATH, 'problem_outputs.xml')
    Beechcraft_1000nm_OUTPUT_FILE = pth.join(WORK_FOLDER_PATH, 'problem_outputs_Beechcraft_1000nm_mda.xml')
    copy(OUTPUT_FILE, Beechcraft_1000nm_OUTPUT_FILE)

    # Try geometry plot -------------------------------------------------------
    fig = wing_geometry_plot(Beechcraft_800nm_OUTPUT_FILE, name='Beechcraft 800 nm MDA')
    fig = wing_geometry_plot(Beechcraft_1000nm_OUTPUT_FILE, name='Beechcraft 1000 nm MDA', fig=fig)
    fig.show()
    fig = aircraft_geometry_plot(Beechcraft_800nm_OUTPUT_FILE, name='Beechcraft 800 nm MDA')
    fig = aircraft_geometry_plot(Beechcraft_1000nm_OUTPUT_FILE, name='Beechcraft 1000 nm MDA', fig=fig)
    fig.show()

    # Try mass breakdown ------------------------------------------------------
    fig = mass_breakdown_sun_plot(Beechcraft_800nm_OUTPUT_FILE)
    fig.show()
    fig = mass_breakdown_bar_plot(Beechcraft_800nm_OUTPUT_FILE, name='Beechcraft 800 nm MDA')
    fig = mass_breakdown_bar_plot(Beechcraft_1000nm_OUTPUT_FILE, name='Beechcraft 1000 nm MDA', fig=fig)
    fig.show()

    # Run optimization --------------------------------------------------------
    viewer = VariableViewer()
    viewer.load(SOURCE_FILE)
    index = np.where(viewer.dataframe['Name'] == 'data:TLAR:range')[0][0]
    viewer.dataframe.at[index, 'Value'] = 800.0
    viewer.save(SOURCE_FILE)

    api.generate_inputs(configuration_file, SOURCE_FILE, overwrite=True)
    problem = api.optimize_problem(configuration_file, True)
    assert not problem.optim_failed

    # Check that weight-performances loop correctly converged
    assert_allclose(
        problem["data:weight:aircraft:OWE"],
        problem["data:weight:airframe:mass"]
        + problem["data:weight:propulsion:mass"]
        + problem["data:weight:systems:mass"]
        + problem["data:weight:furniture:mass"]
        + problem["data:weight:crew:mass"],
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
        + problem["data:weight:aircraft:payload"]
        + problem["data:mission:sizing:fuel"],
        atol=1,
    )

    # Design Variable
    assert_allclose(problem["data:geometry:wing:MAC:at25percent:x"], 17.06, atol=1e-1)

    # Constraint
    assert_allclose(problem["data:handling_qualities:static_margin"], 0.05, atol=1e-2)

    # Objective
    assert_allclose(problem["data:mission:sizing:fuel"], 20565, atol=50)

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print('\n')
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))
