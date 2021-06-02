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
import numpy as np

from models.aerodynamics.external.xfoil import resources
from models.aerodynamics.external.vlm.compute_aero import DEFAULT_WING_AIRFOIL, DEFAULT_HTP_AIRFOIL
from command import api as api_cs23
from models.aerodynamics.external.xfoil import XfoilGroup as Xfoil
from models.aerodynamics.aerodynamics_high_speed import AerodynamicsHighSpeed

INPUT_FOLDER_PATH = pth.join(pth.dirname(__file__), "data")
RESULTS_FOLDER_PATH = pth.join(pth.dirname(__file__), "results")
PATH = pth.dirname(__file__).split(os.path.sep)
NOTEBOOKS_PATH = PATH[0] + os.path.sep
for folder in PATH[1:len(PATH) - 3]:
    NOTEBOOKS_PATH = pth.join(NOTEBOOKS_PATH, folder)
NOTEBOOKS_PATH = pth.join(NOTEBOOKS_PATH, "notebooks")
# XML_NAME = "cirrus_sr22.xml"
XML_NAME = "blank.xml"


@pytest.fixture(scope="module")
def cleanup():
    rmtree(RESULTS_FOLDER_PATH, ignore_errors=True)
    rmtree("D:/tmp", ignore_errors=True)


def test_block_analysis_xfoil(cleanup):
    """
    Test the overall aircraft design process without and with optimization.
    """
    airfoil = "naca23012.af"

    clear_polar_results()

    # Define the wing primary geometry parameters name as a list
    var_inputs = ["data:TLAR:v_approach", "data:Xfoil_pre_processing:reynolds"]

    # Declare function
    compute_xfoil = api_cs23.generate_block_analysis(
        Xfoil(
            airfoil_file="naca23012.af"
        ),
        var_inputs,
        pth.join(INPUT_FOLDER_PATH, XML_NAME),
        True,
    )

    inputs_dict = {"data:TLAR:v_approach": (79.0, "kn"),
                   "data:Xfoil_pre_processing:reynolds": (4000000, None)}

    outputs_dict = compute_xfoil(inputs_dict)

    alpha_result = np.array([0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.,
                             5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5,
                             11., 11.5, 12., 12.5, 13., 13.5, 14., 14.5, 15., 15.5, 16.,
                             16.5, 17., 17.5, 18., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                             0., 0., 0., 0., 0., 0., 0.])
    alpha_vector = outputs_dict["pre_xfoil_polar.xfoil:alpha"][0]

    assert np.max(abs(alpha_result - alpha_vector)) < 1e-2

def clear_polar_results():
    # Clear saved polar results (for wing and htp airfoils)
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('af', 'csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('af', 'csv')))
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('.af', '_sym.csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('.af', '_sym.csv')))
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_HTP_AIRFOIL.replace('af', 'csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_HTP_AIRFOIL.replace('af', 'csv')))