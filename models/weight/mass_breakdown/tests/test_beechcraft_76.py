"""
Test module for mass breakdown functions
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

# pylint: disable=redefined-outer-name  # needed for pytest fixtures
import os.path as pth

import openmdao.api as om
import pytest
from fastoad.io import VariableIO

from ....tests.testing_utilities import run_system
from ..a_airframe import (
    ComputeTailWeight,
    ComputeFlightControlsWeight,
    ComputeFuselageWeight,
    ComputeWingWeight,
    ComputeLandingGearWeight,
)
from ..b_propulsion import (
    ComputeFuelLinesWeight,
    ComputeEngineWeight,
)
from ..c_systems import (
    ComputeLifeSupportSystemsWeight,
    ComputeNavigationSystemsWeight,
    ComputePowerSystemsWeight,
)
from ..d_furniture import (
    ComputePassengerSeatsWeight,
)

from ..mass_breakdown import MassBreakdown, ComputeOperatingWeightEmpty
from ..payload import ComputePayload

XML_FILE = "beechcraft_76.xml"


def get_indep_var_comp(var_names):
    """ Reads required input data and returns an IndepVarcomp() instance"""
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    ivc = reader.read(only=var_names).to_ivc()
    return ivc


def list_inputs(group):
    """ Reads input variables from a group and return as a list (run model with 0 value can lead to raise configuration
    errors in models)"""
    prob = om.Problem(model=group)
    prob.setup()
    prob.run_model()
    data = prob.model.list_inputs(out_stream=None)
    list_names = []
    for idx in range(len(data)):
        variable_name = data[idx][0].split('.')
        list_names.append(variable_name[len(variable_name) - 1])
    return list(dict.fromkeys(list_names))


def test_compute_payload():

    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output("data:TLAR:NPAX", val=2.0)
    problem = run_system(ComputePayload(), ivc)
    assert problem["data:weight:aircraft:payload"] == pytest.approx(320.0, abs=1e-2)
    assert problem["data:weight:aircraft:max_payload"] == pytest.approx(360.0, abs=1e-2)

    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output("data:TLAR:NPAX", val=10.0)
    ivc.add_output("settings:weight:aircraft:payload:design_mass_per_passenger", 1.0, units="kg")
    ivc.add_output("settings:weight:aircraft:payload:max_mass_per_passenger", 2.0, units="kg")
    problem = run_system(ComputePayload(), ivc)
    assert problem["data:weight:aircraft:payload"] == pytest.approx(12.0, abs=0.1)
    assert problem["data:weight:aircraft:max_payload"] == pytest.approx(24.0, abs=0.1)


def test_compute_wing_weight():
    """ Tests wing weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeWingWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingWeight(), ivc)
    weight_a1 = problem.get_val("data:weight:airframe:wing:mass", units="kg")
    assert weight_a1 == pytest.approx(218.74, abs=1e-2)  # difference because of integer conversion error


def test_compute_fuselage_weight():
    """ Tests fuselage weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeFuselageWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageWeight(), ivc)
    weight_a2 = problem.get_val("data:weight:airframe:fuselage:mass", units="kg")
    assert weight_a2 == pytest.approx(153.90, abs=1e-2)


def test_compute_empennage_weight():
    """ Tests empennage weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeTailWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeTailWeight(), ivc)
    weight_a31 = problem.get_val("data:weight:airframe:horizontal_tail:mass", units="kg")
    assert weight_a31 == pytest.approx(32.24, abs=1e-2)
    weight_a32 = problem.get_val("data:weight:airframe:vertical_tail:mass", units="kg")
    assert weight_a32 == pytest.approx(0.0, abs=1e-2)


def test_compute_flight_controls_weight():
    """ Tests flight controls weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeFlightControlsWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFlightControlsWeight(), ivc)
    weight_a4 = problem.get_val("data:weight:airframe:flight_controls:mass", units="kg")
    assert weight_a4 == pytest.approx(89.95, abs=1e-2)


def test_compute_landing_gear_weight():
    """ Tests landing gear weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeLandingGearWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLandingGearWeight(), ivc)
    weight_a51 = problem.get_val("data:weight:airframe:landing_gear:main:mass", units="kg")
    assert weight_a51 == pytest.approx(18.73, abs=1e-2)
    weight_a52 = problem.get_val("data:weight:airframe:landing_gear:front:mass", units="kg")
    assert weight_a52 == pytest.approx(9.36, abs=1e-2)


def test_compute_engine_weight():
    """ Tests engine weight computation from sample XML data """

    # Input list from model (not generated because of the assertion error on bad motor configuration)
    input_list = [
        "data:geometry:propulsion:engine:count",
        "data:propulsion:IC_engine:max_power",
        "data:propulsion:IC_engine:fuel_type",
        "data:propulsion:IC_engine:n_strokes",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeEngineWeight(), ivc)
    weight_b1 = problem.get_val("data:weight:propulsion:engine:mass", units="kg")
    assert weight_b1 == pytest.approx(255.41, abs=1e-2)


def test_compute_fuel_lines_weight():
    """ Tests fuel lines weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeFuelLinesWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuelLinesWeight(), ivc)
    weight_b2 = problem.get_val("data:weight:propulsion:fuel_lines:mass", units="kg")
    assert weight_b2 == pytest.approx(58.23, abs=1e-2)


def test_compute_navigation_systems_weight():
    """ Tests navigation systems weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeNavigationSystemsWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeNavigationSystemsWeight(), ivc)
    weight_c3 = problem.get_val("data:weight:systems:navigation:mass", units="kg")
    assert weight_c3 == pytest.approx(33.46, abs=1e-2)


def test_compute_power_systems_weight():
    """ Tests power systems weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputePowerSystemsWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:weight:systems:navigation:mass", 33.46, units="kg")
    ivc.add_output("data:weight:propulsion:fuel_lines:mass", 32.95, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePowerSystemsWeight(), ivc)
    weight_c12 = problem.get_val("data:weight:systems:power:electric_systems:mass", units="kg")
    assert weight_c12 == pytest.approx(72.53, abs=1e-2)
    weight_c13 = problem.get_val("data:weight:systems:power:hydraulic_systems:mass", units="kg")
    assert weight_c13 == pytest.approx(13.41, abs=1e-2)


def test_compute_life_support_systems_weight():
    """ Tests life support systems weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeLifeSupportSystemsWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:weight:systems:navigation:mass", 33.46, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLifeSupportSystemsWeight(), ivc)
    weight_c22 = problem.get_val("data:weight:systems:life_support:air_conditioning:mass", units="kg")
    assert weight_c22 == pytest.approx(43.66, abs=1e-2)


def test_compute_passenger_seats_weight():
    """ Tests passenger seats weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputePassengerSeatsWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePassengerSeatsWeight(), ivc)
    weight_d2 = problem.get_val("data:weight:furniture:passenger_seats:mass", units="kg")
    assert weight_d2 == pytest.approx(86.18, abs=1e-2)  # additional 2 pilots seats (differs from old version)


def test_evaluate_owe():
    """ Tests a simple evaluation of Operating Weight Empty from sample XML data. """

    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()

    mass_computation = run_system(ComputeOperatingWeightEmpty(), input_vars, setup_mode="fwd")

    oew = mass_computation.get_val("data:weight:aircraft:OWE", units="kg")
    assert oew == pytest.approx(1098.77, abs=1e-2)


def test_loop_compute_owe():
    """ Tests a weight computation loop matching the max payload criterion. """

    # with payload computed from NPAX
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read(
        ignore=[
            "data:weight:aircraft:max_payload",
            "data:weight:aircraft:MTOW",
        ]
    ).to_ivc()

    mass_computation_1 = run_system(MassBreakdown(payload_from_npax=True), input_vars)
    oew = mass_computation_1.get_val("data:weight:aircraft:OWE", units="kg")
    assert oew == pytest.approx(1026.50, abs=1e-2)

    # with payload as input
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read(
        ignore=[
            "data:weight:aircraft:MTOW",
        ]
    ).to_ivc()
    mass_computation_2 = run_system(MassBreakdown(payload_from_npax=False), input_vars)
    oew = mass_computation_2.get_val("data:weight:aircraft:OWE", units="kg")
    assert oew == pytest.approx(1009.19, abs=1e-2)
