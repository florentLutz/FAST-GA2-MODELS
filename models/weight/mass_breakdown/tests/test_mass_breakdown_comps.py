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

from tests.testing_utilities import run_system
from ..a_airframe import (
    EmpennageWeight,
    FlightControlsWeight,
    FuselageWeight,
    WingWeight,
    LandingGearWeight,
)
from ..b_propulsion import (
    FuelLinesWeight,
    EngineWeight,
)
from ..c_systems import (
    LifeSupportSystemsWeight,
    NavigationSystemsWeight,
    PowerSystemsWeight,
)
from ..d_furniture import (
    PassengerSeatsWeight,
)

from ..mass_breakdown import MassBreakdown, OperatingWeightEmpty
from ..payload import ComputePayload


def get_indep_var_comp(var_names):
    """ Reads required input data and returns an IndepVarcomp() instance"""
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", "mass_breakdown_inputs.xml"))
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
    ivc.add_output("data:TLAR:NPAX", val=5.0)
    problem = run_system(ComputePayload(), ivc)
    assert problem["data:weight:aircraft:payload"] == pytest.approx(453.6, abs=0.1)
    assert problem["data:weight:aircraft:max_payload"] == pytest.approx(653.6, abs=0.1)

    ivc = om.IndepVarComp()

    # Run problem and check obtained value(s) is/(are) correct
    ivc.add_output("data:TLAR:NPAX", val=10.0)
    ivc.add_output("settings:weight:aircraft:payload:design_mass_per_passenger", val=1.0, units="kg")
    ivc.add_output("settings:weight:aircraft:payload:max_mass_per_passenger", val=2.0, units="kg")
    problem = run_system(ComputePayload(), ivc)
    assert problem["data:weight:aircraft:payload"] == pytest.approx(10.0, abs=0.1)
    assert problem["data:weight:aircraft:max_payload"] == pytest.approx(20.0, abs=0.1)

def test_compute_wing_weight():
    """ Tests wing weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model",WingWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(WingWeight(), ivc)
    val = problem["data:weight:airframe:wing:mass"]
    assert val == pytest.approx(4931, abs=1)


def test_compute_fuselage_weight():
    """ Tests fuselage weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", FuselageWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FuselageWeight(), ivc)
    val = problem["data:weight:airframe:fuselage:mass"]
    assert val == pytest.approx(4110, abs=1)


def test_compute_empenage_weight():
    """ Tests empennage weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", EmpennageWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(EmpennageWeight(), ivc)
    val1 = problem["data:weight:airframe:horizontal_tail:mass"]
    val2 = problem["data:weight:airframe:vertical_tail:mass"]
    assert val1 == pytest.approx(121, abs=1)
    assert val2 == pytest.approx(0, abs=1)


def test_compute_flight_controls_weight():
    """ Tests flight controls weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", FlightControlsWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FlightControlsWeight(), ivc)
    val = problem["data:weight:airframe:flight_controls:mass"]
    assert val == pytest.approx(899, abs=1)


def test_compute_landing_gear_weight():
    """ Tests landing gear weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", LandingGearWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(LandingGearWeight(), ivc)
    val1 = problem["data:weight:airframe:landing_gear:main:mass"]
    val2 = problem["data:weight:airframe:landing_gear:front:mass"]
    assert val1 == pytest.approx(72, abs=1)
    assert val2 == pytest.approx(36, abs=1)


def test_compute_engine_weight():
    """ Tests engine weight computation from sample XML data """

    # Input list from model (not generated because of the assertion error on bad motor configuration)
    input_list = [
        "data:propulsion:engine:power_SL",
        "data:geometry:propulsion:engine:count",
        "data:propulsion:engine:fuel_type",
        "data:propulsion:engine:n_strokes",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(EngineWeight(), ivc)
    val = problem["data:weight:propulsion:engine:mass"]
    assert val == pytest.approx(2242, abs=1)


def test_compute_fuel_lines_weight():
    """ Tests fuel lines weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", FuelLinesWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(FuelLinesWeight(), ivc)
    val = problem["data:weight:propulsion:fuel_lines:mass"]
    assert val == pytest.approx(73, abs=1)


def test_compute_navigation_systems_weight():
    """ Tests navigation systems weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", NavigationSystemsWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    ivc = get_indep_var_comp(input_list)
    problem = run_system(NavigationSystemsWeight(), ivc)
    val = problem["data:weight:systems:navigation:mass"]
    assert val == pytest.approx(624, abs=1)


def test_compute_power_systems_weight():
    """ Tests power systems weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", PowerSystemsWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:weight:systems:navigation:mass", 624)
    ivc.add_output("data:weight:propulsion:fuel_lines:mass", 86, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PowerSystemsWeight(), ivc)
    val1 = problem["data:weight:systems:power:electric_systems:mass"]
    val2 = problem["data:weight:systems:power:hydraulic_systems:mass"]
    assert val1 == pytest.approx(243, abs=1)
    assert val2 == pytest.approx(530, abs=1)


def test_compute_life_support_systems_weight():
    """ Tests life support systems weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", LifeSupportSystemsWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:weight:systems:navigation:mass", 624, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(LifeSupportSystemsWeight(), ivc)
    val = problem["data:weight:systems:life_support:air_conditioning:mass"]
    assert val == pytest.approx(966, abs=1)


def test_compute_passenger_seats_weight():
    """ Tests passenger seats weight computation from sample XML data """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", PassengerSeatsWeight(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(PassengerSeatsWeight(), ivc)
    val = problem["data:weight:furniture:passenger_seats:mass"]
    assert val == pytest.approx(934, abs=1)


def test_evaluate_oew():
    """ Tests a simple evaluation of Operating Empty Weight from sample XML data. """

    reader = VariableIO(pth.join(pth.dirname(__file__), "data", "mass_breakdown_inputs.xml"))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()

    mass_computation = run_system(OperatingWeightEmpty(), input_vars, setup_mode="fwd")

    oew = mass_computation["data:weight:aircraft:OWE"]
    assert oew == pytest.approx(14975, abs=1)


def test_loop_compute_oew():
    """ Tests a weight computation loop using matching the max payload criterion. """

    # with payload computed from NPAX
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", "mass_breakdown_inputs.xml"))
    reader.path_separator = ":"
    input_vars = reader.read(
        ignore=[
            "data:weight:aircraft:MLW",
            "data:weight:aircraft:MZFW",
            "data:weight:aircraft:max_payload",
        ]
    ).to_ivc()

    mass_computation_1 = run_system(MassBreakdown(payload_from_npax=True), input_vars)
    oew = mass_computation_1["data:weight:aircraft:OWE"]
    assert oew == pytest.approx(15782, abs=1)

    # with payload as input
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", "mass_breakdown_inputs.xml"))
    reader.path_separator = ":"
    input_vars = reader.read(
        ignore=[
            "data:weight:aircraft:MLW",
            "data:weight:aircraft:MZFW",
        ]
    ).to_ivc()
    mass_computation_2 = run_system(MassBreakdown(payload_from_npax=False), input_vars)
    oew = mass_computation_2["data:weight:aircraft:OWE"]
    assert oew == pytest.approx(15782, abs=1) # FIXME: the problem is that result remain the same whereas max_payload differs by 150kg