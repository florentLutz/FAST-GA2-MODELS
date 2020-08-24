"""
Test module for geometry functions of cg components
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

from ..cg import ComputeAircraftCG
from ..cg_components import ComputeControlSurfacesCG
from ..cg_components import ComputeEngineCG
from ..cg_components import ComputeHTcg
from ..cg_components import ComputeCGLoadCase
from ..cg_components import ComputeOthersCG
from ..cg_components import ComputeCGRatioAft
from ..cg_components import ComputeTanksCG
from ..cg_components import ComputeWingCG
from ..cg_components import ComputeVTcg
from ..cg_components import ComputeMaxCGratio
from ..cg_components import UpdateMLG


def get_indep_var_comp(var_names):
    """ Reads required input data and returns an IndepVarcomp() instance"""
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", "cg_inputs.xml"))
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


def test_compute_cg_control_surfaces():
    """ Tests computation of control surfaces center of gravity """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeControlSurfacesCG(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeControlSurfacesCG(), ivc)
    x_cg_a4 = problem["data:weight:airframe:flight_controls:CG:x"]
    assert x_cg_a4 == pytest.approx(20.1, abs=1e-1)


def test_compute_cg_engine():
    """ Tests computation of engine(s) center of gravity """

    # Input list from model (not generated because of assertion error for bad propulsion layout values)
    input_list = [
        "data:geometry:propulsion:layout",
        "data:geometry:wing:MAC:leading_edge:x:local",
        "data:geometry:wing:MAC:y",
        "data:geometry:wing:root:y",
        "data:geometry:wing:root:chord",
        "data:geometry:wing:tip:leading_edge:x:local",
        "data:geometry:wing:tip:y",
        "data:geometry:wing:tip:chord",
        "data:geometry:wing:MAC:at25percent:x",
        "data:geometry:propulsion:nacelle:length",
        "data:geometry:propulsion:nacelle:y",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeEngineCG(), ivc)
    x_cg_eng = problem["data:weight:propulsion:engine:CG:x"]
    assert x_cg_eng == pytest.approx(2.2, abs=1e-1)


def test_compute_cg_ht():
    """ Tests computation of horizontal tail center of gravity """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeHTcg(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTcg(), ivc)
    x_cg_ht = problem["data:weight:airframe:horizontal_tail:CG:x"]
    assert x_cg_ht == pytest.approx(18.5, abs=1e-1)


def test_compute_cg_vt():
    """ Tests computation of vertical tail center of gravity """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeVTcg(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTcg(), ivc)
    x_cg_vt = problem["data:weight:airframe:vertical_tail:CG:x"]
    assert x_cg_vt == pytest.approx(20.8, abs=1e-1)


def test_compute_cg_tank():
    """ Tests tank center of gravity """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeTanksCG(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeTanksCG(), ivc)
    x_cg_tank = problem["data:weight:fuel_tank:CG:x"]
    assert x_cg_tank == pytest.approx(17.57, abs=1e-2)


def test_compute_cg_wing():
    """ Tests computation of wing center of gravity """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeWingCG(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingCG(), ivc)
    x_cg_wing = problem["data:weight:airframe:wing:CG:x"]
    assert x_cg_wing == pytest.approx(3.47, abs=1e-2)


def test_compute_cg_loadcase():
    """ Tests computation of center of gravity for all load case conf. """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeCGLoadCase(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file  and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:weight:payload:PAX:CG:x", 22.1)
    ivc.add_output("data:weight:payload:rear_fret:CG:x", 22.1)
    ivc.add_output("data:weight:payload:front_fret:CG:x", 22.1)
    ivc.add_output("data:weight:aircraft_empty:CG:x", 22.1)
    ivc.add_output("data:weight:aircraft_empty:mass", 22.1)
    ivc.add_output("data:weight:fuel_tank:CG:x", 22.1)

    # Run problem and check obtained value(s) is/(are) correct
    for case in range(1,7):
        problem = run_system(ComputeCGLoadCase(load_case=case), ivc)
        cg_ratio_lc = problem["data:weight:aircraft:load_case_"+str(case)+":CG:MAC_position"]
        if case == 1:
            assert cg_ratio_lc == pytest.approx(1.51, abs=1e-1)
        elif case == 2:
            assert cg_ratio_lc == pytest.approx(1.51, abs=1e-1)
        elif case == 3:
            assert cg_ratio_lc == pytest.approx(1.51, abs=1e-1)
        elif case == 4:
            assert cg_ratio_lc == pytest.approx(1.51, abs=1e-1)
        elif case == 5:
            assert cg_ratio_lc == pytest.approx(1.51, abs=1e-1)
        elif case == 6:
            assert cg_ratio_lc == pytest.approx(1.51, abs=1e-1)


def test_compute_cg_others():
    """ Tests computation of other components center of gravity """

    # Input list from model (not generated because of assertion error for bad propulsion layout values)
    input_list = [
        "data:geometry:cabin:NPAX",
        "data:geometry:propulsion:layout",
        "data:geometry:wing:MAC:length",
        "data:geometry:fuselage:length",
        "data:geometry:wing:MAC:at25percent:x",
        "data:geometry:fuselage:front_length",
        "data:weight:propulsion:engine:CG:x",
        "data:geometry:propulsion:propeller:depth",
        "data:geometry:cabin:seats:passenger:count_by_row",
        "data:geometry:cabin:seats:pilot:length",
        "data:geometry:cabin:seats:passenger:length",
    ]

    # Research independent input value in .xml file  and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:weight:propulsion:engine:CG:x", 22.1)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeOthersCG(), ivc)
    x_cg_a2 = problem["data:weight:airframe:fuselage:CG:x"]
    assert x_cg_a2 == pytest.approx(4.02, abs=1e-2)
    x_cg_a4 = problem["data:weight:airframe:flight_controls:CG:x"]
    assert x_cg_a4 == pytest.approx(20.91, abs=1e-2)
    x_cg_a52 = problem["data:weight:airframe:landing_gear:front:CG:x"]
    assert x_cg_a52 == pytest.approx(2.55, abs=1e-2)
    x_cg_b2 = problem["data:weight:propulsion:fuel_lines:CG:x"]
    assert x_cg_b2 == pytest.approx(22.1, abs=1e-1)
    x_cg_c12 = problem["data:weight:systems:power:electric_systems:CG:x"]
    assert x_cg_c12 == pytest.approx(0, abs=1e-2)
    x_cg_c13 = problem["data:weight:systems:power:hydraulic_systems:CG:x"]
    assert x_cg_c13 == pytest.approx(0, abs=1e-2)
    x_cg_c22 = problem["data:weight:systems:life_support:air_conditioning:CG:x"]
    assert x_cg_c22 == pytest.approx(0, abs=1e-2)
    x_cg_c3 = problem["data:weight:systems:navigation:CG:x"]
    assert x_cg_c3 == pytest.approx(4.1, abs=1e-2)
    x_cg_d2 = problem["data:weight:furniture:passenger_seats:CG:x"]
    assert x_cg_d2 == pytest.approx(5.25, abs=1e-1)
    x_cg_pl = problem["data:weight:payload:PAX:CG:x"]
    assert x_cg_pl == pytest.approx(5.25, abs=1e-1)
    x_cg_rear_fret = problem["data:weight:payload:rear_fret:CG:x"]
    assert x_cg_rear_fret == pytest.approx(3.4, abs=1e-2)
    x_cg_front_fret = problem["data:weight:payload:front_fret:CG:x"]
    assert x_cg_front_fret == pytest.approx(3.4, abs=1e-2)


def test_compute_cg_ratio_aft():
    """ Tests computation of center of gravity with aft estimation """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeCGRatioAft(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file  and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:weight:airframe:wing:CG:x", 2.3)
    ivc.add_output("data:weight:airframe:fuselage:CG:x", 2.3)
    ivc.add_output("data:weight:airframe:horizontal_tail:CG:x", 2.3)
    ivc.add_output("data:weight:airframe:vertical_tail:CG:x", 2.3)
    ivc.add_output("data:weight:airframe:flight_controls:CG:x", 2.3)
    ivc.add_output("data:weight:airframe:landing_gear:main:CG:x", 2.3)
    ivc.add_output("data:weight:airframe:landing_gear:front:CG:x", 2.3)
    ivc.add_output("data:weight:propulsion:engine:CG:x", 2.3)
    ivc.add_output("data:weight:propulsion:fuel_lines:CG:x", 2.3)
    ivc.add_output("data:weight:systems:power:electric_systems:CG:x", 2.3)
    ivc.add_output("data:weight:systems:power:hydraulic_systems:CG:x", 2.3)
    ivc.add_output("data:weight:systems:life_support:air_conditioning:CG:x", 2.3)
    ivc.add_output("data:weight:systems:navigation:CG:x", 2.3)
    ivc.add_output("data:weight:furniture:passenger_seats:CG:x", 2.3)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCGRatioAft(), ivc)
    empty_mass = problem["data:weight:aircraft_empty:mass"]
    assert empty_mass == pytest.approx(5141.34, abs=1)
    cg_ratio_aft = problem["data:weight:aircraft_empty:CG:x"]
    assert cg_ratio_aft == pytest.approx(2.3, abs=1e-1)
    cg_mac_pos = problem["data:weight:aircraft:empty:CG:MAC_position"]
    assert cg_mac_pos == pytest.approx(-2.9, abs=1e-1)


def test_compute_max_cg_ratio():
    """ Tests computation of maximum center of gravity ratio """

    # Define the independent input values that should be filled if basic function is choosen
    ivc = om.IndepVarComp()
    ivc.add_output("data:weight:aircraft:empty:CG:MAC_position", 0.387846)
    ivc.add_output("data:weight:aircraft:load_case_1:CG:MAC_position", 0.364924)
    ivc.add_output("data:weight:aircraft:load_case_2:CG:MAC_position", 0.285139)
    ivc.add_output("data:weight:aircraft:load_case_3:CG:MAC_position", 0.386260)
    ivc.add_output("data:weight:aircraft:load_case_4:CG:MAC_position", 0.388971)
    ivc.add_output("data:weight:aircraft:load_case_5:CG:MAC_position", 0.388971)
    ivc.add_output("data:weight:aircraft:load_case_6:CG:MAC_position", 0.388971)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMaxCGratio(), ivc)
    cg_ratio = problem["data:weight:aircraft:CG:aft:MAC_position"]
    assert cg_ratio == pytest.approx(0.438971, abs=1e-6)


def test_compute_aircraft_cg():
    """ Tests computation of static margin """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeAircraftCG(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file  and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:weight:aircraft:CG:aft:MAC_position", 0.388971)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeAircraftCG(), ivc)
    cg_global = problem["data:weight:aircraft:CG:aft:x"]
    assert cg_global == pytest.approx(17.1, abs=1e-1)


def test_update_mlg():
    """ Tests computation of MLG update """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", UpdateMLG(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file  and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:weight:aircraft:CG:aft:MAC_position", 0.388971)
    ivc.add_output("data:weight:airframe:landing_gear:front:CG:x", 1.453)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(UpdateMLG(), ivc)
    cg_a51 = problem["data:weight:airframe:landing_gear:main:CG:x"]
    assert cg_a51 == pytest.approx(23.7, abs=1e-1)
