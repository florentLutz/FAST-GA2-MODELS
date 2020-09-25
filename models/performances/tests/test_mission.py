"""
Test takeoff module
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

import os.path as pth
import openmdao.api as om

import pytest
from fastoad.io import VariableIO
from fastoad.module_management import OpenMDAOSystemRegistry

from tests.testing_utilities import run_system
from ..mission import _compute_taxi, _compute_climb, _compute_cruise, _compute_descent

def get_indep_var_comp(var_names):
    """ Reads required input data and returns an IndepVarcomp() instance"""
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", "performances_inputs.xml"))
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

def register_wrappers():
    path_split = pth.dirname(__file__).split('\\')
    drive = path_split[0]
    del path_split[0]
    del path_split[-1]
    del path_split[-1]
    path = drive + "\\" + pth.join(*path_split)
    OpenMDAOSystemRegistry.explore_folder(path)

def test_compute_taxi():
    """ Tests taxi in/out phase """

    # Input list from model (not generated because NaN values not supported by vspcript/vspaero)
    input_list = [
        "data:geometry:propulsion:engine:count",
        "data:mission:sizing:taxi_out:thrust_rate",
        "data:mission:sizing:taxi_out:duration",
        "data:mission:sizing:taxi_in:thrust_rate",
        "data:mission:sizing:taxi_in:duration",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:propulsion:IC_engine:max_power", 134000)
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(_compute_taxi(
        propulsion_id="fastoad.wrapper.propulsion.basicIC_engine",
        taxi_out=True), ivc)
    fuel_mass = problem["data:mission:operational:taxi_out:fuel"]
    assert fuel_mass == pytest.approx(0.16, abs=1e-2)
    problem = run_system(_compute_taxi(
        propulsion_id="fastoad.wrapper.propulsion.basicIC_engine",
        taxi_out=False), ivc)
    fuel_mass = problem["data:mission:operational:taxi_in:fuel"]
    assert fuel_mass == pytest.approx(0.10, abs=1e-2)


def test_compute_climb():
    """ Tests climb phase """

    # Input list from model (not generated because NaN values not supported by vspcript/vspaero)
    input_list = [
        "data:geometry:propulsion:engine:count",
        "data:mission:sizing:main_route:cruise:altitude",
        "data:aerodynamics:aircraft:cruise:CD0",
        "data:aerodynamics:aircraft:cruise:coef_k",
        "data:geometry:wing:area",
        "data:weight:aircraft:MTOW",
        "data:mission:sizing:main_route:climb:thrust_rate"
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:propulsion:IC_engine:max_power", 134000)
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)
    ivc.add_output("data:mission:sizing:taxi_out:fuel", 0.16, units="kg")
    ivc.add_output("data:mission:sizing:takeoff:fuel", 1.2, units="kg")
    ivc.add_output("data:mission:sizing:initial_climb:fuel", 0.3, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(_compute_climb(
        propulsion_id="fastoad.wrapper.propulsion.basicIC_engine"), ivc)
    fuel_mass = problem["data:mission:sizing:main_route:climb:fuel"]
    assert fuel_mass == pytest.approx(5.5, abs=1e-1)
    distance = problem["data:mission:sizing:main_route:climb:distance"]/1000 # in km
    assert distance == pytest.approx(11.3, abs=1e-1)
    duration = problem["data:mission:sizing:main_route:climb:duration"]/60 # in min
    assert duration == pytest.approx(6, abs=1)


def test_compute_cruise():
    """ Tests cruise phase """

    # Input list from model (not generated because NaN values not supported by vspcript/vspaero)
    input_list = [
        "data:geometry:propulsion:engine:count",
        "data:TLAR:v_cruise",
        "data:TLAR:range",
        "data:mission:sizing:main_route:cruise:altitude",
        "data:aerodynamics:aircraft:cruise:CD0",
        "data:aerodynamics:aircraft:cruise:coef_k",
        "data:geometry:wing:area",
        "data:weight:aircraft:MTOW",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:propulsion:IC_engine:max_power", 134000)
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)
    ivc.add_output("data:mission:sizing:taxi_out:fuel", 0.16, units="kg")
    ivc.add_output("data:mission:sizing:takeoff:fuel", 1.2, units="kg")
    ivc.add_output("data:mission:sizing:initial_climb:fuel", 0.3, units="kg")
    ivc.add_output("data:mission:sizing:main_route:climb:fuel", 5.5, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(_compute_cruise(
        propulsion_id="fastoad.wrapper.propulsion.basicIC_engine"), ivc)
    fuel_mass = problem["data:mission:sizing:main_route:cruise:fuel"]
    assert fuel_mass == pytest.approx(356, abs=1)
    duration = problem["data:mission:sizing:main_route:cruise:duration"]/3600 # in hour
    assert duration == pytest.approx(5.2, abs=1e-1)


def test_compute_descent():
    """ Tests descent phase """

    # Input list from model (not generated because NaN values not supported by vspcript/vspaero)
    input_list = [
        "data:geometry:propulsion:engine:count",
        "data:mission:sizing:main_route:cruise:altitude",
        "data:mission:sizing:main_route:descent:descent_rate",
        "data:aerodynamics:aircraft:cruise:optimal_CL",
        "data:aerodynamics:aircraft:cruise:CD0",
        "data:aerodynamics:aircraft:cruise:coef_k",
        "data:geometry:wing:area",
        "data:weight:aircraft:MTOW",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:propulsion:IC_engine:max_power", 134000)
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)
    ivc.add_output("data:mission:sizing:taxi_out:fuel", 0.16, units="kg")
    ivc.add_output("data:mission:sizing:takeoff:fuel", 1.2, units="kg")
    ivc.add_output("data:mission:sizing:initial_climb:fuel", 0.3, units="kg")
    ivc.add_output("data:mission:sizing:main_route:climb:fuel", 5.5, units="kg")
    ivc.add_output("data:mission:sizing:main_route:cruise:fuel", 356, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(_compute_descent(
        propulsion_id="fastoad.wrapper.propulsion.basicIC_engine"), ivc)
    fuel_mass = problem["data:mission:sizing:main_route:descent:fuel"]
    assert fuel_mass == pytest.approx(0.5, abs=1e-1)
    distance = problem["data:mission:sizing:main_route:descent:distance"] / 1000  # in km
    assert distance == pytest.approx(45, abs=1)
    duration = problem["data:mission:sizing:main_route:descent:duration"]/60 # in min
    assert duration == pytest.approx(18.7, abs=1e-1)