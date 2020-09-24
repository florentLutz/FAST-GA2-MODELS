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
import shutil
import openmdao.api as om
import numpy as np

import pytest
from fastoad.io import VariableIO
from fastoad.module_management import OpenMDAOSystemRegistry

from pytest import approx
from fastoad.models.aerodynamics.constants import POLAR_POINT_COUNT
from tests.testing_utilities import run_system
from ..takeoff import _v2, _vr, _vloff, _simulate_takeoff

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
    path = drive + "\\" + pth.join(*path_split) #+ "\\propulsion\\fuel_propulsion\\basicIC_engine"
    OpenMDAOSystemRegistry.explore_folder(path)
    #OpenMDAOSystemRegistry.explore_folder("D:\\a.reysset\\Documents\\Github\\FAST-GA2-MODELS\\models\\propulsion\\fuel_propulsion\\basicIC_engine")

def test_v2():
    """ Tests safety speed """

    # Input list from model (not generated because NaN values not supported by vspcript/vspaero)
    input_list = [
        "data:geometry:propulsion:engine:count",
        "data:aerodynamics:aircraft:takeoff:CL_max",
        "data:aerodynamics:aircraft:low_speed:CL0_clean",
        "data:aerodynamics:flaps:takeoff:CL",
        "data:aerodynamics:aircraft:low_speed:CL_alpha",
        "data:aerodynamics:aircraft:low_speed:CD0",
        "data:aerodynamics:flaps:takeoff:CD",
        "data:aerodynamics:aircraft:low_speed:coef_k",
        "data:geometry:wing:area",
        "data:weight:aircraft:MTOW",
        "data:mission:sizing:takeoff:thrust_rate",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:propulsion:IC_engine:max_power", 134000)
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)
    ivc.add_output("data:mission:sizing:takeoff:thrust_rate", 1.0)
    ivc.add_output("data:mission:sizing:main_route:climb:min_climb_rate", 0.083)

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(_v2(propulsion_id="fastoad.wrapper.propulsion.basicIC_engine"), ivc)
    v2 = problem["v2:v2"]
    assert v2 == pytest.approx(33.6, abs=1e-1)
    alpha = problem["v2:alpha"]
    assert alpha == pytest.approx(11.3, abs=1e-1)

def test_vloff():
    """ Tests lift-off speed """

    # Input list from model (not generated because NaN values not supported by vspcript/vspaero)
    input_list = [
        "data:geometry:propulsion:engine:count",
        "data:aerodynamics:aircraft:takeoff:CL_max",
        "data:aerodynamics:aircraft:low_speed:CL0_clean",
        "data:aerodynamics:flaps:takeoff:CL",
        "data:aerodynamics:aircraft:low_speed:CL_alpha",
        "data:aerodynamics:aircraft:low_speed:CD0",
        "data:aerodynamics:flaps:takeoff:CD",
        "data:aerodynamics:aircraft:low_speed:coef_k",
        "data:geometry:wing:area",
        "data:weight:aircraft:MTOW",
        "data:mission:sizing:takeoff:thrust_rate",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:propulsion:IC_engine:max_power", 134000)
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)
    ivc.add_output("data:mission:sizing:takeoff:thrust_rate", 1.0)
    ivc.add_output("vloff:v2", 33.6, units='m/s')
    ivc.add_output("vloff:alpha_v2", 11.3, units='deg')

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(_vloff(propulsion_id="fastoad.wrapper.propulsion.basicIC_engine"), ivc)
    vloff = problem["vloff:vloff"]
    assert vloff == pytest.approx(32.3, abs=1e-1)
    alpha = problem["vloff:alpha"]
    assert alpha == pytest.approx(11.3, abs=1e-1)


def test_vr():
    """ Tests rotation speed """

    # Input list from model (not generated because NaN values not supported by vspcript/vspaero)
    input_list = [
        "data:geometry:propulsion:engine:count",
        "data:aerodynamics:aircraft:takeoff:CL_max",
        "data:aerodynamics:aircraft:low_speed:CL0_clean",
        "data:aerodynamics:flaps:takeoff:CL",
        "data:aerodynamics:aircraft:low_speed:CL_alpha",
        "data:aerodynamics:aircraft:low_speed:CD0",
        "data:aerodynamics:flaps:takeoff:CD",
        "data:aerodynamics:aircraft:low_speed:coef_k",
        "data:geometry:wing:area",
        "data:weight:aircraft:MTOW",
        "data:mission:sizing:takeoff:thrust_rate",
        "data:mission:sizing:takeoff:friction_coefficient_no_brake",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:propulsion:IC_engine:max_power", 134000)
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)
    ivc.add_output("data:mission:sizing:takeoff:thrust_rate", 1.0)
    ivc.add_output("vr:vloff", 32.3, units='m/s')
    ivc.add_output("vr:alpha_vloff", 11.3, units='deg')

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(_vr(propulsion_id="fastoad.wrapper.propulsion.basicIC_engine"), ivc)
    vr = problem["data:mission:sizing:takeoff:VR"]
    assert vr == pytest.approx(15.0, abs=1e-1)


def test_simulate_takeoff():
    """ Tests simulate takeoff """

    # Input list from model (not generated because NaN values not supported by vspcript/vspaero)
    input_list = [
        "data:geometry:propulsion:engine:count",
        "data:aerodynamics:aircraft:takeoff:CL_max",
        "data:aerodynamics:aircraft:low_speed:CL0_clean",
        "data:aerodynamics:flaps:takeoff:CL",
        "data:aerodynamics:aircraft:low_speed:CL_alpha",
        "data:aerodynamics:aircraft:low_speed:CD0",
        "data:aerodynamics:flaps:takeoff:CD",
        "data:aerodynamics:aircraft:low_speed:coef_k",
        "data:geometry:wing:area",
        "data:weight:aircraft:MTOW",
        "data:mission:sizing:takeoff:thrust_rate",
        "data:mission:sizing:takeoff:friction_coefficient_no_brake",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:propulsion:IC_engine:max_power", 134000)
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)
    ivc.add_output("data:mission:sizing:takeoff:thrust_rate", 1.0)
    ivc.add_output("data:mission:sizing:takeoff:VR", 15.0, units='m/s')
    ivc.add_output("takeoff:alpha_v2", 11.3, units='deg')

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(_simulate_takeoff(propulsion_id="fastoad.wrapper.propulsion.basicIC_engine"), ivc)
    vloff = problem["data:mission:sizing:takeoff:VLOF"]
    assert vloff == pytest.approx(32.5, abs=1e-1)
    v2 = problem["data:mission:sizing:takeoff:V2"]
    assert v2 == pytest.approx(37.6, abs=1e-1)
    tofl = problem["data:mission:sizing:takeoff:TOFL"]
    assert tofl == pytest.approx(105, abs=1)
    duration = problem["data:mission:sizing:takeoff:duration"]
    assert duration == pytest.approx(4.8, abs=1e-1)
    fuel1 = problem["data:mission:sizing:takeoff:fuel"]
    assert fuel1 == pytest.approx(0.10, abs=1e-2)
    fuel2 = problem["data:mission:sizing:initial_climb:fuel"]
    assert fuel2 == pytest.approx(0.08, abs=1e-2)