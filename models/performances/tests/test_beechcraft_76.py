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

# pylint: disable=raising-bad-type
import os.path as pth
import openmdao.api as om

import pytest
from fastoad.io import VariableIO
from fastoad.module_management import OpenMDAOSystemRegistry

from ...tests.testing_utilities import run_system
from ..takeoff import TakeOffPhase, _v2, _vr_from_v2, _vloff_from_v2, _simulate_takeoff
from ..mission import _compute_taxi, _compute_climb, _compute_cruise, _compute_descent
from ..sizing import Sizing

XML_FILE = "beechcraft_76.xml"
ENGINE_WRAPPER = "fastga.wrapper.propulsion.basicIC_engine"


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


def register_wrappers():
    path_split = pth.dirname(__file__).split('\\')
    drive = path_split[0]
    del path_split[0]
    del path_split[-1]
    del path_split[-1]
    path = drive + "\\" + pth.join(*path_split)
    OpenMDAOSystemRegistry.explore_folder(path)


def test_v2():
    """ Tests safety speed """

    # Input list from model (not generated because NaN values not supported by vspcript/vspaero)
    input_list = [
        "data:geometry:propulsion:engine:count",
        "data:aerodynamics:wing:low_speed:CL_max_clean",
        "data:aerodynamics:aircraft:low_speed:CL0_clean",
        "data:aerodynamics:flaps:takeoff:CL",
        "data:aerodynamics:aircraft:low_speed:CL_alpha",
        "data:aerodynamics:aircraft:low_speed:CD0",
        "data:aerodynamics:flaps:takeoff:CD",
        "data:aerodynamics:aircraft:low_speed:coef_k",
        "data:geometry:wing:area",
        "data:geometry:wing:span",
        "data:geometry:landing_gear:height",
        "data:weight:aircraft:MTOW",
        "data:mission:sizing:takeoff:thrust_rate",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:propulsion:IC_engine:max_power", 130000)  # correct value to fit old version def.
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(_v2(propulsion_id=ENGINE_WRAPPER), ivc)
    v2 = problem.get_val("v2:v2", units="m/s")
    assert v2 == pytest.approx(37.79, abs=1e-2)
    alpha = problem.get_val("v2:alpha", units="deg")
    assert alpha == pytest.approx(8.49, abs=1e-2)


def test_vloff():
    """ Tests lift-off speed """

    # Input list from model (not generated because NaN values not supported by vspcript/vspaero)
    input_list = [
        "data:geometry:propulsion:engine:count",
        "data:aerodynamics:aircraft:low_speed:CL0_clean",
        "data:aerodynamics:flaps:takeoff:CL",
        "data:aerodynamics:aircraft:low_speed:CL_alpha",
        "data:aerodynamics:aircraft:low_speed:CD0",
        "data:aerodynamics:flaps:takeoff:CD",
        "data:aerodynamics:aircraft:low_speed:coef_k",
        "data:geometry:wing:area",
        "data:geometry:wing:span",
        "data:geometry:landing_gear:height",
        "data:weight:aircraft:MTOW",
        "data:mission:sizing:takeoff:thrust_rate",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:propulsion:IC_engine:max_power", 130000)  # correct value to fit old version def.
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)
    ivc.add_output("vloff:v2", 37.79, units='m/s')
    ivc.add_output("vloff:alpha_v2", 8.49, units='deg')

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(_vloff_from_v2(propulsion_id=ENGINE_WRAPPER), ivc)
    vloff = problem.get_val("vloff:vloff", units="m/s")
    assert vloff == pytest.approx(36.88, abs=1e-2)
    alpha = problem.get_val("vloff:alpha", units="deg")
    assert alpha == pytest.approx(8.49, abs=1e-2)


def test_vr():
    """ Tests rotation speed """

    # Input list from model (not generated because NaN values not supported by vspcript/vspaero)
    input_list = [
        "data:geometry:propulsion:engine:count",
        "data:aerodynamics:aircraft:low_speed:CL0_clean",
        "data:aerodynamics:flaps:takeoff:CL",
        "data:aerodynamics:aircraft:low_speed:CL_alpha",
        "data:aerodynamics:aircraft:low_speed:CD0",
        "data:aerodynamics:flaps:takeoff:CD",
        "data:aerodynamics:aircraft:low_speed:coef_k",
        "data:geometry:wing:area",
        "data:geometry:wing:span",
        "data:geometry:landing_gear:height",
        "data:weight:aircraft:MTOW",
        "data:mission:sizing:takeoff:thrust_rate",
        "data:mission:sizing:takeoff:friction_coefficient_no_brake",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:propulsion:IC_engine:max_power", 130000)  # correct value to fit old version def.
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)
    ivc.add_output("vr:vloff", 36.88, units='m/s')
    ivc.add_output("vr:alpha_vloff", 8.49, units='deg')

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(_vr_from_v2(propulsion_id=ENGINE_WRAPPER), ivc)
    vr = problem.get_val("vr:vr", units="m/s")
    assert vr == pytest.approx(28.51, abs=1e-2)


def test_simulate_takeoff():
    """ Tests simulate takeoff """

    # Input list from model (not generated because NaN values not supported by vspcript/vspaero)
    input_list = [
        "data:geometry:propulsion:engine:count",
        "data:aerodynamics:wing:low_speed:CL_max_clean",
        "data:aerodynamics:aircraft:low_speed:CL0_clean",
        "data:aerodynamics:flaps:takeoff:CL",
        "data:aerodynamics:aircraft:low_speed:CL_alpha",
        "data:aerodynamics:aircraft:low_speed:CD0",
        "data:aerodynamics:flaps:takeoff:CD",
        "data:aerodynamics:aircraft:low_speed:coef_k",
        "data:geometry:wing:area",
        "data:geometry:wing:span",
        "data:geometry:landing_gear:height",
        "data:weight:aircraft:MTOW",
        "data:mission:sizing:takeoff:thrust_rate",
        "data:mission:sizing:takeoff:friction_coefficient_no_brake",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:propulsion:IC_engine:max_power", 130000)  # correct value to fit old version def.
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)
    ivc.add_output("takeoff:min_vr", 28.51, units='m/s')
    ivc.add_output("takeoff:alpha_v2", 8.49, units='deg')

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(_simulate_takeoff(propulsion_id=ENGINE_WRAPPER), ivc)
    vr = problem.get_val("data:mission:sizing:takeoff:VR", units='m/s')
    assert vr == pytest.approx(34.62, abs=1e-2)
    vloff = problem.get_val("data:mission:sizing:takeoff:VLOF", units='m/s')
    assert vloff == pytest.approx(40.15, abs=1e-2)
    v2 = problem.get_val("data:mission:sizing:takeoff:V2", units='m/s')
    assert v2 == pytest.approx(42.61, abs=1e-2)
    tofl = problem.get_val("data:mission:sizing:takeoff:TOFL", units='m')
    assert tofl == pytest.approx(291, abs=1)
    duration = problem.get_val("data:mission:sizing:takeoff:duration", units='s')
    assert duration == pytest.approx(17.5, abs=1e-1)
    fuel1 = problem.get_val("data:mission:sizing:takeoff:fuel", units='kg')
    assert fuel1 == pytest.approx(0.29, abs=1e-2)
    fuel2 = problem.get_val("data:mission:sizing:initial_climb:fuel", units='kg')
    assert fuel2 == pytest.approx(0.07, abs=1e-2)


def test_takeoffphase_connections():
    """ Tests complete take-off phase connection with speeds """

    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    input_vars.add_output("data:propulsion:IC_engine:max_power", 130000)  # correct value to fit old version def.
    input_vars.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    input_vars.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)
    register_wrappers()
    problem = run_system(TakeOffPhase(propulsion_id=ENGINE_WRAPPER), input_vars)
    vr = problem.get_val("data:mission:sizing:takeoff:VR", units='m/s')
    assert vr == pytest.approx(34.62, abs=1e-2)
    vloff = problem.get_val("data:mission:sizing:takeoff:VLOF", units='m/s')
    assert vloff == pytest.approx(40.15, abs=1e-2)
    v2 = problem.get_val("data:mission:sizing:takeoff:V2", units='m/s')
    assert v2 == pytest.approx(42.61, abs=1e-2)
    tofl = problem.get_val("data:mission:sizing:takeoff:TOFL", units='m')
    assert tofl == pytest.approx(291, abs=1)
    duration = problem.get_val("data:mission:sizing:takeoff:duration", units='s')
    assert duration == pytest.approx(17.5, abs=1e-1)
    fuel1 = problem.get_val("data:mission:sizing:takeoff:fuel", units='kg')
    assert fuel1 == pytest.approx(0.29, abs=1e-2)
    fuel2 = problem.get_val("data:mission:sizing:initial_climb:fuel", units='kg')
    assert fuel2 == pytest.approx(0.07, abs=1e-2)


def test_compute_taxi():
    """ Tests taxi in/out phase """

    # Input list from model (not generated because NaN values not supported by vspcript/vspaero)
    input_list = [
        "data:geometry:propulsion:engine:count",
        "data:mission:sizing:taxi_out:thrust_rate",
        "data:mission:sizing:taxi_out:duration",
        "data:mission:sizing:taxi_out:speed",
        "data:mission:sizing:taxi_in:thrust_rate",
        "data:mission:sizing:taxi_in:duration",
        "data:mission:sizing:taxi_in:speed",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:propulsion:IC_engine:max_power", 130000)  # correct value to fit old version def.
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(_compute_taxi(propulsion_id=ENGINE_WRAPPER, taxi_out=True), ivc)
    fuel_mass = problem.get_val("data:mission:sizing:taxi_out:fuel", units="kg")
    assert fuel_mass == pytest.approx(0.50, abs=1e-2)  # result strongly dependent on the defined Thrust limit
    problem = run_system(_compute_taxi(propulsion_id=ENGINE_WRAPPER, taxi_out=False), ivc)
    fuel_mass = problem.get_val("data:mission:sizing:taxi_in:fuel", units="kg")
    assert fuel_mass == pytest.approx(0.50, abs=1e-2)  # result strongly dependent on the defined Thrust limit


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
    ivc.add_output("data:propulsion:IC_engine:max_power", 130000)  # correct value to fit old version def.
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)
    ivc.add_output("data:mission:sizing:taxi_out:fuel", 0.50, units="kg")
    ivc.add_output("data:mission:sizing:takeoff:fuel", 0.29, units="kg")
    ivc.add_output("data:mission:sizing:initial_climb:fuel", 0.07, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(_compute_climb(propulsion_id=ENGINE_WRAPPER), ivc)
    v_cas = problem.get_val("data:mission:sizing:main_route:climb:v_cas", units="kn")
    assert v_cas == pytest.approx(71.5, abs=1)
    fuel_mass = problem.get_val("data:mission:sizing:main_route:climb:fuel", units="kg")
    assert fuel_mass == pytest.approx(5.56, abs=1e-2)
    distance = problem.get_val("data:mission:sizing:main_route:climb:distance", units="m") / 1000.0  # conversion to km
    assert distance == pytest.approx(13.2, abs=1e-1)
    duration = problem.get_val("data:mission:sizing:main_route:climb:duration", units="min")
    assert duration == pytest.approx(5.7, abs=1e-1)


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
    ivc.add_output("data:propulsion:IC_engine:max_power", 130000)  # correct value to fit old version def.
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)
    ivc.add_output("data:mission:sizing:taxi_out:fuel", 0.50, units="kg")
    ivc.add_output("data:mission:sizing:takeoff:fuel", 0.29, units="kg")
    ivc.add_output("data:mission:sizing:initial_climb:fuel", 0.07, units="kg")
    ivc.add_output("data:mission:sizing:main_route:climb:fuel", 5.56, units="kg")
    ivc.add_output("data:mission:sizing:main_route:climb:distance", 13.2, units="km")

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(_compute_cruise(propulsion_id=ENGINE_WRAPPER), ivc)
    fuel_mass = problem.get_val("data:mission:sizing:main_route:cruise:fuel", units="kg")
    assert fuel_mass == pytest.approx(187.87, abs=1e-2)
    duration = problem.get_val("data:mission:sizing:main_route:cruise:duration", units="h")
    assert duration == pytest.approx(4.9, abs=1e-1)


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
    ivc.add_output("data:propulsion:IC_engine:max_power", 130000)  # correct value to fit old version def.
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)
    ivc.add_output("data:mission:sizing:taxi_out:fuel", 0.98, units="kg")
    ivc.add_output("data:mission:sizing:takeoff:fuel", 0.29, units="kg")
    ivc.add_output("data:mission:sizing:initial_climb:fuel", 0.07, units="kg")
    ivc.add_output("data:mission:sizing:main_route:climb:fuel", 5.56, units="kg")
    ivc.add_output("data:mission:sizing:main_route:cruise:fuel", 188.05, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(_compute_descent(propulsion_id=ENGINE_WRAPPER), ivc)
    fuel_mass = problem["data:mission:sizing:main_route:descent:fuel"]
    assert fuel_mass == pytest.approx(0.09, abs=1e-2)
    distance = problem.get_val("data:mission:sizing:main_route:descent:distance", units="m") / 1000  # conversion to km
    assert distance == pytest.approx(48.4, abs=1e-1)
    duration = problem.get_val("data:mission:sizing:main_route:descent:duration", units="min")
    assert duration == pytest.approx(15.9, abs=1e-1)


def test_loop_cruise_distance():
    """ Tests a distance computation loop matching the descent value/TLAR total range. """

    # Get the parameters from .xml
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    input_vars.add_output("data:propulsion:IC_engine:max_power", 130000)  # correct value to fit old version def.
    input_vars.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    input_vars.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(Sizing(propulsion_id=ENGINE_WRAPPER), input_vars)
    m_total = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert m_total == pytest.approx(214.0, abs=1e-1)
    climb_distance = problem.get_val("data:mission:sizing:main_route:climb:distance", units="NM")
    cruise_distance = problem.get_val("data:mission:sizing:main_route:cruise:distance", units="NM")
    descent_distance = problem.get_val("data:mission:sizing:main_route:descent:distance", units="NM")
    total_distance = problem.get_val("data:TLAR:range", units="NM")
    error_distance = total_distance - (climb_distance + cruise_distance + descent_distance)
    assert error_distance == pytest.approx(0.0, abs=1e-1)
