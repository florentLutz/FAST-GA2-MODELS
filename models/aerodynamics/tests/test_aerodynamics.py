"""
Test module for aerodynamics groups
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

from pytest import approx
from tests.testing_utilities import run_system
from ..aerodynamics_high_speed import AerodynamicsHighSpeed
from ..aerodynamics_low_speed import AerodynamicsLowSpeed

def get_indep_var_comp(var_names):
    """ Reads required input data and returns an IndepVarcomp() instance"""
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", "aerodynamics_inputs.xml"))
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


def test_aerodynamics_high_speed():
    """ Tests AerodynamicsHighSpeed """


    input_list = [
        "data:geometry:propulsion:engine:count",
        "data:geometry:propulsion:fan:length",
        "data:geometry:flap:chord_ratio",
        "data:geometry:flap:span_ratio",
        "data:geometry:fuselage:maximum_height",
        "data:geometry:fuselage:length",
        "data:geometry:fuselage:wetted_area",
        "data:geometry:fuselage:maximum_width",
        "data:geometry:horizontal_tail:MAC:length",
        "data:geometry:horizontal_tail:sweep_25",
        "data:geometry:horizontal_tail:thickness_ratio",
        "data:geometry:horizontal_tail:wetted_area",
        "data:geometry:propulsion:nacelle:length",
        "data:geometry:propulsion:nacelle:wetted_area",
        "data:geometry:propulsion:pylon:length",
        "data:geometry:propulsion:pylon:wetted_area",
        "data:geometry:aircraft:wetted_area",
        "data:geometry:slat:chord_ratio",
        "data:geometry:slat:span_ratio",
        "data:geometry:vertical_tail:MAC:length",
        "data:geometry:vertical_tail:sweep_25",
        "data:geometry:vertical_tail:thickness_ratio",
        "data:geometry:vertical_tail:wetted_area",
        "data:geometry:wing:area",
        "data:geometry:wing:MAC:length",
        "data:geometry:wing:root:chord",
        "data:geometry:wing:tip:chord",
        "data:geometry:wing:span",
        "data:geometry:wing:sweep_0",
        "data:geometry:wing:sweep_100_outer",
        "data:geometry:wing:sweep_25",
        "data:geometry:wing:thickness_ratio",
        "data:geometry:wing:wetted_area",
        "tuning:aerodynamics:aircraft:cruise:CD:k",
        "tuning:aerodynamics:aircraft:cruise:CL:k",
        "tuning:aerodynamics:aircraft:landing:CL_max:landing_gear_effect:k",
        "tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:k",
        "tuning:aerodynamics:aircraft:cruise:CL:winglet_effect:k",
        "tuning:aerodynamics:aircraft:cruise:CD:offset",
        "tuning:aerodynamics:aircraft:cruise:CL:offset",
        "tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:offset",
        "tuning:aerodynamics:aircraft:cruise:CL:winglet_effect:offset",
        "data:mission:sizing:cruise:altitude",
        "data:mission:sizing:landing:flap_angle",
        "data:mission:sizing:landing:slat_angle",
        "data:TLAR:cruise_mach",
        "data:TLAR:approach_speed",
    ]

    # Generate input list from model
    group = om.Group()
    ivc = om.IndepVarComp()
    ivc.add_output("data:TLAR:v_cruise", 150)
    ivc.add_output("data:mission:sizing:cruise:altitude", 50)
    group.add_subsystem("constants", ivc, promotes=["*"])
    group.add_subsystem("my_model", AerodynamicsHighSpeed(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    problem = run_system(AerodynamicsHighSpeed(), ivc)

    cd = problem["data:aerodynamics:aircraft:cruise:CD"]
    cl = problem["data:aerodynamics:aircraft:cruise:CL"]

    assert cd[cl == 0.0] == approx(0.02030, abs=1e-5)
    assert cd[cl == 0.2] == approx(0.02209, abs=1e-5)
    assert cd[cl == 0.42] == approx(0.02897, abs=1e-5)
    assert cd[cl == 0.85] == approx(0.11781, abs=1e-5)

    assert problem["data:aerodynamics:aircraft:cruise:optimal_CL"] == approx(0.54, abs=1e-3)
    assert problem["data:aerodynamics:aircraft:cruise:optimal_CD"] == approx(0.03550, abs=1e-5)


def test_aerodynamics_low_speed():
    """ Tests AerodynamicsLowSpeed """
    input_list = [
        "data:geometry:propulsion:engine:count",
        "data:geometry:propulsion:fan:length",
        "data:geometry:flap:chord_ratio",
        "data:geometry:flap:span_ratio",
        "data:geometry:fuselage:maximum_height",
        "data:geometry:fuselage:length",
        "data:geometry:fuselage:wetted_area",
        "data:geometry:fuselage:maximum_width",
        "data:geometry:horizontal_tail:MAC:length",
        "data:geometry:horizontal_tail:sweep_25",
        "data:geometry:horizontal_tail:thickness_ratio",
        "data:geometry:horizontal_tail:wetted_area",
        "data:geometry:propulsion:nacelle:length",
        "data:geometry:propulsion:nacelle:wetted_area",
        "data:geometry:propulsion:pylon:length",
        "data:geometry:propulsion:pylon:wetted_area",
        "data:geometry:aircraft:wetted_area",
        "data:geometry:slat:chord_ratio",
        "data:geometry:slat:span_ratio",
        "data:geometry:vertical_tail:MAC:length",
        "data:geometry:vertical_tail:sweep_25",
        "data:geometry:vertical_tail:thickness_ratio",
        "data:geometry:vertical_tail:wetted_area",
        "data:geometry:wing:area",
        "data:geometry:wing:aspect_ratio",
        "data:geometry:wing:MAC:length",
        "data:geometry:wing:root:chord",
        "data:geometry:wing:tip:chord",
        "data:geometry:wing:span",
        "data:geometry:wing:sweep_0",
        "data:geometry:wing:sweep_100_outer",
        "data:geometry:wing:sweep_25",
        "data:geometry:wing:thickness_ratio",
        "data:geometry:wing:tip:thickness_ratio",
        "data:geometry:wing:wetted_area",
        "tuning:aerodynamics:aircraft:cruise:CD:k",
        "tuning:aerodynamics:aircraft:cruise:CL:k",
        "tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:k",
        "tuning:aerodynamics:aircraft:cruise:CL:winglet_effect:k",
        "tuning:aerodynamics:aircraft:cruise:CD:offset",
        "tuning:aerodynamics:aircraft:cruise:CL:offset",
        "tuning:aerodynamics:aircraft:cruise:CD:winglet_effect:offset",
        "tuning:aerodynamics:aircraft:cruise:CL:winglet_effect:offset",
        "data:mission:sizing:landing:flap_angle",
        "data:mission:sizing:landing:slat_angle",
        "data:TLAR:approach_speed",
    ]

    ivc = get_indep_var_comp(input_list)
    problem = run_system(AerodynamicsLowSpeed(), ivc)

    cd = problem["aerodynamics:Cd_low_speed"]
    cl = problem["cl_low_speed"]

    assert cd[cl == 0.0] == approx(0.02173, abs=1e-5)
    assert cd[cl == 0.2] == approx(0.02339, abs=1e-5)
    assert cd[cl == 0.42] == approx(0.02974, abs=1e-5)
    assert cd[cl == 0.85] == approx(0.06010, abs=1e-5)
