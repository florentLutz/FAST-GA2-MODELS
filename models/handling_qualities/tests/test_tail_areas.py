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
import numpy as np

import pytest
from numpy.testing import assert_allclose
from fastoad.io import VariableIO
from fastoad.models.aerodynamics.constants import HT_POINT_COUNT, ELEV_POINT_COUNT

from tests.testing_utilities import run_system
from ..tail_sizing.compute_ht_area import ComputeHTArea
from ..tail_sizing.compute_vt_area import ComputeVTArea

def get_indep_var_comp(var_names):
    """ Reads required input data and returns an IndepVarcomp() instance"""
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", "hq_inputs.xml"))
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


def test_compute_vt_area():
    """ Tests computation of the vertical tail area """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeVTArea(), promotes=["*"])
    input_list = list_inputs(group)
    # print(input_list)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:weight:aircraft:CG:aft:MAC_position", 0.364924)
    ivc.add_output("data:aerodynamics:fuselage:cruise:CnBeta", -0.117901)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTArea(), ivc)
    vt_area = problem["data:geometry:vertical_tail:area"]
    assert vt_area == pytest.approx(7275, abs=1)


def test_compute_ht_area():
    """ Tests computation of the horizontal tail area """


    input_list = [
        "data:aerodynamics:aircraft:landing:CL_max",
        "data:aerodynamics:aircraft:low_speed:CL0_clean",
        "data:aerodynamics:aircraft:low_speed:CL_alpha",
        "data:aerodynamics:aircraft:takeoff:CL_max",
        "data:aerodynamics:flaps:landing:CL",
        "data:aerodynamics:flaps:takeoff:CL",
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha",
        "data:geometry:propulsion:engine:count",
        "data:geometry:propulsion:engine:z:from_aeroCenter",
        "data:geometry:propulsion:engine:z:from_wingMAC25",
        "data:geometry:wing:MAC:at25percent:x",
        "data:geometry:wing:MAC:length",
        "data:geometry:wing:area",
        "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
        "data:mission:sizing:landing:elevator_angle",
        "data:mission:sizing:landing:thrust_rate",
        "data:mission:sizing:takeoff:elevator_angle",
        "data:mission:sizing:takeoff:thrust_rate",
        "data:propulsion:MTO_thrust",
        "data:weight:aircraft:MTOW",
        "data:weight:aircraft:MLW",
        "data:weight:aircraft:CG:aft:x",
        "data:weight:airframe:landing_gear:main:CG:x",
        "settings:weight:aircraft:CG:range",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:alpha", np.linspace(0.0, 15.0, HT_POINT_COUNT))
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:CL", np.linspace(0.0, 15.0, HT_POINT_COUNT) * 0.05 + 0.005)
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:CM", np.linspace(0.0, 15.0, HT_POINT_COUNT) * 0.001 + 0.05)
    ivc.add_output("data:aerodynamics:elevator:low_speed:angle", np.linspace(-25.0, 25.0, ELEV_POINT_COUNT))
    ivc.add_output("data:aerodynamics:elevator:low_speed:CL", np.linspace(-25.0, 25.0, ELEV_POINT_COUNT) * -0.02 - 0.001)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTArea(), ivc)
    assert_allclose(problem["data:geometry:horizontal_tail:area"], 2.40, atol=10)
