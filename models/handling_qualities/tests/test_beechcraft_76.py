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
import numpy as np
import openmdao.api as om

import pytest
from fastoad.io import VariableIO
from fastoad.module_management import OpenMDAOSystemRegistry
from typing import Union

from ...tests.testing_utilities import run_system
from ..compute_static_margin import ComputeStaticMargin
from ..tail_sizing.compute_ht_area import ComputeHTArea
from ..tail_sizing.compute_vt_area import ComputeVTArea

XML_FILE = "beechcraft_76.xml"
ENGINE_WRAPPER = "fastga.wrapper.propulsion.basicIC_engine"


def get_indep_var_comp(var_names):
    """ Reads required input data and returns an IndepVarcomp() instance"""
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    ivc = reader.read(only=var_names).to_ivc()
    return ivc


def list_inputs(component: Union[om.ExplicitComponent, om.Group]) -> list:
    """ Reads input variables from a component/problem and return as a list """

    register_wrappers()
    if isinstance(component, om.ExplicitComponent):
        prob = om.Problem(model=component)
        prob.setup()
        data = prob.model.list_inputs(out_stream=None)
        list_names = []
        for idx in range(len(data)):
            variable_name = data[idx][0]
            list_names.append(variable_name)
    else:
        data = []
        component.setup()
        for subcomponent in component._static_subsystems_allprocs:
            subprob = om.Problem(model=subcomponent)
            subprob.setup()
            data.extend(subprob.model.list_inputs(out_stream=None))
        list_names = []
        for idx in range(len(data)):
            variable_name = data[idx][0].split('.')[-1]
            list_names.append(variable_name)

    return list_names


def register_wrappers():
    path_split = pth.dirname(__file__).split('\\')
    drive = path_split[0]
    del path_split[0]
    del path_split[-1]
    del path_split[-1]
    path = drive + "\\" + pth.join(*path_split)
    OpenMDAOSystemRegistry.explore_folder(path)


def test_compute_vt_area():
    """ Tests computation of the vertical tail area """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeVTArea(propulsion_id=ENGINE_WRAPPER)))
    ivc.add_output("data:weight:aircraft:CG:aft:MAC_position", 0.364924)
    ivc.add_output("data:aerodynamics:fuselage:cruise:CnBeta", -0.0599)
    ivc.add_output("data:propulsion:IC_engine:max_power", 130000, units="W")  # correct value to fit old version def.
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(ComputeVTArea(propulsion_id=ENGINE_WRAPPER), ivc)
    vt_area = problem.get_val("data:geometry:vertical_tail:area", units="m**2")
    assert vt_area == pytest.approx(2.44, abs=1e-2)  # old-version obtained value 2.4m²


def test_compute_ht_area():
    """ Tests computation of the horizontal tail area """

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeHTArea(propulsion_id=ENGINE_WRAPPER)))
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:alpha",
                   np.array([0.0, 7.5, 15.0, 22.5, 30.0]), units="deg")
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:CL",
                   np.array([-0.00472, 0.08476, 0.16876, 0.23578, 0.28567]))
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:CM",
                   np.array([0.01308, -0.24393, -0.50001, -0.71332, -0.88161]))
    ivc.add_output("data:aerodynamics:wing:low_speed:alpha",
                   np.array([0.0, 7.5, 15.0, 22.5, 30.0]), units="deg")
    ivc.add_output("data:aerodynamics:wing:low_speed:CM",
                   np.array([-0.01332, 0.02356, 0.10046, 0.20401, 0.31282]))
    ivc.add_output("data:aerodynamics:elevator:low_speed:CL_alpha", 0.6167, units="rad**-1")
    ivc.add_output("data:propulsion:IC_engine:max_power", 130000, units="W")  # correct value to fit old version def.
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    # noinspection PyTypeChecker
    problem = run_system(ComputeHTArea(propulsion_id=ENGINE_WRAPPER), ivc)
    ht_area = problem.get_val("data:geometry:horizontal_tail:area", units="m**2")
    assert ht_area == pytest.approx(5.48, abs=1e-2)  # old-version obtained value 3.9m²


def test_compute_static_margin():
    """ Tests computation of static margin """

    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    input_vars.add_output("data:weight:aircraft:CG:aft:MAC_position", 0.20)

    problem = run_system(ComputeStaticMargin(), input_vars)
    static_margin = problem["data:handling_qualities:static_margin"]
    assert static_margin == pytest.approx(0.55, rel=1e-2)
