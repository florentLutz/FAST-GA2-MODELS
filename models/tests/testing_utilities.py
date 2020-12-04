"""
Convenience functions for helping tests
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

import logging
import os.path as pth
import openmdao.api as om
from typing import Union, List
import time

from fastoad.io import VariableIO
from fastoad.openmdao.types import SystemSubclass
from fastoad.openmdao.utils import get_unconnected_input_names
from fastoad.module_management import OpenMDAOSystemRegistry
from fastoad.openmdao.variables import VariableList

_LOGGER = logging.getLogger(__name__)


def run_system(
    component: SystemSubclass, input_vars: om.IndepVarComp, setup_mode="auto", add_solvers=False, check=False
):
    """ Runs and returns an OpenMDAO problem with provided component and data"""
    problem = om.Problem()
    model = problem.model
    model.add_subsystem("inputs", input_vars, promotes=["*"])
    model.add_subsystem("component", component, promotes=["*"])
    if add_solvers:
        # noinspection PyTypeChecker
        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        model.linear_solver = om.DirectSolver()

    if check:
        print('\n')
    problem.setup(mode=setup_mode, check=check)
    missing, _ = get_unconnected_input_names(problem, _LOGGER)
    assert not missing, "These inputs are not provided: %s" % missing

    problem.run_model()

    return problem


def register_wrappers():
    """ Register all the wrappers from models """
    path, folder_name = pth.dirname(__file__), None
    unsplit_path = path
    while folder_name != "models":
        unsplit_path = path
        path, folder_name = pth.split(path)
    OpenMDAOSystemRegistry.explore_folder(unsplit_path)


def get_indep_var_comp(var_names: List[str], test_file: str, xml_file_name: str) -> om.IndepVarComp:
    """ Reads required input data from xml file and returns an IndepVarcomp() instance"""
    reader = VariableIO(pth.join(pth.dirname(test_file), "data", xml_file_name))
    reader.path_separator = ":"
    ivc = reader.read(only=var_names).to_ivc()

    return ivc


def list_inputs(component: Union[om.ExplicitComponent, om.Group]) -> list:
    """ Reads input variables from a component/problem and return as a list """
    register_wrappers()
    variables = VariableList.from_system(component)
    input_names = [var.name for var in variables if var.is_input]

    return input_names


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print('\n')
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))