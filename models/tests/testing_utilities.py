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
import os
import openmdao.api as om
from typing import Union, List
import time
from openmdao.core.system import System
from copy import deepcopy

from fastoad.io import VariableIO
# noinspection PyProtectedMember
from fastoad.module_management.service_registry import _RegisterOpenMDAOService
from fastoad.openmdao.variables import VariableList
from fastoad.io.configuration.configuration import AutoUnitsDefaultGroup

_LOGGER = logging.getLogger(__name__)


def run_system(
    component: System, input_vars: om.IndepVarComp, setup_mode="auto", add_solvers=False, check=False
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
    variables = VariableList.from_unconnected_inputs(problem)
    assert not variables, "These inputs are not provided: %s" % variables.names()

    problem.run_model()

    return problem


# FIXME: problem to be solved on the register
def register_wrappers():
    """ Register all the wrappers from models """
    path, folder_name = pth.dirname(__file__), None
    unsplit_path = path
    while folder_name != "models":
        unsplit_path = path
        path, folder_name = pth.split(path)
    for directory in os.listdir(unsplit_path):
        if pth.isdir(pth.join(unsplit_path, directory)) and not ('.' in directory):
            # noinspection PyBroadException
            try:
                _RegisterOpenMDAOService.explore_folder(pth.join(unsplit_path, directory))
            except:
                pass


def get_indep_var_comp(var_names: List[str], test_file: str, xml_file_name: str) -> om.IndepVarComp:
    """ Reads required input data from xml file and returns an IndepVarcomp() instance"""
    reader = VariableIO(pth.join(pth.dirname(test_file), "data", xml_file_name))
    reader.path_separator = ":"
    ivc = reader.read(only=var_names).to_ivc()

    return ivc


class VariableListLocal(VariableList):

    @classmethod
    def from_system(cls, system: System) -> "VariableList":
        """
        Creates a VariableList instance containing inputs and outputs of a an OpenMDAO System.
        The inputs (is_input=True) correspond to the variables of IndepVarComp
        components and all the unconnected variables.

        Warning: setup() must NOT have been called.

        In the case of a group, if variables are promoted, the promoted name
        will be used. Otherwise, the absolute name will be used.

        :param system: OpenMDAO Component instance to inspect
        :return: VariableList instance
        """

        problem = om.Problem()
        if isinstance(system, om.Group):
            problem.model = deepcopy(system)
        else:
            # problem.model has to be a group
            problem.model.add_subsystem("comp", deepcopy(system), promotes=["*"])
        problem.setup()
        return VariableListLocal.from_problem(problem, use_initial_values=True)


def list_inputs(component: Union[om.ExplicitComponent, om.Group]) -> list:
    """ Reads input variables from a component/problem and return as a list """
    # register_wrappers()
    if isinstance(component, om.Group):
        new_component = AutoUnitsDefaultGroup()
        new_component.add_subsystem("system", component, promotes=['*'])
        component = new_component
    variables = VariableListLocal.from_system(component)
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
