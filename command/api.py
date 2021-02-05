"""
API
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
import warnings
from importlib_resources import path
from typing import Union, List
from openmdao.core.explicitcomponent import ExplicitComponent
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.core.group import Group
from openmdao.utils.file_wrap import InputFileGenerator
import openmdao.api as om
from fastoad.module_management import OpenMDAOSystemRegistry

from fastoad.openmdao.variables import VariableList
from fastoad.cmd.exceptions import FastFileExistsError
from fastoad.openmdao.problem import FASTOADProblem
from fastoad.io.xml import VariableXmlStandardFormatter
from fastoad.io import VariableIO

from . import resources

_LOGGER = logging.getLogger(__name__)

SAMPLE_FILENAME = "fastga.toml"


def generate_configuration_file(configuration_file_path: str, overwrite: bool = False):
    """
    Generates a sample configuration file.

    :param configuration_file_path: the path of file to be written
    :param overwrite: if True, the file will be written, even if it already exists
    :raise FastFileExistsError: if overwrite==False and configuration_file_path already exists
    """
    if not overwrite and pth.exists(configuration_file_path):
        raise FastFileExistsError(
            "Configuration file %s not written because it already exists. "
            "Use overwrite=True to bypass." % configuration_file_path,
            configuration_file_path,
        )

    if not pth.exists(pth.split(configuration_file_path)[0]):
        os.mkdir(pth.split(configuration_file_path)[0])
    parser = InputFileGenerator()
    root_folder = resources.__path__[0]
    for i in range(2):
        root_folder = pth.split(root_folder)[0]
    package_path = "[\"" + root_folder.replace('\\', '/') + "\"]"
    with path(resources, SAMPLE_FILENAME) as input_template_path:
        parser.set_template_file(str(input_template_path))
        # noinspection PyTypeChecker
        parser.set_generated_file(configuration_file_path)
        parser.reset_anchor()
        parser.mark_anchor("module_folders")
        parser.transfer_var(package_path, 0, 3)
        parser.generate()

    _LOGGER.info("Sample configuration written in %s", configuration_file_path)


def generate_block_analysis(
        system: Union[ExplicitComponent, ImplicitComponent, Group],
        var_inputs: List,
        xml_file_path: str,
        overwrite: bool = False,
):
    # List openmdao component/group inputs
    all_inputs = list_inputs(system)

    # Search what are the component/group outputs
    variables = VariableList.from_system(system)
    outputs_names = [var.name for var in variables if not var.is_input]

    # Check that variable inputs are in the group/component list
    if not(set(var_inputs) == set(all_inputs).intersection(set(var_inputs))):
        raise Exception('The input list contains name(s) out of component/group input list!')

    # Perform some tests on the .xml availability and completeness
    if not(os.path.exists(xml_file_path)) and not(var_inputs.sort() == all_inputs.sort()):
        # If no input file and some inputs are missing, generate it and return None
        if isinstance(system, Group):
            problem = FASTOADProblem(system)
        else:
            group = Group()
            group.add_subsystem('system', system, promotes=["*"])
            problem = FASTOADProblem(group)
        problem.input_file_path = xml_file_path
        problem.write_needed_inputs(None, VariableXmlStandardFormatter())
        warnings.warn('Input .xml file not found, a default file has been created with default NaN values, '
                      'but no function is returned!\nConsider defining proper values before second execution!')
        return None
    elif os.path.exists(xml_file_path):

        reader = VariableIO(xml_file_path, VariableXmlStandardFormatter()).read(ignore=(var_inputs + outputs_names))
        xml_inputs = reader.names()
        if not(set(xml_inputs + var_inputs).intersection(set(all_inputs)) == set(all_inputs)):
            # If some inputs are missing add them to the problem if authorized
            if overwrite:
                reader.path_separator = ":"
                ivc = reader.to_ivc()
                group = Group()
                group.add_subsystem('system', system, promotes=["*"])
                group.add_subsystem('ivc', ivc, promotes=["*"])
                problem = FASTOADProblem(group)
                problem.output_file_path = xml_file_path
                problem.write_outputs()
                warnings.warn('Some inputs are missing in the given .xml file, they have been added with default NaN, '
                              'but no function is returned!\nConsider defining proper values before second execution!')
                return None
            else:
                # Else raise an error mentioning missing inputs
                missing_inputs = list(
                    set(xml_inputs + var_inputs).intersection(set(all_inputs)).difference(set(all_inputs))
                )
                message = 'Following inputs are missing in .xml file: '
                message += ['[' + item + '], ' for item in list(missing_inputs)]
                raise Exception(message[:-1])
        else:
            # If all inputs addressed either by .xml or var_inputs, construct the function
            def patched_function(inputs_dict: dict) -> dict:
                """
                The patched function perform a run of an openmdao component or group applying FASTOAD formalism.

                @param inputs_dict: dictionary of input (values, units) saved with their key name,
                as an example: inputs_dict = {'in1': (3.0, "m")}.
                @return: dictionary of the component/group outputs saving names as keys and (value, units) as tuple.
                """


                # Read .xml file and construct Independent Variable Component excluding outputs
                reader.path_separator = ":"
                ivc_local = reader.to_ivc()
                for name, value in inputs_dict.items():
                    ivc_local.add_output(name, value[0], units=value[1])
                group_local = Group()
                group_local.add_subsystem('system', system, promotes=["*"])
                group_local.add_subsystem('ivc', ivc_local, promotes=["*"])
                problem_local = FASTOADProblem(group_local)
                problem_local.setup()
                problem_local.run_model()
                if overwrite:
                    problem_local.output_file_path = xml_file_path
                    problem_local.write_outputs()
                # Get output names from component/group and construct dictionary
                outputs_units = [var.units for var in variables if not var.is_input]
                outputs_dict = {}
                for idx in range(len(outputs_names)):
                    value = problem_local.get_val(outputs_names[idx], outputs_units[idx])
                    outputs_dict[outputs_names[idx]] = (value, outputs_units[idx])
                return outputs_dict
            return patched_function


def list_inputs(component: Union[om.ExplicitComponent, om.Group]) -> list:
    """ Reads input variables from a component/problem and return as a list """
    register_wrappers()
    variables = VariableList.from_system(component)
    input_names = [var.name for var in variables if var.is_input]

    return input_names


def register_wrappers():
    """ Register all the wrappers from models """
    path_name, folder_name = pth.split(pth.dirname(__file__))
    path_name = pth.join(path_name, "models")
    OpenMDAOSystemRegistry.explore_folder(path_name)
