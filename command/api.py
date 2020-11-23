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
from importlib_resources import path

from fastoad.cmd.exceptions import FastFileExistsError
from openmdao.utils.file_wrap import InputFileGenerator
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
    package_path = "[\"" + root_folder.replace('\\','\\\\') + "\"]"
    with path(resources, SAMPLE_FILENAME) as input_template_path:
        parser.set_template_file(str(input_template_path))
        # noinspection PyTypeChecker
        parser.set_generated_file(configuration_file_path)
        parser.reset_anchor()
        parser.mark_anchor("module_folders")
        parser.transfer_var(package_path, 0, 3)
        parser.generate()

    _LOGGER.info("Sample configuration written in %s", configuration_file_path)

