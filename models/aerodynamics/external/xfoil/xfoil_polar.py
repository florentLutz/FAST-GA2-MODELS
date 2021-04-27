"""
This module launches XFOIL computations
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
import os
import os.path as pth
import shutil
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
import warnings
import pandas as pd
from typing import Tuple

import numpy as np
from fastoad.models.aerodynamics.external.xfoil import xfoil699
from fastoad.utils.resource_management.copy import copy_resource
from importlib_resources import path
from openmdao.components.external_code_comp import ExternalCodeComp
from openmdao.utils.file_wrap import InputFileGenerator
from ...constants import POLAR_POINT_COUNT
from ....geometry.profiles.get_profile import get_profile

from . import resources as local_resources

OPTION_RESULT_POLAR_FILENAME = "result_polar_filename"
OPTION_RESULT_FOLDER_PATH = "result_folder_path"
OPTION_XFOIL_EXE_PATH = "xfoil_exe_path"
OPTION_ALPHA_START = "alpha_start"
OPTION_ALPHA_END = "alpha_end"
OPTION_ITER_LIMIT = "iter_limit"
DEFAULT_2D_CL_MAX = 1.9
DEFAULT_2D_CL_MIN = -1.7
ALPHA_STEP = 0.5

_INPUT_FILE_NAME = "polar_session.txt"
_STDOUT_FILE_NAME = "polar_calc.log"
_STDERR_FILE_NAME = "polar_calc.err"
_DEFAULT_AIRFOIL_FILE = "naca23012.af"
_TMP_PROFILE_FILE_NAME = "in"  # as short as possible to avoid problems of path length
_TMP_RESULT_FILE_NAME = "out"  # as short as possible to avoid problems of path length
XFOIL_EXE_NAME = "xfoil.exe"  # name of embedded XFoil executable

_LOGGER = logging.getLogger(__name__)

_XFOIL_PATH_LIMIT = 64


class XfoilPolar(ExternalCodeComp):
    """
    Runs a polar computation with XFOIL and returns the 2D max lift coefficient
    """

    _xfoil_output_names = ["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]
    """Column names in XFOIL polar result"""

    def initialize(self):

        self.options.declare(OPTION_XFOIL_EXE_PATH, default="", types=str, allow_none=True)
        self.options.declare("airfoil_file", default=_DEFAULT_AIRFOIL_FILE, types=str)
        self.options.declare(OPTION_RESULT_FOLDER_PATH, default="", types=str)
        self.options.declare(OPTION_RESULT_POLAR_FILENAME, default="polar_result.txt", types=str)
        self.options.declare(OPTION_ALPHA_START, default=0.0, types=float)
        self.options.declare(OPTION_ALPHA_END, default=20.0, types=float)
        self.options.declare(OPTION_ITER_LIMIT, default=100, types=int)
        self.options.declare('symmetrical', default=False, types=bool)

    def setup(self):

        self.add_input("xfoil:mach", val=np.nan)
        self.add_input("xfoil:reynolds", val=np.nan)
        self.add_output("xfoil:alpha", shape=POLAR_POINT_COUNT, units="deg")
        self.add_output("xfoil:CL", shape=POLAR_POINT_COUNT)
        self.add_output("xfoil:CD", shape=POLAR_POINT_COUNT)
        self.add_output("xfoil:CDp", shape=POLAR_POINT_COUNT)
        self.add_output("xfoil:CM", shape=POLAR_POINT_COUNT)
        self.add_output("xfoil:CL_max_2D")
        self.add_output("xfoil:CL_min_2D")

        self.declare_partials("*", "*", method="fd")

    def check_config(self, logger):
        # let void to avoid logger error on "The command cannot be empty"
        pass

    def compute(self, inputs, outputs):

        # Get inputs and initialise outputs
        mach = round(float(inputs["xfoil:mach"]) * 1e4) / 1e4
        reynolds = round(float(inputs["xfoil:reynolds"]))

        # Search if data already stored for this profile and mach with reynolds values bounding current value.
        # If so, use linear interpolation with the nearest upper/lower reynolds
        no_file = True
        data_saved = None
        interpolated_result = None
        if self.options['symmetrical']:
            result_file = pth.join(pth.split(os.path.realpath(__file__))[0], "resources",
                                   self.options["airfoil_file"].replace('.af', '_sym') + '.csv')
        else:
            result_file = pth.join(pth.split(os.path.realpath(__file__))[0], "resources",
                                   self.options["airfoil_file"].replace('.af', '') + '.csv')
        if pth.exists(result_file):
            no_file = False
            data_saved = pd.read_csv(result_file)
            values = data_saved.to_numpy()[:, 1:len(data_saved.to_numpy()[0])]
            labels = data_saved.to_numpy()[:, 0].tolist()
            data_saved = pd.DataFrame(values, index=labels)
            index_mach = np.where(data_saved.loc["mach", :].to_numpy() == str(mach))[0]
            data_reduced = data_saved.loc[labels, index_mach]
            # Search if this exact reynolds has been computed and save results
            reynolds_vect = np.array([float(x) for x in list(data_reduced.loc["reynolds", :].to_numpy())])
            index_reynolds = index_mach[np.where(reynolds_vect == reynolds)[0]]
            if len(index_reynolds) == 1:
                interpolated_result = data_reduced.loc[labels, index_reynolds]
            # Else search for lower/upper Reynolds
            else:
                lower_reynolds = reynolds_vect[np.where(reynolds_vect < reynolds)[0]]
                upper_reynolds = reynolds_vect[np.where(reynolds_vect > reynolds)[0]]
                if not (len(lower_reynolds) == 0 or len(upper_reynolds) == 0):
                    index_lower_reynolds = index_mach[np.where(reynolds_vect == max(lower_reynolds))[0]]
                    index_upper_reynolds = index_mach[np.where(reynolds_vect == min(upper_reynolds))[0]]
                    lower_values = data_reduced.loc[labels, index_lower_reynolds]
                    upper_values = data_reduced.loc[labels, index_upper_reynolds]
                    interpolated_result = lower_values
                    # Calculate reynolds interval ratio
                    x_ratio = (min(upper_reynolds) - reynolds) / (min(upper_reynolds) - max(lower_reynolds))
                    # Search for common alpha range
                    alpha_lower = eval(lower_values.loc['alpha', index_lower_reynolds].to_numpy()[0])
                    alpha_upper = eval(upper_values.loc['alpha', index_upper_reynolds].to_numpy()[0])
                    alpha_shared = np.array(list(set(alpha_upper).intersection(alpha_lower)))
                    interpolated_result.loc['alpha', index_lower_reynolds] = str(alpha_shared.tolist())
                    labels.remove('alpha')
                    # Calculate average values (cd, cl...) with linear interpolation
                    for label in labels:
                        lower_value = np.array(eval(lower_values.loc[label, index_lower_reynolds].to_numpy()[0]))
                        upper_value = np.array(eval(upper_values.loc[label, index_upper_reynolds].to_numpy()[0]))
                        # If values relative to alpha vector, performs interpolation with shared vector
                        if np.size(lower_value) == len(alpha_lower):
                            lower_value = np.interp(alpha_shared, np.array(alpha_lower), lower_value)
                            upper_value = np.interp(alpha_shared, np.array(alpha_upper), upper_value)
                        value = (lower_value * x_ratio + upper_value * (1 - x_ratio)).tolist()
                        interpolated_result.loc[label, index_lower_reynolds] = str(value)

        if interpolated_result is None:
            # Create result folder first (if it must fail, let it fail as soon as possible)
            result_folder_path = self.options[OPTION_RESULT_FOLDER_PATH]
            if result_folder_path != "":
                os.makedirs(result_folder_path, exist_ok=True)

            # Pre-processing (populating temp directory) -----------------------------------------------
            # XFoil exe
            tmp_directory = self._create_tmp_directory()
            if self.options[OPTION_XFOIL_EXE_PATH]:
                # if a path for Xfoil has been provided, simply use it
                self.options["command"] = [self.options[OPTION_XFOIL_EXE_PATH]]
            else:
                # otherwise, copy the embedded resource in tmp dir
                # noinspection PyTypeChecker
                copy_resource(xfoil699, XFOIL_EXE_NAME, tmp_directory.name)
                self.options["command"] = [pth.join(tmp_directory.name, XFOIL_EXE_NAME)]

            # I/O files
            self.stdin = pth.join(tmp_directory.name, _INPUT_FILE_NAME)
            self.stdout = pth.join(tmp_directory.name, _STDOUT_FILE_NAME)
            self.stderr = pth.join(tmp_directory.name, _STDERR_FILE_NAME)

            # profile file
            tmp_profile_file_path = pth.join(tmp_directory.name, _TMP_PROFILE_FILE_NAME)
            profile = get_profile(
                file_name=self.options["airfoil_file"],
            ).get_sides()
            # noinspection PyTypeChecker
            np.savetxt(
                tmp_profile_file_path,
                profile.to_numpy(),
                fmt="%.15f",
                delimiter=" ",
                header="Wing",
                comments="",
            )

            # standard input file
            tmp_result_file_path = pth.join(tmp_directory.name, _TMP_RESULT_FILE_NAME)
            self._write_script_file(
                reynolds, mach, tmp_profile_file_path, tmp_result_file_path, self.options[OPTION_ALPHA_START],
                self.options[OPTION_ALPHA_END], ALPHA_STEP)

            # Run XFOIL --------------------------------------------------------------------------------
            self.options["external_input_files"] = [self.stdin, tmp_profile_file_path]
            self.options["external_output_files"] = [tmp_result_file_path]
            super().compute(inputs, outputs)

            if self.options['symmetrical']:
                result_array_p = self._read_polar(tmp_result_file_path)
                os.remove(self.stdin)
                os.remove(self.stdout)
                os.remove(self.stderr)
                os.remove(tmp_result_file_path)
                self._write_script_file(
                    reynolds, mach, tmp_profile_file_path, tmp_result_file_path, -1 * self.options[OPTION_ALPHA_START],
                                                                                 -1 * self.options[OPTION_ALPHA_END],
                    -ALPHA_STEP)
                super().compute(inputs, outputs)
                result_array_n = self._read_polar(tmp_result_file_path)
            else:
                result_array_p = self._read_polar(tmp_result_file_path)
                result_array_n = result_array_p

            # Post-processing --------------------------------------------------------------------------
            cl_max_2d, error = self._get_max_cl(result_array_p["alpha"], result_array_p["CL"])
            cl_min_2d, error = self._get_min_cl(result_array_n["alpha"], result_array_n["CL"])
            if POLAR_POINT_COUNT < len(result_array_p["alpha"]):
                alpha = np.linspace(result_array_p["alpha"][0], result_array_p["alpha"][-1], POLAR_POINT_COUNT)
                cl = np.interp(alpha, result_array_p["alpha"], result_array_p["CL"])
                cd = np.interp(alpha, result_array_p["alpha"], result_array_p["CD"])
                cdp = np.interp(alpha, result_array_p["alpha"], result_array_p["CDp"])
                cm = np.interp(alpha, result_array_p["alpha"], result_array_p["CM"])
                warnings.warn("Defined polar point in fast aerodynamics\\constants.py exceeded!")
            else:
                additional_zeros = list(np.zeros(POLAR_POINT_COUNT - len(result_array_p["alpha"])))
                alpha = result_array_p["alpha"].tolist()
                alpha.extend(additional_zeros)
                alpha = np.asarray(alpha)
                cl = result_array_p["CL"].tolist()
                cl.extend(additional_zeros)
                cl = np.asarray(cl)
                cd = result_array_p["CD"].tolist()
                cd.extend(additional_zeros)
                cd = np.asarray(cd)
                cdp = result_array_p["CDp"].tolist()
                cdp.extend(additional_zeros)
                cdp = np.asarray(cdp)
                cm = result_array_p["CM"].tolist()
                cm.extend(additional_zeros)
                cm = np.asarray(cm)

            # Save results to defined path -------------------------------------------------------------
            if not error:
                results = [np.array(mach), np.array(reynolds), np.array(cl_max_2d), np.array(cl_min_2d),
                           str(self._reshape(alpha, alpha).tolist()), str(self._reshape(alpha, cl).tolist()),
                           str(self._reshape(alpha, cd).tolist()), str(self._reshape(alpha, cdp).tolist()),
                           str(self._reshape(alpha, cm).tolist())]
                labels = ["mach", "reynolds", "cl_max_2d", "cl_min_2d", "alpha", "cl", "cd", "cdp", "cm"]
                if no_file or (data_saved is None):
                    data = pd.DataFrame(results, index=labels)
                    data.to_csv(result_file)
                else:
                    data = pd.DataFrame(np.c_[data_saved, results], index=labels)
                    data.to_csv(result_file)

            # Getting output files if needed ---------------------------------------------------------
            if self.options[OPTION_RESULT_FOLDER_PATH] != "":
                if pth.exists(tmp_result_file_path):
                    polar_file_path = pth.join(
                        result_folder_path, self.options[OPTION_RESULT_POLAR_FILENAME]
                    )
                    shutil.move(tmp_result_file_path, polar_file_path)

                if pth.exists(self.stdout):
                    stdout_file_path = pth.join(result_folder_path, _STDOUT_FILE_NAME)
                    shutil.move(self.stdout, stdout_file_path)

                if pth.exists(self.stderr):
                    stderr_file_path = pth.join(result_folder_path, _STDERR_FILE_NAME)
                    shutil.move(self.stderr, stderr_file_path)

            tmp_directory.cleanup()

        else:
            # Extract results
            cl_max_2d = np.array(eval(interpolated_result.loc["cl_max_2d", :].to_numpy()[0]))
            cl_min_2d = np.array(eval(interpolated_result.loc["cl_min_2d", :].to_numpy()[0]))
            ALPHA = np.array(eval(interpolated_result.loc["alpha", :].to_numpy()[0]))
            CL = np.array(eval(interpolated_result.loc["cl", :].to_numpy()[0]))
            CD = np.array(eval(interpolated_result.loc["cd", :].to_numpy()[0]))
            CDP = np.array(eval(interpolated_result.loc["cdp", :].to_numpy()[0]))
            CM = np.array(eval(interpolated_result.loc["cm", :].to_numpy()[0]))

            # Modify vector length if necessary
            if POLAR_POINT_COUNT < len(ALPHA):
                alpha = np.linspace(ALPHA[0], ALPHA[-1], POLAR_POINT_COUNT)
                cl = np.interp(alpha, ALPHA, CL)
                cd = np.interp(alpha, ALPHA, CD)
                cdp = np.interp(alpha, ALPHA, CDP)
                cm = np.interp(alpha, ALPHA, CM)
            else:
                additional_zeros = list(np.zeros(POLAR_POINT_COUNT - len(ALPHA)))
                alpha = ALPHA.tolist()
                alpha.extend(additional_zeros)
                # noinspection PyTypeChecker
                alpha = np.asarray(alpha)
                cl = CL.tolist()
                cl.extend(additional_zeros)
                # noinspection PyTypeChecker
                cl = np.asarray(cl)
                cd = CD.tolist()
                cd.extend(additional_zeros)
                # noinspection PyTypeChecker
                cd = np.asarray(cd)
                cdp = CDP.tolist()
                cdp.extend(additional_zeros)
                # noinspection PyTypeChecker
                cdp = np.asarray(cdp)
                cm = CM.tolist()
                cm.extend(additional_zeros)
                # noinspection PyTypeChecker
                cm = np.asarray(cm)

        # Defining outputs -------------------------------------------------------------------------
        outputs["xfoil:alpha"] = alpha
        outputs["xfoil:CL"] = cl
        outputs["xfoil:CD"] = cd
        outputs["xfoil:CDp"] = cdp
        outputs["xfoil:CM"] = cm
        outputs["xfoil:CL_max_2D"] = cl_max_2d
        outputs["xfoil:CL_min_2D"] = cl_min_2d

    def _write_script_file(self, reynolds, mach, tmp_profile_file_path, tmp_result_file_path, alpha_start,
                           alpha_end, step):
        parser = InputFileGenerator()
        with path(local_resources, _INPUT_FILE_NAME) as input_template_path:
            parser.set_template_file(str(input_template_path))
            parser.set_generated_file(self.stdin)
            parser.mark_anchor("RE")
            parser.transfer_var(float(reynolds), 1, 1)
            parser.mark_anchor("M")
            parser.transfer_var(float(mach), 1, 1)
            parser.mark_anchor("ITER")
            parser.transfer_var(self.options[OPTION_ITER_LIMIT], 1, 1)
            parser.mark_anchor("ASEQ")
            parser.transfer_var(alpha_start, 1, 1)
            parser.transfer_var(alpha_end, 2, 1)
            parser.transfer_var(step, 3, 1)
            parser.reset_anchor()
            parser.mark_anchor("/profile")
            parser.transfer_var(tmp_profile_file_path, 0, 1)
            parser.mark_anchor("/polar_result")
            parser.transfer_var(tmp_result_file_path, 0, 1)
            parser.generate()

    @staticmethod
    def _read_polar(xfoil_result_file_path: str) -> np.ndarray:
        """
        :param xfoil_result_file_path:
        :return: numpy array with XFoil polar results
        """
        if os.path.isfile(xfoil_result_file_path):
            dtypes = [(name, "f8") for name in XfoilPolar._xfoil_output_names]
            result_array = np.genfromtxt(xfoil_result_file_path, skip_header=12, dtype=dtypes)
            return result_array

        _LOGGER.error("XFOIL results file not found")
        return np.array([])

    def _get_max_cl(self, alpha: np.ndarray, lift_coeff: np.ndarray) -> Tuple[float, bool]:
        """

        :param alpha:
        :param lift_coeff: CL
        :return: max CL within 90/110% linear zone if enough alpha computed, or default value otherwise
        """
        alpha_range = self.options[OPTION_ALPHA_END] - self.options[OPTION_ALPHA_START]
        if len(alpha) > 2:
            covered_range = max(alpha) - min(alpha)
            if np.abs(covered_range / alpha_range) >= 0.5:
                lift_fct = lambda x: (lift_coeff[1] - lift_coeff[0]) / (alpha[1] - alpha[0]) * (x - alpha[0]) \
                                     + lift_coeff[0]
                delta = np.abs((lift_coeff - lift_fct(alpha)) / (lift_coeff + 1e-12 * (lift_coeff == 0.0)))
                return max(lift_coeff[delta <= 0.1]), False

        _LOGGER.warning("2D CL max not found, les than 50% of angle range computed: using default value {}".format(
            DEFAULT_2D_CL_MAX))
        return DEFAULT_2D_CL_MAX, True

    def _get_min_cl(self, alpha: np.ndarray, lift_coeff: np.ndarray) -> Tuple[float, bool]:
        """

        :param alpha:
        :param lift_coeff: CL
        :return: min CL within 90/110% linear zone if enough alpha computed, or default value otherwise
        """
        alpha_range = self.options[OPTION_ALPHA_END] - self.options[OPTION_ALPHA_START]
        if len(alpha) > 2:
            covered_range = max(alpha) - min(alpha)
            if covered_range / alpha_range >= 0.5:
                lift_fct = lambda x: (lift_coeff[1] - lift_coeff[0]) / (alpha[1] - alpha[0]) * (x - alpha[0]) \
                                     + lift_coeff[0]
                delta = np.abs(lift_coeff - lift_fct(alpha)) / np.abs(lift_coeff + 1e-12 * (lift_coeff == 0.0))
                return min(lift_coeff[delta <= 0.1]), False

        _LOGGER.warning("2D CL min not found, les than 50% of angle range computed: using default value {}".format(
            DEFAULT_2D_CL_MIN))
        return DEFAULT_2D_CL_MIN, True

    @staticmethod
    def _reshape(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ Delete ending 0.0 values """
        for idx in range(len(x)):
            if np.sum(x[idx:len(x)] == 0.0) == (len(x) - idx):
                y = y[0:idx]
                break
        return y

    @staticmethod
    def _create_tmp_directory() -> TemporaryDirectory:
        # Dev Note: XFOIL fails if length of provided file path exceeds 64 characters.
        #           Changing working directory to the tmp dir would allow to just provide file name,
        #           but it is not really safe (at least, it does mess with the coverage report).
        #           Then the point is to get a tmp directory with a short path.
        #           On Windows, the default (user-dependent) tmp dir can exceed the limit.
        #           Therefore, as a second choice, tmp dir is created as close of user home
        #           directory as possible.
        tmp_candidates = []
        for tmp_base_path in [None, pth.join(str(Path.home()), ".fast")]:
            if tmp_base_path is not None:
                os.makedirs(tmp_base_path, exist_ok=True)
            tmp_directory = tempfile.TemporaryDirectory(prefix="x", dir=tmp_base_path)
            tmp_candidates.append(tmp_directory.name)
            tmp_profile_file_path = pth.join(tmp_directory.name, _TMP_PROFILE_FILE_NAME)
            tmp_result_file_path = pth.join(tmp_directory.name, _TMP_RESULT_FILE_NAME)

            if max(len(tmp_profile_file_path), len(tmp_result_file_path)) <= _XFOIL_PATH_LIMIT:
                # tmp_directory is OK. Stop there
                break
            # tmp_directory has a too long path. Erase and continue...
            tmp_directory.cleanup()

        if max(len(tmp_profile_file_path), len(tmp_result_file_path)) > _XFOIL_PATH_LIMIT:
            raise IOError(
                "Could not create a tmp directory where file path will respects XFOIL "
                "limitation (%i): tried %s" % (_XFOIL_PATH_LIMIT, tmp_candidates)
            )

        return tmp_directory
