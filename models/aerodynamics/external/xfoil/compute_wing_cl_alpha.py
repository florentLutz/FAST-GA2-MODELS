"""
    Estimation of wing lift coefficient using VLM-XFOIL
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
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import math
from fastoad.utils.physics import Atmosphere
from fastoad.models.aerodynamics.external.xfoil import xfoil699, VLM
from fastoad.utils.resource_management.copy import copy_resource
from importlib_resources import path
from openmdao.utils.file_wrap import InputFileGenerator

from . import resources

OPTION_RESULT_POLAR_FILENAME = "result_polar_filename"
OPTION_RESULT_FOLDER_PATH = "result_folder_path"
OPTION_PROFILE_NAME = "profile_name"
OPTION_XFOIL_EXE_PATH = "xfoil_exe_path"
OPTION_ALPHA_START = "alpha_start"
OPTION_ALPHA_END = "alpha_end"
OPTION_ITER_LIMIT = "iter_limit"
DEFAULT_ALPHA = 0.0

_INPUT_FILE_NAME = "polar_session.txt"
_INPUT_AOAList = [2.0, 7.0] # ???: why such angles choosen ?
_STDOUT_FILE_NAME = "polar_calc.log"
_STDERR_FILE_NAME = "polar_calc.err"
_TMP_RESULT_FILE_NAME = "out"  # as short as possible to avoid problems of path length
XFOIL_EXE_NAME = "xfoil.exe"  # name of embedded XFoil executable
DEFAULT_PROFILE_FILENAME = "naca23012.txt"

_LOGGER = logging.getLogger(__name__)

_XFOIL_PATH_LIMIT = 64

class ComputeWINGCLALPHAvlm(VLM):
    """
    Runs a polar computation with XFOIL and returns the drag coefficient using VLM
    """
    
    _xfoil_output_names = ["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]
    """Column names in XFOIL polar result"""

    def initialize(self):
        
        self.options.declare(OPTION_XFOIL_EXE_PATH, default="", types=str, allow_none=True)
        self.options.declare(OPTION_PROFILE_NAME, default=DEFAULT_PROFILE_FILENAME, types=str)
        self.options.declare(OPTION_RESULT_FOLDER_PATH, default="", types=str)
        self.options.declare(OPTION_RESULT_POLAR_FILENAME, default="polar_result.txt", types=str)
        self.options.declare(OPTION_ALPHA_START, default=0.0, types=float)
        self.options.declare(OPTION_ALPHA_END, default=30.0, types=float)
        self.options.declare(OPTION_ITER_LIMIT, default=500, types=int)
        
    def setup(self):
        
        super().setup()
        
        self.add_input("xfoil:altitude", val=np.nan, units='ft')
        self.add_input("xfoil:mach", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units='m')
        self.add_input("data:geometry:wing:span", val=np.nan, units='m')
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units='m')
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:area", val=np.nan, units='m**2')
        
        self.add_output("vlm:coef_k")
        
        self.declare_partials("*", "*", method="fd")        
    
    def compute(self, inputs, outputs):
        
        # Create result folder first (if it must fail, let it fail as soon as possible)
        result_folder_path = self.options[OPTION_RESULT_FOLDER_PATH]
        if result_folder_path != "":
            os.makedirs(result_folder_path, exist_ok=True)

        # Get inputs
        mach = inputs["xfoil:mach"]
        altitude = inputs["xfoil:altitude"]
        b_f = inputs['data:geometry:fuselage:maximum_width']
        span = inputs['data:geometry:wing:span']
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        atm = Atmosphere(altitude)
        speed_of_sound = atm.speed_of_sound
        viscosity = atm.kinematic_viscosity
        V_inf = min(speed_of_sound * mach, 0.1) # avoid V=0 m/s crashes
        reynolds = V_inf * l0_wing / viscosity
        
        # Launch VLM internal functions ------------------------------------------------------------
        super()._run()
        Cl, Cdi, Oswald, Cm = super().compute_wing(self, inputs, _INPUT_AOAList, V_inf, flaps_angle=0.0, use_airfoil=True)

        # Pre-processing (populating temp directory) -----------------------------------------------
        # XFoil exe
        tmp_directory = self._create_tmp_directory()
        if self.options[OPTION_XFOIL_EXE_PATH]:
            # if a path for Xfoil has been provided, simply use it
            self.options["command"] = [self.options[OPTION_XFOIL_EXE_PATH]]
        else:
            # otherwise, copy the embedded resource in tmp dir
            copy_resource(xfoil699, XFOIL_EXE_NAME, DEFAULT_PROFILE_FILENAME, tmp_directory.name)
            self.options["command"] = [pth.join(tmp_directory.name, XFOIL_EXE_NAME)]

        # I/O files
        self.stdin = pth.join(tmp_directory.name, _INPUT_FILE_NAME)
        self.stdout = pth.join(tmp_directory.name, _STDOUT_FILE_NAME)
        self.stderr = pth.join(tmp_directory.name, _STDERR_FILE_NAME)

        # profile file
        tmp_profile_file_path = pth.join(tmp_directory.name, DEFAULT_PROFILE_FILENAME)

        # standard input file
        tmp_result_file_path = pth.join(tmp_directory.name, _TMP_RESULT_FILE_NAME)
        parser = InputFileGenerator()
        with path(resources, _INPUT_FILE_NAME) as input_template_path:
            parser.set_template_file(input_template_path)
            parser.set_generated_file(self.stdin)
            parser.mark_anchor("LOAD")
            parser.transfer_var(tmp_profile_file_path, 1, 1)
            parser.mark_anchor("RE")
            parser.transfer_var(float(reynolds), 1, 1)
            parser.mark_anchor("M")
            parser.transfer_var(float(mach), 1, 1)
            parser.mark_anchor("ITER")
            parser.transfer_var(self.options[OPTION_ITER_LIMIT], 1, 1)
            parser.mark_anchor("PACC")
            parser.transfer_var(tmp_result_file_path, 1, 1)
            parser.mark_anchor("ASEQ")
            parser.transfer_var(self.options[OPTION_ALPHA_START], 1, 1)
            parser.transfer_var(self.options[OPTION_ALPHA_END], 2, 1)
            parser.generate()

        # Run XFOIL --------------------------------------------------------------------------------
        self.options["external_input_files"] = [self.stdin, tmp_profile_file_path]
        self.options["external_output_files"] = [tmp_result_file_path]
        super().compute(inputs, outputs)

        # Post-processing --------------------------------------------------------------------------
        result_array = self._read_polar(tmp_result_file_path)
        alpha_0 = self._interpolate_alpha(result_array["CL"], result_array["alpha"], Cl[0])
        
        # Ends the VLM calculation -----------------------------------------------------------------
        k_fus = 1 + 0.025*b_f/span - 0.025*(b_f/span)**2 # Fuselage correction
        beta = math.sqrt(1 - mach**2) # Prandtl-Glauert
        cl_alpha = (Cl[1] - Cl[0]) / ((_INPUT_AOAList[1]-_INPUT_AOAList[0])*math.pi/180) * k_fus / beta
        cl_0 = -alpha_0 * cl_alpha

        outputs['vlm:cl_0_clean'] = cl_0
        outputs['vlm:cl_alpha'] = cl_alpha

    @staticmethod
    def _read_polar(xfoil_result_file_path: str) -> np.ndarray:
        """
        :param xfoil_result_file_path:
        :return: numpy array with XFoil polar results
        """
        if os.path.isfile(xfoil_result_file_path):
            dtypes = [(name, "f8") for name in ComputeWINGCLALPHAvlm._xfoil_output_names]
            result_array = np.genfromtxt(xfoil_result_file_path, skip_header=12, dtype=dtypes)
            return result_array

        _LOGGER.error("XFOIL results file not found")
        return np.array([])
    
    @staticmethod
    def _interpolate_alpha(lift_coeff: np.ndarray, alpha: np.ndarray, ojective:float) -> float:
        """

        :param drag_coeff: CDp array
        :param lift_coeff: CL array
        :param ojective: CL objective value
        :return: alpha if if CL ref encountered, or default value otherwise
        """
        if len(alpha) > 0 and max(alpha) >= 5.0:
            idx_max = np.where(lift_coeff == max(lift_coeff))
            return alpha[idx_max]

        _LOGGER.warning("CL_0 not found. Using default value (%s) for alpha_0", DEFAULT_ALPHA)
        return DEFAULT_ALPHA
    
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
            tmp_profile_file_path = pth.join(tmp_directory.name, DEFAULT_PROFILE_FILENAME)
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