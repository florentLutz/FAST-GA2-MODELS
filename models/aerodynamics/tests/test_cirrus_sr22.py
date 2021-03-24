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
import os
import pandas as pd
import openmdao.api as om
from openmdao.core.component import Component
import numpy as np
from platform import system
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
from typing import Union
import time

from fastoad.io import VariableIO
from fastoad.module_management.service_registry import RegisterPropulsion
from fastoad import BundleLoader
from fastoad.base.flight_point import FlightPoint
from fastoad.models.propulsion.propulsion import IOMPropulsionWrapper
from scipy import integrate

from ...tests.testing_utilities import run_system, register_wrappers, get_indep_var_comp, list_inputs
from ..components.cd0 import Cd0
from ..external.xfoil import XfoilPolar
from ..external.xfoil import resources
from ..external.vlm import ComputeAEROvlm, ComputeVNvlmNoVH
from ..external.openvsp import ComputeAEROopenvsp, ComputeVNopenvspNoVH
from ..external.openvsp.compute_aero_slipstream import ComputeSlipstreamOpenvsp
from ..components import ComputeExtremeCL, ComputeUnitReynolds, ComputeCnBetaFuselage, ComputeLDMax, \
    ComputeDeltaHighLift, Compute2DHingeMomentsTail, Compute3DHingeMomentsTail
from ..aerodynamics_high_speed import AerodynamicsHighSpeed
from ..aerodynamics_low_speed import AerodynamicsLowSpeed

from ...tests.xfoil_exe.get_xfoil import get_xfoil_path
from ...propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from ...propulsion.propulsion import IPropulsion

from ..external.vlm.compute_aero import DEFAULT_WING_AIRFOIL, DEFAULT_HTP_AIRFOIL
from ..constants import SPAN_MESH_POINT, POLAR_POINT_COUNT


RESULTS_FOLDER = pth.join(pth.dirname(__file__), "results")
xfoil_path = None if system() == "Windows" else get_xfoil_path()

XML_FILE = "cirrus_sr22.xml"
ENGINE_WRAPPER = "test.wrapper.aerodynamics.beechcraft.dummy_engine"


class DummyEngine(AbstractFuelPropulsion):

    def __init__(self,
                 max_power: float,
                 design_altitude: float,
                 design_speed: float,
                 fuel_type: float,
                 strokes_nb: float,
                 prop_layout: float,
                 ):
        """
        Dummy engine model returning nacelle aerodynamic drag force.

        """
        super().__init__()
        self.prop_layout = prop_layout
        self.max_power = max_power
        self.design_altitude = design_altitude
        self.design_speed = design_speed
        self.fuel_type = fuel_type
        self.strokes_nb = strokes_nb

    def compute_flight_points(self, flight_points: Union[FlightPoint, pd.DataFrame]):
        flight_points.thrust = 1800.0
        flight_points['sfc'] = 0.0

    def compute_weight(self) -> float:
        return 0.0

    def compute_dimensions(self) -> (float, float, float, float, float, float):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def compute_drag(self, mach, unit_reynolds, l0_wing):
        if mach < 0.15:
            return 0.01934377
        else:
            return 0.01771782

    def get_consumed_mass(self, flight_point: FlightPoint, time_step: float) -> float:
        return 0.0


@RegisterPropulsion(ENGINE_WRAPPER)
class DummyEngineWrapper(IOMPropulsionWrapper):
    def setup(self, component: Component):
        component.add_input("data:propulsion:IC_engine:max_power", np.nan, units="W")
        component.add_input("data:propulsion:IC_engine:fuel_type", np.nan)
        component.add_input("data:propulsion:IC_engine:strokes_nb", np.nan)
        component.add_input("data:TLAR:v_cruise", np.nan, units="m/s")
        component.add_input("data:mission:sizing:main_route:cruise:altitude", np.nan, units="m")
        component.add_input("data:geometry:propulsion:layout", np.nan)

    @staticmethod
    def get_model(inputs) -> IPropulsion:
        engine_params = {
            "max_power": inputs["data:propulsion:IC_engine:max_power"],
            "design_altitude": inputs["data:mission:sizing:main_route:cruise:altitude"],
            "design_speed": inputs["data:TLAR:v_cruise"],
            "fuel_type": inputs["data:propulsion:IC_engine:fuel_type"],
            "strokes_nb": inputs["data:propulsion:IC_engine:strokes_nb"],
            "prop_layout": inputs["data:geometry:propulsion:layout"]
        }

        return DummyEngine(**engine_params)


BundleLoader().context.install_bundle(__name__).start()


def _create_tmp_directory() -> TemporaryDirectory:
    """Provide temporary directory for calculation."""

    for tmp_base_path in [None, pth.join(str(Path.home()), ".fast")]:
        if tmp_base_path is not None:
            os.makedirs(tmp_base_path, exist_ok=True)
        tmp_directory = tempfile.TemporaryDirectory(prefix="x", dir=tmp_base_path)
        break

    return tmp_directory


def reshape_curve(y, cl):
    """ Reshape data from openvsp/vlm lift curve """
    for idx in range(len(y)):
        if np.sum(y[idx:len(y)] == 0) == (len(y) - idx):
            y = y[0:idx]
            cl = cl[0:idx]
            break

    return y, cl


def reshape_polar(cl, cdp):
    """ Reshape data from xfoil polar vectors """
    for idx in range(len(cl)):
        if np.sum(cl[idx:len(cl)] == 0) == (len(cl) - idx):
            cl = cl[0:idx]
            cdp = cdp[0:idx]
            break
    return cl, cdp


def clear_polar_results():
    # Clear saved polar results (for wing and htp airfoils)
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('af', 'csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('af', 'csv')))
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('.af', '_sym.csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('.af', '_sym.csv')))
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_HTP_AIRFOIL.replace('af', 'csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_HTP_AIRFOIL.replace('af', 'csv')))


def test_slipstream_openvsp():
    register_wrappers()

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeSlipstreamOpenvsp(
        propulsion_id=ENGINE_WRAPPER,
        result_folder_path=results_folder.name,
    )), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:wing:cruise:CL0_clean", val=0.1173)
    ivc.add_output("data:aerodynamics:wing:cruise:CL_alpha", val=4.5996, units="rad**-1")
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_max_clean", val=1.4465)
    # Run problem and check obtained value(s) is/(are) correct
    start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(ComputeSlipstreamOpenvsp(propulsion_id=ENGINE_WRAPPER,
                                                  result_folder_path=results_folder.name,
                                                  ), ivc)
    stop = time.time()
    duration_1st_run = stop - start
    y_vector_prop_on = problem.get_val("data:aerodynamics:slipstream:wing:prop_on:Y_vector", units="m")
    y_result_prop_on = np.array([0.05, 0.14, 0.23, 0.32, 0.41, 0.5 , 0.59, 0.72, 0.88, 1.04, 1.21,
                                 1.38, 1.54, 1.71, 1.88, 2.05, 2.21, 2.38, 2.55, 2.72, 2.89, 3.06,
                                 3.22, 3.39, 3.56, 3.72, 3.89, 4.05, 4.21, 4.37, 4.53, 4.69, 4.85,
                                 5.01, 5.16, 5.32, 5.47, 5.62, 5.77, 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0.])
    assert np.max(np.abs(y_vector_prop_on-y_result_prop_on)) <= 1e-2
    cl_vector_prop_on = problem.get_val("data:aerodynamics:slipstream:wing:prop_on:CL_vector")
    cl_result_prop_on = np.array([1.317, 1.664, 1.663, 1.656, 1.645, 1.625, 1.606, 1.553, 1.564,
                                  1.584, 1.602, 1.616, 1.62, 1.631, 1.645, 1.657, 1.663, 1.673,
                                  1.678, 1.684, 1.681, 1.686, 1.687, 1.692, 1.686, 1.688, 1.683,
                                  1.681, 1.667, 1.663, 1.648, 1.638, 1.61, 1.581, 1.525, 1.454,
                                  1.342, 1.163, 0.964, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    assert np.max(np.abs(cl_vector_prop_on - cl_result_prop_on)) <= 1e-2
    ct = problem.get_val("data:aerodynamics:slipstream:wing:prop_on:CT_ref")
    assert ct == pytest.approx(0.07245, abs=1e-4)
    cl_vector_prop_off = problem.get_val("data:aerodynamics:slipstream:wing:prop_off:CL_vector")
    y_vector_prop_off = problem.get_val("data:aerodynamics:slipstream:wing:prop_off:Y_vector")
    delta_cl = problem.get_val("data:aerodynamics:slipstream:wing:prop_on:CL") - \
               problem.get_val("data:aerodynamics:slipstream:wing:prop_off:CL")
    assert delta_cl == pytest.approx(0.00845, abs=1e-4)
