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

from ...tests.testing_utilities import run_system, register_wrappers, get_indep_var_comp, list_inputs
from ..components.cd0 import Cd0
from ..external.xfoil import XfoilPolar
from ..external.xfoil import resources
from ..external.vlm import ComputeAEROvlm, ComputeVNvlmNoVH
from ..external.openvsp import ComputeAEROopenvsp, ComputeVNopenvspNoVH
from ..external.openvsp.compute_aero_slipstream import ComputeSlipstreamOpenvsp
from ..components import ComputeExtremeCL, ComputeUnitReynolds, ComputeCnBetaFuselage, ComputeLDMax, \
    ComputeDeltaHighLift, Compute2DHingeMomentsTail, Compute3DHingeMomentsTail, ComputeMachInterpolation, \
    ComputeCyDeltaRudder, ComputeClalphaVT, ComputeAirfoilLiftCurveSlope
from ..aerodynamics_high_speed import AerodynamicsHighSpeed
from ..aerodynamics_low_speed import AerodynamicsLowSpeed

from ...tests.xfoil_exe.get_xfoil import get_xfoil_path
from ...propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from ...propulsion.propulsion import IPropulsion

from ..external.vlm.compute_aero import DEFAULT_WING_AIRFOIL, DEFAULT_HTP_AIRFOIL
from ..constants import SPAN_MESH_POINT, POLAR_POINT_COUNT

RESULTS_FOLDER = pth.join(pth.dirname(__file__), "results")
xfoil_path = None if system() == "Windows" else get_xfoil_path()

XML_FILE = "beechcraft_76.xml"
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
        flight_points.thrust = 1200.0
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

    def compute_max_power(self, flight_points: Union[FlightPoint, pd.DataFrame]) -> float:
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


def test_compute_reynolds():
    """ Tests high and low speed reynolds calculation """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeUnitReynolds()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeUnitReynolds(), ivc)
    mach = problem["data:aerodynamics:cruise:mach"]
    assert mach == pytest.approx(0.255041, abs=1e-4)
    reynolds = problem.get_val("data:aerodynamics:cruise:unit_reynolds", units="m**-1")
    assert reynolds == pytest.approx(4745380, abs=1)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeUnitReynolds(low_speed_aero=True)), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeUnitReynolds(low_speed_aero=True), ivc)
    mach = problem["data:aerodynamics:low_speed:mach"]
    assert mach == pytest.approx(0.1179, abs=1e-4)
    reynolds = problem.get_val("data:aerodynamics:low_speed:unit_reynolds", units="m**-1")
    assert reynolds == pytest.approx(2746999, abs=1)


def test_cd0_high_speed():
    """ Tests drag coefficient @ high speed """

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(Cd0(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.2457)
    ivc.add_output("data:aerodynamics:cruise:unit_reynolds", 4571770, units="m**-1")

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    # noinspection PyTypeChecker
    problem = run_system(Cd0(propulsion_id=ENGINE_WRAPPER), ivc)
    cd0_wing = problem["data:aerodynamics:wing:cruise:CD0"]
    assert cd0_wing == pytest.approx(0.00503, abs=1e-5)
    cd0_fus = problem["data:aerodynamics:fuselage:cruise:CD0"]
    assert cd0_fus == pytest.approx(0.00490, abs=1e-5)
    cd0_ht = problem["data:aerodynamics:horizontal_tail:cruise:CD0"]
    assert cd0_ht == pytest.approx(0.00123, abs=1e-5)
    cd0_vt = problem["data:aerodynamics:vertical_tail:cruise:CD0"]
    assert cd0_vt == pytest.approx(0.00077, abs=1e-5)
    cd0_nac = problem["data:aerodynamics:nacelles:cruise:CD0"]
    assert cd0_nac == pytest.approx(0.00185, abs=1e-5)
    cd0_lg = problem["data:aerodynamics:landing_gear:cruise:CD0"]
    assert cd0_lg == pytest.approx(0.0, abs=1e-5)
    cd0_other = problem["data:aerodynamics:other:cruise:CD0"]
    assert cd0_other == pytest.approx(0.00187, abs=1e-5)
    cd0_total = 1.25 * (cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac + cd0_lg + cd0_other)
    assert cd0_total == pytest.approx(0.01958, abs=1e-5)


def test_cd0_low_speed():
    """ Tests drag coefficient @ low speed """

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(Cd0(propulsion_id=ENGINE_WRAPPER, low_speed_aero=True)), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)  # correction to compensate old version conversion error
    ivc.add_output("data:aerodynamics:low_speed:unit_reynolds", 2613822, units="m**-1")  # correction to ...

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    # noinspection PyTypeChecker
    problem = run_system(Cd0(propulsion_id=ENGINE_WRAPPER, low_speed_aero=True), ivc)
    cd0_wing = problem["data:aerodynamics:wing:low_speed:CD0"]
    assert cd0_wing == pytest.approx(0.00552, abs=1e-5)
    cd0_fus = problem["data:aerodynamics:fuselage:low_speed:CD0"]
    assert cd0_fus == pytest.approx(0.00547, abs=1e-5)
    cd0_ht = problem["data:aerodynamics:horizontal_tail:low_speed:CD0"]
    assert cd0_ht == pytest.approx(0.00135, abs=1e-5)
    cd0_vt = problem["data:aerodynamics:vertical_tail:low_speed:CD0"]
    assert cd0_vt == pytest.approx(0.00086, abs=1e-5)
    cd0_nac = problem["data:aerodynamics:nacelles:low_speed:CD0"]
    assert cd0_nac == pytest.approx(0.00202, abs=1e-5)
    cd0_lg = problem["data:aerodynamics:landing_gear:low_speed:CD0"]
    assert cd0_lg == pytest.approx(0.01900, abs=1e-5)
    cd0_other = problem["data:aerodynamics:other:low_speed:CD0"]
    assert cd0_other == pytest.approx(0.00187, abs=1e-5)
    cd0_total = 1.25 * (cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac + cd0_lg + cd0_other)
    assert cd0_total == pytest.approx(0.04513, abs=1e-5)


def test_polar():
    """ Tests polar execution (XFOIL) @ high and low speed """

    # Clear saved polar results (for wing and htp airfoils)
    clear_polar_results()

    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, XML_FILE)
    ivc.add_output("xfoil:mach", 0.245)
    ivc.add_output("xfoil:reynolds", 4571770 * 1.549)

    # Run problem and check obtained value(s) is/(are) correct
    xfoil_comp = XfoilPolar(
        alpha_start=0.0, alpha_end=25.0, iter_limit=20, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)
    cl = problem["xfoil:CL"]
    cdp = problem["xfoil:CDp"]
    cl, cdp = reshape_polar(cl, cdp)
    cdp_1 = np.interp(1.0, cl, cdp)
    assert cdp_1 == pytest.approx(0.00466, abs=1e-4)

    # Define low-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, XML_FILE)
    ivc.add_output("xfoil:mach", 0.1179)
    ivc.add_output("xfoil:reynolds", 2746999 * 1.549)

    # Run problem and check obtained value(s) is/(are) correct
    xfoil_comp = XfoilPolar(
        alpha_start=0.0, alpha_end=25.0, iter_limit=20, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)
    cl = problem["xfoil:CL"]
    cdp = problem["xfoil:CDp"]
    cl_max_2d = problem["xfoil:CL_max_2D"]
    assert cl_max_2d == pytest.approx(1.6966, abs=1e-4)
    cl_min_2d = problem["xfoil:CL_min_2D"]
    assert cl_min_2d == pytest.approx(-1.3437, abs=1e-4)
    cl, cdp = reshape_polar(cl, cdp)
    cdp_1 = np.interp(1.0, cl, cdp)
    assert cdp_1 == pytest.approx(0.00486, abs=1e-4)


def test_airfoil_slope():
    """ Tests polar execution (XFOIL) @ high and low speed """

    # Clear saved polar results (for wing and htp airfoils)
    clear_polar_results()

    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(ComputeAirfoilLiftCurveSlope()), __file__, XML_FILE)
    ivc.add_output('data:aerodynamics:low_speed:unit_reynolds', val=2782216)
    ivc.add_output("data:aerodynamics:low_speed:mach", val=0.1194)
    problem = run_system(ComputeAirfoilLiftCurveSlope(wing_airfoil_file="naca63_415.af",
                                                      htp_airfoil_file="naca0012.af",
                                                      vtp_airfoil_file="naca0012.af"),ivc)

    cl_alpha_wing = problem.get_val("data:aerodynamics:wing:airfoil:Cl_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(5.3021, abs=1e-4)
    cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:airfoil:Cl_alpha", units="rad**-1")
    assert cl_alpha_htp == pytest.approx(6.4038, abs=1e-4)
    cl_alpha_vtp = problem.get_val("data:aerodynamics:vertical_tail:airfoil:Cl_alpha", units="rad**-1")
    assert cl_alpha_vtp == pytest.approx(6.4038, abs=1e-4)


def test_vlm_comp_high_speed():
    """ Tests vlm components @ high speed """

    # Create result temporary directory
    results_folder = _create_tmp_directory()
    mach_interpolation = True

    # Clear saved polar results (for wing and htp airfoils)
    clear_polar_results()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeAEROvlm()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(ComputeAEROvlm(result_folder_path=results_folder.name,
                                        compute_mach_interpolation=mach_interpolation), ivc)
    stop = time.time()
    duration_1st_run = stop - start
    cl0_wing = problem["data:aerodynamics:wing:cruise:CL0_clean"]
    assert cl0_wing == pytest.approx(0.1537, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(4.832, abs=1e-3)
    cm0 = problem["data:aerodynamics:wing:cruise:CM0_clean"]
    assert cm0 == pytest.approx(-0.0576, abs=1e-4)
    coef_k_wing = problem["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
    assert coef_k_wing == pytest.approx(0.05214, abs=1e-4)
    cl0_htp = problem["data:aerodynamics:horizontal_tail:cruise:CL0"]
    assert cl0_htp == pytest.approx(-0.0124, abs=1e-4)
    cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_htp == pytest.approx(0.6266, abs=1e-4)
    cl_alpha_htp_isolated = problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated",
                                            units="rad**-1")
    assert cl_alpha_htp_isolated == pytest.approx(1.0182, abs=1e-4)
    coef_k_htp = problem["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"]
    assert coef_k_htp == pytest.approx(0.2760, abs=1e-4)
    if mach_interpolation:
        cl_alpha_vector = problem["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"]
        assert cl_alpha_vector == pytest.approx([4.8224, 4.8224, 4.883, 4.964, 5.064, 5.184], abs=1e-2)
        mach_vector = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
        assert mach_vector == pytest.approx([0., 0.15, 0.217, 0.28, 0.339, 0.395], abs=1e-2)

    # Run problem 2nd time to check time reduction
    start = time.time()
    # noinspection PyTypeChecker
    run_system(ComputeAEROvlm(result_folder_path=results_folder.name), ivc)
    stop = time.time()
    duration_2nd_run = stop - start
    assert (duration_2nd_run / duration_1st_run) <= 0.1

    # Remove existing result files
    results_folder.cleanup()


def test_vlm_comp_low_speed():
    """ Tests vlm components @ low speed """

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Clear saved polar results (for wing and htp airfoils)
    clear_polar_results()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeAEROvlm(low_speed_aero=True)), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(ComputeAEROvlm(low_speed_aero=True, result_folder_path=results_folder.name), ivc)
    stop = time.time()
    duration_1st_run = stop - start
    cl0_wing = problem["data:aerodynamics:wing:low_speed:CL0_clean"]
    assert cl0_wing == pytest.approx(0.1497, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(4.705, abs=1e-3)
    cm0 = problem["data:aerodynamics:wing:low_speed:CM0_clean"]
    assert cm0 == pytest.approx(-0.0561, abs=1e-4)
    y_vector_wing = np.array(
        [0.09983333, 0.2995, 0.49916667, 0.918, 1.556,
         2.194, 2.832, 3.47, 4.108, 4.746,
         5.14475, 5.30425, 5.46375, 5.62325, 5.78275,
         5.94225, 6.10175]
    )
    cl_vector_wing = np.array(
        [0.14646494, 0.14888531, 0.15486327, 0.17197724, 0.17576075,
         0.17556516, 0.1729675, 0.16789989, 0.15929893, 0.14418323,
         0.11857587, 0.10500625, 0.0938145, 0.08305617, 0.07155494,
         0.05793449, 0.0393112]
    )
    chord_vector_wing = np.array(
        [1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549, 1.549,
         1.549, 1.549, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0.]
    )
    y, cl = reshape_curve(problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", "m"),
                          problem["data:aerodynamics:wing:low_speed:CL_vector"])
    chord = problem.get_val("data:aerodynamics:wing:low_speed:chord_vector", "m")
    assert np.max(np.abs(y_vector_wing - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_wing - cl)) <= 1e-3
    assert np.max(np.abs(chord_vector_wing - chord)) <= 1e-3
    coef_k_wing = problem["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
    assert coef_k_wing == pytest.approx(0.04978358, abs=1e-4)
    cl0_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL0"]
    assert cl0_htp == pytest.approx(-0.0118, abs=1e-4)
    cl_ref_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_ref"]
    assert cl_ref_htp == pytest.approx(0.0964, abs=1e-4)
    y_vector_htp = np.array(
        [0.07492647, 0.22477941, 0.37463235, 0.52448529, 0.67433824,
         0.82419118, 0.97404412, 1.12389706, 1.27375, 1.42360294,
         1.57345588, 1.72330882, 1.87316176, 2.02301471, 2.17286765,
         2.32272059, 2.47257353]
    )
    cl_vector_htp = np.array(
        [0.1127779, 0.11258886, 0.11220528, 0.1116158, 0.11080241,
         0.10973912, 0.10839005, 0.10670632, 0.10462163, 0.10204526,
         0.09885125, 0.09486042, 0.08980921, 0.08329037, 0.07462436,
         0.06251549, 0.0437194]
    )
    y, cl = reshape_curve(problem.get_val("data:aerodynamics:horizontal_tail:low_speed:Y_vector", "m"),
                          problem["data:aerodynamics:horizontal_tail:low_speed:CL_vector"])
    assert np.max(np.abs(y_vector_htp - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_htp - cl)) <= 1e-3
    cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_htp == pytest.approx(0.6202, abs=1e-4)
    cl_alpha_htp_isolated = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated",
                                            units="rad**-1")
    assert cl_alpha_htp_isolated == pytest.approx(0.9915, abs=1e-4)
    coef_k_htp = problem["data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"]
    assert coef_k_htp == pytest.approx(0.2800, abs=1e-4)

    # Run problem 2nd time to check time reduction
    start = time.time()
    # noinspection PyTypeChecker
    run_system(ComputeAEROvlm(low_speed_aero=True, result_folder_path=results_folder.name), ivc)
    stop = time.time()
    duration_2nd_run = stop - start
    assert (duration_2nd_run / duration_1st_run) <= 0.1

    # Remove existing result files
    results_folder.cleanup()


def test_openvsp_comp_high_speed():
    """ Tests openvsp components @ high speed """

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeAEROopenvsp()), __file__, XML_FILE)

    mach_interpolation = True
    # Run problem and check obtained value(s) is/(are) correct
    start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(ComputeAEROopenvsp(result_folder_path=results_folder.name,
                                            compute_mach_interpolation=mach_interpolation), ivc)
    stop = time.time()
    duration_1st_run = stop - start
    cl0_wing = problem["data:aerodynamics:wing:cruise:CL0_clean"]
    assert cl0_wing == pytest.approx(0.0908, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(4.659, abs=1e-3)
    cm0 = problem["data:aerodynamics:wing:cruise:CM0_clean"]
    assert cm0 == pytest.approx(-0.0142, abs=1e-4)
    coef_k_wing = problem["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
    assert coef_k_wing == pytest.approx(0.0479, abs=1e-4)
    if mach_interpolation:
        cl_alpha_vector = problem["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"]
        assert cl_alpha_vector == pytest.approx([5.26, 5.26, 5.30, 5.36, 5.438, 5.524], abs=1e-2)
        mach_vector = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
        assert mach_vector == pytest.approx([0., 0.15, 0.21, 0.28, 0.34, 0.39], abs=1e-2)
    cl0_htp = problem["data:aerodynamics:horizontal_tail:cruise:CL0"]
    assert cl0_htp == pytest.approx(-0.0048, abs=1e-4)
    cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_htp == pytest.approx(0.6816, abs=1e-4)
    cl_alpha_htp_isolated = problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_alpha_isolated",
                                            units="rad**-1")
    assert cl_alpha_htp_isolated == pytest.approx(1.0387, abs=1e-4)
    coef_k_htp = problem["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"]
    assert coef_k_htp == pytest.approx(0.4563, abs=1e-4)

    # Run problem 2nd time to check time reduction
    start = time.time()
    # noinspection PyTypeChecker
    run_system(ComputeAEROopenvsp(result_folder_path=results_folder.name), ivc)
    stop = time.time()
    duration_2nd_run = stop - start
    assert (duration_2nd_run / duration_1st_run) <= 0.01

    # Remove existing result files
    results_folder.cleanup()


def test_openvsp_comp_low_speed():
    """ Tests openvsp components @ low speed """

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeAEROopenvsp(low_speed_aero=True)), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(ComputeAEROopenvsp(low_speed_aero=True, result_folder_path=results_folder.name), ivc)
    stop = time.time()
    duration_1st_run = stop - start
    cl0_wing = problem["data:aerodynamics:wing:low_speed:CL0_clean"]
    # assert cl0_wing == pytest.approx(0.0890, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1")
    # assert cl_alpha_wing == pytest.approx(4.571, abs=1e-3)
    cm0 = problem["data:aerodynamics:wing:low_speed:CM0_clean"]
    # assert cm0 == pytest.approx(-0.0138, abs=1e-4)
    y_vector_wing = np.array(
        [0.04279, 0.12836, 0.21393, 0.2995, 0.38507, 0.47064, 0.55621,
         0.68649, 0.862, 1.0385, 1.21588, 1.39404, 1.57287, 1.75226,
         1.93212, 2.11231, 2.29274, 2.47328, 2.65384, 2.83429, 3.01452,
         3.19443, 3.37389, 3.5528, 3.73106, 3.90855, 4.08517, 4.26082,
         4.43539, 4.60879, 4.78093, 4.9517, 5.12103, 5.28882, 5.45499,
         5.61947, 5.78218, 5.94305, 6.10201]
    )
    cl_vector_wing = np.array(
        [0.10023882, 0.10037913, 0.10035909, 0.10029896, 0.10019874,
         0.10011856, 0.10005843, 0.09982793, 0.09919655, 0.09960745,
         0.09998828, 0.09992815, 0.09934688, 0.09897607, 0.09888587,
         0.0986253, 0.09815427, 0.09771331, 0.097112, 0.09648062,
         0.09557865, 0.09483703, 0.09409541, 0.09330368, 0.09211108,
         0.09094854, 0.08949537, 0.08806224, 0.08623826, 0.08458465,
         0.08256023, 0.08040553, 0.07766955, 0.07486343, 0.07080457,
         0.06589385, 0.05898877, 0.04879652, 0.03465565])

    y, cl = reshape_curve(problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", "m"),
                          problem["data:aerodynamics:wing:low_speed:CL_vector"])
    assert np.max(np.abs(y_vector_wing - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_wing - cl)) <= 1e-3
    coef_k_wing = problem["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
    # assert coef_k_wing == pytest.approx(0.0479, abs=1e-4)
    cl0_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL0"]
    # assert cl0_htp == pytest.approx(-0.00472, abs=1e-4)
    cl_ref_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_ref"]
    # assert cl_ref_htp == pytest.approx(0.1131, abs=1e-4)
    y_vector_htp = np.array(
        [0.05307, 0.15922, 0.26536, 0.37151, 0.47766, 0.5838, 0.68995,
         0.79609, 0.90224, 1.00839, 1.11453, 1.22068, 1.32682, 1.43297,
         1.53911, 1.64526, 1.75141, 1.85755, 1.9637, 2.06984, 2.17599,
         2.28214, 2.38828, 2.49443]
    )
    cl_vector_htp = np.array(
        [0.12782928, 0.1302331, 0.13057254, 0.13035779, 0.13057023,
         0.12999525, 0.12855666, 0.12766995, 0.12682712, 0.12582726,
         0.12432632, 0.12271916, 0.12039617, 0.11851884, 0.11624895,
         0.11392827, 0.11091484, 0.10757121, 0.10261349, 0.09740408,
         0.08983241, 0.08127013, 0.06790021, 0.05280999]
    )
    test = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:Y_vector")
    y, cl = reshape_curve(problem.get_val("data:aerodynamics:horizontal_tail:low_speed:Y_vector", "m"),
                          problem["data:aerodynamics:horizontal_tail:low_speed:CL_vector"])
    assert np.max(np.abs(y_vector_htp - y)) <= 1e-3
    assert np.max(np.abs(cl_vector_htp - cl)) <= 1e-3
    cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_htp == pytest.approx(0.6752, abs=1e-4)
    cl_alpha_htp_isolated = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha_isolated",
                                            units="rad**-1")
    assert cl_alpha_htp_isolated == pytest.approx(1.0198, abs=1e-4)
    coef_k_htp = problem["data:aerodynamics:horizontal_tail:low_speed:induced_drag_coefficient"]
    assert coef_k_htp == pytest.approx(0.4547, abs=1e-4)

    # Run problem 2nd time to check time reduction
    start = time.time()
    # noinspection PyTypeChecker
    run_system(ComputeAEROopenvsp(low_speed_aero=True, result_folder_path=results_folder.name), ivc)
    stop = time.time()
    duration_2nd_run = stop - start
    assert (duration_2nd_run / duration_1st_run) <= 0.1

    # Remove existing result files
    results_folder.cleanup()


def test_2d_hinge_moment():
    """ Tests tail hinge-moments """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(Compute2DHingeMomentsTail()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:horizontal_tail:cruise:CL_alpha", 0.6826, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(Compute2DHingeMomentsTail(), ivc)
    ch_alpha_2d = problem.get_val("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D", units="rad**-1")
    assert ch_alpha_2d == pytest.approx(-0.3339, abs=1e-4)
    ch_delta_2d = problem.get_val("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D", units="rad**-1")
    assert ch_delta_2d == pytest.approx(-0.6358, abs=1e-4)


def test_3d_hinge_moment():
    """ Tests tail hinge-moments """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(Compute3DHingeMomentsTail()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha_2D", -0.3339, units="rad**-1")
    ivc.add_output("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta_2D", -0.6358, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(Compute3DHingeMomentsTail(), ivc)
    ch_alpha = problem.get_val("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_alpha", units="rad**-1")
    assert ch_alpha == pytest.approx(-0.2486, abs=1e-4)
    ch_delta = problem.get_val("data:aerodynamics:horizontal_tail:cruise:hinge_moment:CH_delta", units="rad**-1")
    assert ch_delta == pytest.approx(-0.6765, abs=1e-4)


def test_high_lift():
    """ Tests high-lift contribution """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeDeltaHighLift()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_alpha", 4.569, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeDeltaHighLift(), ivc)
    delta_cl0_landing = problem["data:aerodynamics:flaps:landing:CL"]
    assert delta_cl0_landing == pytest.approx(0.7321, abs=1e-4)
    delta_clmax_landing = problem["data:aerodynamics:flaps:landing:CL_max"]
    assert delta_clmax_landing == pytest.approx(0.5258, abs=1e-4)
    delta_cm_landing = problem["data:aerodynamics:flaps:landing:CM"]
    assert delta_cm_landing == pytest.approx(-0.0988, abs=1e-4)
    delta_cd_landing = problem["data:aerodynamics:flaps:landing:CD"]
    assert delta_cd_landing == pytest.approx(0.01383, abs=1e-4)
    delta_cl0_takeoff = problem["data:aerodynamics:flaps:takeoff:CL"]
    assert delta_cl0_takeoff == pytest.approx(0.2805, abs=1e-4)
    delta_clmax_takeoff = problem["data:aerodynamics:flaps:takeoff:CL_max"]
    assert delta_clmax_takeoff == pytest.approx(0.1076, abs=1e-4)
    delta_cm_takeoff = problem["data:aerodynamics:flaps:takeoff:CM"]
    assert delta_cm_takeoff == pytest.approx(-0.0378, abs=1e-4)
    delta_cd_takeoff = problem["data:aerodynamics:flaps:takeoff:CD"]
    assert delta_cd_takeoff == pytest.approx(0.00111811, abs=1e-4)
    cl_delta_elev = problem.get_val("data:aerodynamics:elevator:low_speed:CL_delta", units="rad**-1")
    assert cl_delta_elev == pytest.approx(0.6167, abs=1e-4)


def test_extreme_cl():
    """ Tests maximum/minimum cl component with default result cl=f(y) curve"""

    # Clear saved polar results (for wing and htp airfoils)
    clear_polar_results()

    # Research independent input value in .xml file for Openvsp test
    ivc = get_indep_var_comp(list_inputs(ComputeExtremeCL()), __file__, XML_FILE)
    y_vector_wing = np.zeros(SPAN_MESH_POINT)
    cl_vector_wing = np.zeros(SPAN_MESH_POINT)
    y_vector_htp = np.zeros(SPAN_MESH_POINT)
    cl_vector_htp = np.zeros(SPAN_MESH_POINT)
    y_vector_wing[0:39] = [0.04279, 0.12836, 0.21393, 0.2995, 0.38507, 0.47064, 0.55621,
                           0.68649, 0.862, 1.0385, 1.21588, 1.39404, 1.57287, 1.75226,
                           1.93212, 2.11231, 2.29274, 2.47328, 2.65384, 2.83429, 3.01452,
                           3.19443, 3.37389, 3.5528, 3.73106, 3.90855, 4.08517, 4.26082,
                           4.43539, 4.60879, 4.78093, 4.9517, 5.12103, 5.28882, 5.45499,
                           5.61947, 5.78218, 5.94305, 6.10201]
    cl_vector_wing[0:39] = [0.0989, 0.09908, 0.09901, 0.09898, 0.09892, 0.09888, 0.09887,
                            0.09871, 0.09823, 0.09859, 0.09894, 0.09888, 0.09837, 0.098,
                            0.0979, 0.09763, 0.09716, 0.09671, 0.0961, 0.09545, 0.09454,
                            0.09377, 0.09295, 0.09209, 0.09087, 0.08965, 0.08812, 0.0866,
                            0.08465, 0.08284, 0.08059, 0.07817, 0.07494, 0.07178, 0.06773,
                            0.06279, 0.05602, 0.04639, 0.03265]
    y_vector_htp[0:24] = [0.05307, 0.15922, 0.26536, 0.37151, 0.47766, 0.5838, 0.68995,
                          0.79609, 0.90224, 1.00839, 1.11453, 1.22068, 1.32682, 1.43297,
                          1.53911, 1.64526, 1.75141, 1.85755, 1.9637, 2.06984, 2.17599,
                          2.28214, 2.38828, 2.49443]
    cl_vector_htp[0:24] = [0.12706265, 0.12950803, 0.12983592, 0.12961425, 0.12981745,
                           0.12923324, 0.12780388, 0.12690332, 0.12606048, 0.12506524,
                           0.12356892, 0.12197331, 0.11966648, 0.11778684, 0.11551696,
                           0.11321936, 0.11022672, 0.10689925, 0.1019831, 0.0968037,
                           0.08927591, 0.08074826, 0.06745917, 0.05250057]
    ivc.add_output("data:aerodynamics:flaps:landing:CL_max", 0.5788)
    ivc.add_output("data:aerodynamics:flaps:takeoff:CL_max", 0.1218)
    ivc.add_output("data:aerodynamics:wing:low_speed:Y_vector", y_vector_wing, units="m")
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_vector", cl_vector_wing)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL0_clean", 0.0877)
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:Y_vector", y_vector_htp, units="m")
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_vector", cl_vector_htp)
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_ref", 0.11245)
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", 0.6760, units="rad**-1")
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)
    ivc.add_output("data:aerodynamics:low_speed:unit_reynolds", 2613822, units="m**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeExtremeCL(), ivc)
    cl_max_clean_wing = problem["data:aerodynamics:wing:low_speed:CL_max_clean"]
    assert cl_max_clean_wing == pytest.approx(1.499, abs=1e-2)
    cl_min_clean_wing = problem["data:aerodynamics:wing:low_speed:CL_min_clean"]
    assert cl_min_clean_wing == pytest.approx(-1.1912, abs=1e-2)
    cl_max_takeoff_wing = problem["data:aerodynamics:aircraft:takeoff:CL_max"]
    assert cl_max_takeoff_wing == pytest.approx(1.621, abs=1e-2)
    cl_max_landing_wing = problem["data:aerodynamics:aircraft:landing:CL_max"]
    assert cl_max_landing_wing == pytest.approx(2.078, abs=1e-2)
    cl_max_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_max_clean"]
    assert cl_max_clean_htp == pytest.approx(1.361, abs=1e-2)
    cl_min_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_min_clean"]
    assert cl_min_clean_htp == pytest.approx(-1.361, abs=1e-2)
    alpha_max_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_max"]
    assert alpha_max_clean_htp == pytest.approx(26.644, abs=1e-2)
    alpha_min_clean_htp = problem["data:aerodynamics:horizontal_tail:low_speed:clean:alpha_aircraft_min"]
    assert alpha_min_clean_htp == pytest.approx(-26.644, abs=1e-2)


def test_l_d_max():
    """ Tests best lift/drag component """

    # Define independent input value (openVSP)
    ivc = om.IndepVarComp()
    ivc.add_output("data:aerodynamics:wing:cruise:CL0_clean", 0.0906)
    ivc.add_output("data:aerodynamics:wing:cruise:CL_alpha", 4.650, units="rad**-1")
    ivc.add_output("data:aerodynamics:aircraft:cruise:CD0", 0.01603)
    ivc.add_output("data:aerodynamics:wing:cruise:induced_drag_coefficient", 0.0480)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLDMax(), ivc)
    l_d_max = problem["data:aerodynamics:aircraft:cruise:L_D_max"]
    assert l_d_max == pytest.approx(18.0, abs=1e-1)
    optimal_cl = problem["data:aerodynamics:aircraft:cruise:optimal_CL"]
    assert optimal_cl == pytest.approx(0.5778, abs=1e-4)
    optimal_cd = problem["data:aerodynamics:aircraft:cruise:optimal_CD"]
    assert optimal_cd == pytest.approx(0.0320, abs=1e-4)
    optimal_alpha = problem.get_val("data:aerodynamics:aircraft:cruise:optimal_alpha", units="deg")
    assert optimal_alpha == pytest.approx(6.00, abs=1e-2)


def test_cnbeta():
    """ Tests cn beta fuselage """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCnBetaFuselage()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.245)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnBetaFuselage(), ivc)
    cn_beta_fus = problem["data:aerodynamics:fuselage:cruise:CnBeta"]
    assert cn_beta_fus == pytest.approx(-0.0599, abs=1e-4)


def test_cl_alpha_vt():
    """ Tests Cl alpha vt """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeClalphaVT(low_speed_aero=True)), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.119)
    ivc.add_output("data:aerodynamics:vertical_tail:airfoil:Cl_alpha", 6.4038, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClalphaVT(low_speed_aero=True), ivc)
    cl_alpha_vt_ls = problem.get_val("data:aerodynamics:vertical_tail:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_vt_ls == pytest.approx(2.92419, abs=1e-4)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeClalphaVT()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.248)
    ivc.add_output("data:aerodynamics:vertical_tail:airfoil:Cl_alpha", 6.4038, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeClalphaVT(), ivc)
    cl_alpha_vt_cruise = problem.get_val("data:aerodynamics:vertical_tail:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_vt_cruise == pytest.approx(3.0044, abs=1e-4)


def test_cy_delta_r():
    """ Tests cy delta of the rudder """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCyDeltaRudder()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:vertical_tail:low_speed:CL_alpha", val=1.94358, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCyDeltaRudder(), ivc)
    cy_delta_r = problem.get_val("data:aerodynamics:rudder:low_speed:Cy_delta_r", units="rad**-1")
    assert cy_delta_r == pytest.approx(1.0269, abs=1e-4)


def test_high_speed_connection():
    """ Tests high speed components connection """

    # Clear saved polar results (for wing and htp airfoils)
    clear_polar_results()

    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    input_vars.add_output("data:aerodynamics:vertical_tail:airfoil:Cl_alpha", 6.4038, units="rad**-1")
    register_wrappers()

    # Run problem with VLM
    # noinspection PyTypeChecker
    run_system(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=False), input_vars)

    # Run problem with OPENVSP
    # noinspection PyTypeChecker
    run_system(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=True), input_vars)


def test_low_speed_connection():
    """ Tests low speed components connection """

    # Clear saved polar results (for wing and htp airfoils)
    clear_polar_results()

    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    register_wrappers()

    # Run problem with VLM
    # noinspection PyTypeChecker
    run_system(AerodynamicsLowSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=False), input_vars)

    # Run problem with OPENVSP
    # noinspection PyTypeChecker
    run_system(AerodynamicsLowSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=True), input_vars)


def test_v_n_diagram_vlm():
    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    cl_wing_airfoil = np.zeros(POLAR_POINT_COUNT)
    cdp_wing_airfoil = np.zeros(POLAR_POINT_COUNT)
    cl_htp_airfoil = np.zeros(POLAR_POINT_COUNT)
    cdp_htp_airfoil = np.zeros(POLAR_POINT_COUNT)
    cl_wing_airfoil[0:38] = np.array(
        [0.1391, 0.1988, 0.2581, 0.3177, 0.377, 0.4903, 0.5477, 0.6062,
         0.6647, 0.7226, 0.7807, 0.838, 0.8939, 0.9473, 1.1335, 1.1968,
         1.2451, 1.296, 1.3424, 1.4014, 1.4597, 1.5118, 1.5575, 1.6006,
         1.6383, 1.664, 1.6845, 1.7023, 1.7152, 1.7196, 1.7121, 1.6871,
         1.6386, 1.563, 1.4764, 1.3993, 1.3418, 1.2981]
    )
    cdp_wing_airfoil[0:38] = np.array(
        [0.00143, 0.00147, 0.00154, 0.00163, 0.00173, 0.00196, 0.00214,
         0.00235, 0.0026, 0.00287, 0.00317, 0.00349, 0.00385, 0.00424,
         0.00572, 0.00636, 0.00701, 0.00777, 0.00908, 0.00913, 0.00923,
         0.00982, 0.01098, 0.01221, 0.01357, 0.01508, 0.01715, 0.01974,
         0.02318, 0.02804, 0.035, 0.04486, 0.05824, 0.07544, 0.09465,
         0.1133, 0.1299, 0.14507]
    )
    cl_htp_airfoil[0:41] = np.array(
        [-0., 0.0582, 0.117, 0.1751, 0.2333, 0.291, 0.3486,
         0.4064, 0.4641, 0.5216, 0.5789, 0.6356, 0.6923, 0.747,
         0.8027, 0.8632, 0.9254, 0.9935, 1.0611, 1.127, 1.1796,
         1.227, 1.2762, 1.3255, 1.3756, 1.4232, 1.4658, 1.5084,
         1.5413, 1.5655, 1.5848, 1.5975, 1.6002, 1.5894, 1.5613,
         1.5147, 1.4515, 1.3761, 1.2892, 1.1988, 1.1276]
    )
    cdp_htp_airfoil[0:41] = np.array(
        [0.00074, 0.00075, 0.00078, 0.00086, 0.00095, 0.00109, 0.00126,
         0.00145, 0.00167, 0.00191, 0.00218, 0.00249, 0.00283, 0.00324,
         0.00365, 0.00405, 0.00453, 0.00508, 0.00559, 0.00624, 0.00679,
         0.0074, 0.00813, 0.00905, 0.01, 0.01111, 0.0126, 0.01393,
         0.0155, 0.01743, 0.01993, 0.02332, 0.0282, 0.03541, 0.04577,
         0.05938, 0.07576, 0.0944, 0.11556, 0.13878, 0.16068]
    )
    input_vars.add_output("data:aerodynamics:wing:cruise:CL", cl_wing_airfoil)
    input_vars.add_output("data:aerodynamics:wing:cruise:CDp", cdp_wing_airfoil)
    input_vars.add_output("data:aerodynamics:horizontal_tail:cruise:CL", cl_htp_airfoil)
    input_vars.add_output("data:aerodynamics:horizontal_tail:cruise:CDp", cdp_htp_airfoil)
    input_vars.add_output("data:aerodynamics:aircraft:landing:CL_max", 1.9)
    input_vars.add_output("data:aerodynamics:wing:low_speed:CL_max_clean", 1.5)
    input_vars.add_output("data:aerodynamics:wing:low_speed:CL_min_clean", -1.5)
    input_vars.add_output("data:weight:aircraft:MTOW", 1700.0, units="kg")
    input_vars.add_output("data:aerodynamics:cruise:mach", 0.21)
    input_vars.add_output("data:aerodynamics:wing:cruise:induced_drag_coefficient", 0.048)
    input_vars.add_output("data:aerodynamics:aircraft:cruise:CD0", 0.016)
    input_vars.add_output("data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector",
                          [4.823, 4.823, 4.884, 4.964, 5.064, 5.184])
    input_vars.add_output("data:aerodynamics:aircraft:mach_interpolation:mach_vector",
                          [0., 0.15, 0.217, 0.28, 0.339, 0.395])

    register_wrappers()

    # Run problem with VLM and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(ComputeVNvlmNoVH(propulsion_id=ENGINE_WRAPPER, compute_cl_alpha=True), input_vars)
    velocity_vect = np.array(
        [30.782, 30.782, 60.006, 37.951, 0., 0., 74.799,
         74.799, 74.799, 101.345, 101.345, 101.345, 101.345, 91.211,
         74.799, 0., 27.351, 38.68, 49.232]
    )
    load_factor_vect = np.array(
        [1., -1., 3.8, -1.52, 0., 0., -1.52, 3.836,
         -1.836, 3.8, 0., 2.957, -0.957, 0., 0., 0.,
         1., 2., 2.]
    )
    velocity_array = problem.get_val("data:flight_domain:velocity", units="m/s")
    load_factor_array = problem["data:flight_domain:load_factor"]
    assert np.max(np.abs(velocity_vect - velocity_array)) <= 1e-3
    assert np.max(np.abs(load_factor_vect - load_factor_array)) <= 1e-3


def test_v_n_diagram_openvsp():
    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    input_vars.add_output("data:aerodynamics:aircraft:landing:CL_max", 1.9)
    input_vars.add_output("data:aerodynamics:wing:low_speed:CL_max_clean", 1.5)
    input_vars.add_output("data:aerodynamics:wing:low_speed:CL_min_clean", -1.5)
    input_vars.add_output("data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector",
                          [5.20, 5.20, 5.24, 5.30, 5.37, 5.45])
    input_vars.add_output("data:aerodynamics:aircraft:mach_interpolation:mach_vector",
                          [0., 0.15, 0.21, 0.28, 0.34, 0.39])
    input_vars.add_output("data:weight:aircraft:MTOW", 1700.0, units="kg")
    input_vars.add_output("data:aerodynamics:cruise:mach", 0.21)
    input_vars.add_output("data:aerodynamics:wing:cruise:induced_drag_coefficient", 0.048)
    input_vars.add_output("data:aerodynamics:aircraft:cruise:CD0", 0.016)

    register_wrappers()

    # Run problem with VLM and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(ComputeVNopenvspNoVH(propulsion_id=ENGINE_WRAPPER, compute_cl_alpha=True), input_vars)
    velocity_vect = np.array(
        [30.782, 30.782, 60.006, 37.951, 0., 0., 74.799, 74.799, 74.799, 101.345, 101.345, 101.345, 101.345, 91.210,
         74.799, 0., 27.350, 38.680, 49.231]
    )
    load_factor_vect = np.array(
        [1., -1., 3.8, -1.52, 0., 0., -1.52, 4.004, -2.004, 3.8, 0., 3.059, -1.059, 0., 0., 0., 1., 2., 2.]
    )
    velocity_array = problem.get_val("data:flight_domain:velocity", units="m/s")
    load_factor_array = problem["data:flight_domain:load_factor"]
    assert np.max(np.abs(velocity_vect - velocity_array)) <= 1e-3
    assert np.max(np.abs(load_factor_vect - load_factor_array)) <= 1e-3


def test_slipstream_openvsp_cruise():
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
                                                  low_speed_aero=False
                                                  ), ivc)
    stop = time.time()
    duration_1st_run = stop - start
    y_vector_prop_on = problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_on:Y_vector", units="m")
    y_result_prop_on = np.array([0.04, 0.13, 0.21, 0.3, 0.39, 0.47, 0.56, 0.69, 0.86, 1.04, 1.22,
                                 1.39, 1.57, 1.75, 1.93, 2.11, 2.29, 2.47, 2.65, 2.83, 3.01, 3.19,
                                 3.37, 3.55, 3.73, 3.91, 4.09, 4.26, 4.44, 4.61, 4.78, 4.95, 5.12,
                                 5.29, 5.45, 5.62, 5.78, 5.94, 6.1, 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0.])
    assert np.max(np.abs(y_vector_prop_on - y_result_prop_on)) <= 1e-2
    cl_vector_prop_on = problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_on:CL_vector")
    cl_result_prop_on = np.array([1.60206, 1.59939, 1.59986, 1.59958, 1.59926, 1.59829, 1.60035,
                                  1.59493, 1.58698, 1.59456, 1.60512, 1.65516, 1.68709, 1.69243,
                                  1.67626, 1.61866, 1.58831, 1.56764, 1.55701, 1.53151, 1.48649,
                                  1.48009, 1.47012, 1.45894, 1.43868, 1.41974, 1.39493, 1.371,
                                  1.33662, 1.30773, 1.27104, 1.23496, 1.18546, 1.13893, 1.07154,
                                  0.99839, 0.90112, 0.77591, 0.74955, 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0.,
                                  0.])
    assert np.max(np.abs(cl_vector_prop_on - cl_result_prop_on)) <= 1e-2
    ct = problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_on:CT_ref")
    assert ct == pytest.approx(0.0483, abs=1e-4)
    delta_cl = problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_on:CL") - \
               problem.get_val("data:aerodynamics:slipstream:wing:cruise:prop_off:CL")
    assert delta_cl == pytest.approx(0.00689, abs=1e-4)


def test_slipstream_openvsp_low_speed():
    register_wrappers()

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeSlipstreamOpenvsp(
        propulsion_id=ENGINE_WRAPPER,
        result_folder_path=results_folder.name,
        low_speed_aero=True
    )), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL0_clean", val=0.1147)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_alpha", val=4.509, units="rad**-1")
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_max_clean", val=1.4465)
    # Run problem and check obtained value(s) is/(are) correct
    start = time.time()
    # noinspection PyTypeChecker
    problem = run_system(ComputeSlipstreamOpenvsp(propulsion_id=ENGINE_WRAPPER,
                                                  result_folder_path=results_folder.name,
                                                  low_speed_aero=True
                                                  ), ivc)
    stop = time.time()
    duration_1st_run = stop - start
    y_vector_prop_on = problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_on:Y_vector", units="m")
    y_result_prop_on = np.array([0.04, 0.13, 0.21, 0.3, 0.39, 0.47, 0.56, 0.69, 0.86, 1.04, 1.22,
                                 1.39, 1.57, 1.75, 1.93, 2.11, 2.29, 2.47, 2.65, 2.83, 3.01, 3.19,
                                 3.37, 3.55, 3.73, 3.91, 4.09, 4.26, 4.44, 4.61, 4.78, 4.95, 5.12,
                                 5.29, 5.45, 5.62, 5.78, 5.94, 6.1, 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0.])
    assert np.max(np.abs(y_vector_prop_on - y_result_prop_on)) <= 1e-2
    cl_vector_prop_on = problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_on:CL_vector")
    cl_result_prop_on = np.array([1.63252, 1.59405, 1.59457, 1.59435, 1.5943, 1.59348, 1.59584,
                                  1.59057, 1.58322, 1.59158, 1.60355, 1.76394, 1.85941, 1.891,
                                  1.86978, 1.76704, 1.71872, 1.68112, 1.63639, 1.58343, 1.46312,
                                  1.46219, 1.45582, 1.44729, 1.4292, 1.41188, 1.38847, 1.36559,
                                  1.33226, 1.30432, 1.26875, 1.23362, 1.18512, 1.13962, 1.07347,
                                  1.00126, 0.90534, 0.78125, 0.75797, 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0.,
                                  0.])
    assert np.max(np.abs(cl_vector_prop_on - cl_result_prop_on)) <= 1e-2
    ct = problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_on:CT_ref")
    assert ct == pytest.approx(0.03796, abs=1e-4)
    delta_cl = problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_on:CL") - \
               problem.get_val("data:aerodynamics:slipstream:wing:low_speed:prop_off:CL")
    assert delta_cl == pytest.approx(0.02301, abs=1e-4)


def test_compute_mach_interpolation_roskam():
    """ Tests computation of the mach interpolation vector using Roskam's approach """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeMachInterpolation()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMachInterpolation(), ivc)
    cl_alpha_vector = problem["data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector"]
    cl_alpha_result = np.array([5.456, 5.48, 5.556, 5.686, 5.879, 6.145])
    assert np.max(np.abs(cl_alpha_vector - cl_alpha_result)) <= 1e-2
    mach_vector = problem["data:aerodynamics:aircraft:mach_interpolation:mach_vector"]
    mach_result = np.array([0., 0.079, 0.158, 0.237, 0.316, 0.395])
    assert np.max(np.abs(mach_vector - mach_result)) <= 1e-2
