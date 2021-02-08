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

from fastoad.io import VariableIO
from fastoad.models.aerodynamics.constants import POLAR_POINT_COUNT
from fastoad.module_management.service_registry import RegisterPropulsion
from fastoad import BundleLoader
from fastoad.base.flight_point import FlightPoint
from fastoad.models.propulsion.propulsion import IOMPropulsionWrapper

from ...tests.testing_utilities import run_system, register_wrappers, get_indep_var_comp, list_inputs, Timer
from ..components.cd0 import Cd0
from ..external.vlm import ComputeOSWALDvlm, ComputeWingCLALPHAvlm, ComputeHTPCLALPHAvlm, ComputeHTPCLCMvlm, \
    ComputeAEROvlm
from ..external.vlm.compute_aero import DEFAULT_WING_AIRFOIL, DEFAULT_HTP_AIRFOIL
from ..external.xfoil import XfoilPolar
from ..external.xfoil import resources
from ..external.openvsp import ComputeOSWALDopenvsp, ComputeWingCLALPHAopenvsp, ComputeHTPCLALPHAopenvsp, \
    ComputeHTPCLCMopenvsp
from ..external.openvsp.compute_vn import ComputeVNopenvsp
from ..external.openvsp.compute_aero2 import ComputeAEROopenvsp
from ..components.compute_cnbeta_fuselage import ComputeCnBetaFuselage
from ..components.compute_cl_max import ComputeMaxCL
from ..components.high_lift_aero import ComputeDeltaHighLift
from ..components.compute_L_D_max import ComputeLDMax
from ..components.compute_reynolds import ComputeUnitReynolds
from ..constants import SPAN_MESH_POINT_OPENVSP
from ..aerodynamics_high_speed import AerodynamicsHighSpeed
from ..aerodynamics_low_speed import AerodynamicsLowSpeed
from ...tests.xfoil_exe.get_xfoil import get_xfoil_path
from ...propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from ...propulsion.propulsion import IPropulsion


RESULTS_FOLDER = pth.join(pth.dirname(__file__), "results")
xfoil_path = None if system() == "Windows" else get_xfoil_path()

XML_FILE = "beechcraft_76.xml"
ENGINE_WRAPPER = "test.wrapper.aerodynamics.beechcraft.dummy_engine"


class DummyEngine(AbstractFuelPropulsion):

    def __init__(self):
        """
        Dummy engine model returning nacelle aerodynamic drag force.

        """
        super().__init__()

    def compute_flight_points(self, flight_points: Union[FlightPoint, pd.DataFrame]):
        flight_points.thrust = 0.0
        flight_points['sfc'] = 0.0

    def compute_weight(self) -> float:
        return 0.0

    def compute_dimensions(self) -> (float, float, float, float):
        return [0.0, 0.0, 0.0, 0.0]

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
        pass

    @staticmethod
    def get_model(inputs) -> IPropulsion:
        return DummyEngine()


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


def _test_compute_reynolds():
    """ Tests high and low speed reynolds calculation """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeUnitReynolds()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeUnitReynolds(), ivc)
    mach = problem["data:aerodynamics:cruise:mach"]
    assert mach == pytest.approx(0.2457, abs=1e-4)
    reynolds = problem.get_val("data:aerodynamics:cruise:unit_reynolds", units="m**-1")
    assert reynolds == pytest.approx(4571770, abs=1)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeUnitReynolds(low_speed_aero=True)), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeUnitReynolds(low_speed_aero=True), ivc)
    mach = problem["data:aerodynamics:low_speed:mach"]
    assert mach == pytest.approx(0.1179, abs=1e-4)
    reynolds = problem.get_val("data:aerodynamics:low_speed:unit_reynolds", units="m**-1")
    assert reynolds == pytest.approx(2746999, abs=1)


def _test_cd0_high_speed():
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
    cd0_total = 1.25*(cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac + cd0_lg + cd0_other)
    assert cd0_total == pytest.approx(0.01958, abs=1e-5)


def _test_cd0_low_speed():
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
    cd0_total = 1.25*(cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac + cd0_lg + cd0_other)
    assert cd0_total == pytest.approx(0.04513, abs=1e-5)


def _test_polar():
    """ Tests polar execution (XFOIL) @ high and low speed """

    # Clear saved polar results (for wing and htp airfoils)
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('af', 'csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('af', 'csv')))
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_HTP_AIRFOIL.replace('af', 'csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_HTP_AIRFOIL.replace('af', 'csv')))

    # Define high-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, XML_FILE)
    ivc.add_output("xfoil:mach", 0.245)
    ivc.add_output("xfoil:reynolds", 4571770*1.549)

    # Run problem and check obtained value(s) is/(are) correct
    xfoil_comp = XfoilPolar(
        alpha_start=0.0, alpha_end=25.0, iter_limit=20, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)
    cl = problem["xfoil:CL"]
    cdp = problem["xfoil:CDp"]
    cl, cdp = reshape_polar(cl, cdp)
    cdp_1 = np.interp(1.0, cl, cdp)
    assert cdp_1 == pytest.approx(0.0046, abs=1e-4)

    # Define low-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(list_inputs(XfoilPolar()), __file__, XML_FILE)
    ivc.add_output("xfoil:mach", 0.1179)
    ivc.add_output("xfoil:reynolds", 2746999*1.549)

    # Run problem and check obtained value(s) is/(are) correct
    xfoil_comp = XfoilPolar(
        alpha_start=0.0, alpha_end=25.0, iter_limit=20, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)
    cl = problem["xfoil:CL"]
    cdp = problem["xfoil:CDp"]
    cl_max_2d = problem["xfoil:CL_max_2D"]
    assert cl_max_2d == pytest.approx(1.6965, abs=1e-4)
    cl, cdp = reshape_polar(cl, cdp)
    cdp_1 = np.interp(1.0, cl, cdp)
    assert cdp_1 == pytest.approx(0.0049, abs=1e-4)


def _test_vlm_comp_high_speed():
    """ Tests vlm components @ high speed """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeOSWALDvlm()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.245)
    cl = np.zeros(POLAR_POINT_COUNT)
    cdp = np.zeros(POLAR_POINT_COUNT)
    cl[0:27] = np.array(
        [0.2682, 0.3105, 0.3606, 0.4073, 0.4572, 0.5105, 0.5645, 0.6183,
         0.6714, 0.7237, 0.7762, 0.828, 0.8785, 0.9255, 0.9705, 1.017,
         1.0644, 1.1107, 1.1563, 1.2421, 1.2836, 1.3243, 1.3635, 1.3993,
         1.4349, 1.4657, 1.494]
    )
    cdp[0:27] = np.array(
        [0.00089, 0.00109, 0.00126, 0.00151, 0.00176, 0.00197, 0.00219,
         0.00241, 0.00268, 0.00298, 0.00322, 0.00351, 0.00383, 0.00419,
         0.00479, 0.00531, 0.00575, 0.00621, 0.00675, 0.00823, 0.009,
         0.00989, 0.01088, 0.012, 0.01319, 0.01455, 0.01626]
    )
    ivc.add_output("data:aerodynamics:wing:cruise:CL", cl)
    ivc.add_output("data:aerodynamics:wing:cruise:CDp", cdp)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeOSWALDvlm(), ivc)
    coef_k = problem["data:aerodynamics:aircraft:cruise:induced_drag_coefficient"]
    assert coef_k == pytest.approx(0.0537, abs=1e-4)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeWingCLALPHAvlm()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.245)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingCLALPHAvlm(), ivc)
    cl0 = problem["data:aerodynamics:aircraft:cruise:CL0_clean"]
    assert cl0 == pytest.approx(0.1503, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:aircraft:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(4.819, abs=1e-3)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeHTPCLALPHAvlm()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.245)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTPCLALPHAvlm(), ivc)
    cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_htp == pytest.approx(0.6259, abs=1e-4)


def _test_vlm_comp_low_speed():
    """ Tests vlm components @ high speed """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeOSWALDvlm(low_speed_aero=True)), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)  # correction to compensate old version conversion error
    cl = np.zeros(POLAR_POINT_COUNT)
    cdp = np.zeros(POLAR_POINT_COUNT)
    cl[0:27] = np.array(
        [0.2685, 0.309, 0.3572, 0.4034, 0.4506, 0.5026, 0.5552, 0.608,
         0.6599, 0.7106, 0.7618, 0.8126, 0.8629, 0.9123, 0.9607, 1.0048,
         1.0476, 1.0921, 1.1364, 1.1794, 1.222, 1.2626, 1.3006, 1.3393,
         1.3775, 1.4147, 1.4832]
    )
    cdp[0:27] = np.array(
        [0.00081, 0.00103, 0.0012, 0.00142, 0.0017, 0.0019, 0.00212,
         0.00232, 0.00254, 0.00285, 0.00312, 0.00337, 0.00365, 0.00396,
         0.00431, 0.00484, 0.00538, 0.00574, 0.00617, 0.00664, 0.0071,
         0.00768, 0.00852, 0.00918, 0.00992, 0.01071, 0.01253]
    )
    ivc.add_output("data:aerodynamics:wing:low_speed:CL", cl)
    ivc.add_output("data:aerodynamics:wing:low_speed:CDp", cdp)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeOSWALDvlm(low_speed_aero=True), ivc)
    coef_k = problem["data:aerodynamics:aircraft:low_speed:induced_drag_coefficient"]
    assert coef_k == pytest.approx(0.0531, abs=1e-4)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeWingCLALPHAvlm(low_speed_aero=True)), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)  # correction to compensate old version conversion error

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingCLALPHAvlm(low_speed_aero=True), ivc)
    cl0 = problem["data:aerodynamics:aircraft:low_speed:CL0_clean"]
    assert cl0 == pytest.approx(0.1467, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:aircraft:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(4.703, abs=1e-3)
    y_interp = problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", units="m")
    cl_interp = problem["data:aerodynamics:wing:low_speed:CL_vector"]
    y_interp, cl_interp = reshape_curve(y_interp, cl_interp)
    cl_med = np.interp((y_interp[0] + y_interp[-1]) / 2.0, y_interp, cl_interp)
    assert cl_med == pytest.approx(0.1673, abs=1e-4)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeHTPCLALPHAvlm(low_speed_aero=True)), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)  # correction to compensate old version conversion error

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTPCLALPHAvlm(low_speed_aero=True), ivc)
    cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_htp == pytest.approx(0.6200, abs=1e-4)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeHTPCLCMvlm()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)  # correction to compensate old version conversion error

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTPCLCMvlm(), ivc)
    alpha_interp = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:alpha", units="deg")
    cl_interp = problem["data:aerodynamics:horizontal_tail:low_speed:CL"]
    cm1_interp = problem["data:aerodynamics:horizontal_tail:low_speed:CM"]
    cm2_interp = problem["data:aerodynamics:wing:low_speed:CM"]
    cl = np.interp(5.0, alpha_interp, cl_interp)
    cm1 = np.interp(5.0, alpha_interp, cm1_interp)
    cm2 = np.interp(5.0, alpha_interp, cm2_interp)
    assert cl == pytest.approx(0.0425, abs=1e-4)
    assert cm1 == pytest.approx(-0.0005, abs=1e-4)
    assert cm2 == pytest.approx(-0.0667, abs=1e-4)


def _test_vlm_comp_low_speed_new():
    """ Tests vlm components @ high speed """

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Clear saved polar results (for wing and htp airfoils)
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('af', 'csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('af', 'csv')))
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_HTP_AIRFOIL.replace('af', 'csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_HTP_AIRFOIL.replace('af', 'csv')))

    with Timer(name="VLM low-speed [NEW]: 1st run"):
        # Research independent input value in .xml file
        # noinspection PyTypeChecker
        ivc = get_indep_var_comp(list_inputs(ComputeAEROvlm(low_speed_aero=True)), __file__, XML_FILE)

        # Run problem and check obtained value(s) is/(are) correct
        # noinspection PyTypeChecker
        problem = run_system(ComputeAEROvlm(low_speed_aero=True, result_folder_path=results_folder.name), ivc)
        cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1")
        assert cl_alpha_htp == pytest.approx(0.6201, abs=1e-4)
        cl0_htp = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL0")
        assert cl0_htp == pytest.approx(-0.0116, abs=1e-4)

    with Timer(name="VLM low-speed [NEW]: 2nd run"):
        # Run problem 2nd time
        # noinspection PyTypeChecker
        run_system(ComputeAEROvlm(low_speed_aero=True, result_folder_path=results_folder.name), ivc)



def _test_openvsp_comp_high_speed():
    """ Tests openvsp components @ high speed """

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeOSWALDopenvsp()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.245)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeOSWALDopenvsp(result_folder_path=results_folder.name), ivc)
    coef_k = problem["data:aerodynamics:aircraft:cruise:induced_drag_coefficient"]
    assert coef_k == pytest.approx(0.0488, abs=1e-4)
    assert pth.exists(pth.join(results_folder.name, 'OSWALD'))

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeWingCLALPHAopenvsp()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.245)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingCLALPHAopenvsp(result_folder_path=results_folder.name), ivc)
    cl0 = problem["data:aerodynamics:aircraft:cruise:CL0_clean"]
    assert cl0 == pytest.approx(0.11686, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:aircraft:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(4.535, abs=1e-3)
    assert pth.exists(pth.join(results_folder.name, 'ClAlphaWING'))

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeHTPCLALPHAopenvsp()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.245)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTPCLALPHAopenvsp(result_folder_path=results_folder.name), ivc)
    cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_htp == pytest.approx(0.7030, abs=1e-4)
    assert pth.exists(pth.join(results_folder.name, 'ClAlphaHT'))

    # Remove existing result files
    results_folder.cleanup()


def _test_openvsp_comp_low_speed():
    """ Tests openvsp components @ low speed """

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    with Timer(name="Openvsp low-speed"):
        # Research independent input value in .xml file
        ivc = get_indep_var_comp(list_inputs(ComputeOSWALDopenvsp()), __file__, XML_FILE)
        ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)  # compensate old version conversion error

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(ComputeOSWALDopenvsp(low_speed_aero=True), ivc)
        coef_k = problem["data:aerodynamics:aircraft:low_speed:induced_drag_coefficient"]
        assert coef_k == pytest.approx(0.0487, abs=1e-4)

        # Research independent input value in .xml file
        ivc = get_indep_var_comp(list_inputs(ComputeWingCLALPHAopenvsp(low_speed_aero=True)), __file__, XML_FILE)
        ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)  # compensate old version conversion error

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(ComputeWingCLALPHAopenvsp(low_speed_aero=True), ivc)
        cl0 = problem["data:aerodynamics:aircraft:low_speed:CL0_clean"]
        assert cl0 == pytest.approx(0.1147, abs=1e-4)
        cl_alpha_wing = problem.get_val("data:aerodynamics:aircraft:low_speed:CL_alpha", units="rad**-1")
        assert cl_alpha_wing == pytest.approx(4.457, abs=1e-3)
        y_interp = problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", units="m")
        cl_interp = problem["data:aerodynamics:wing:low_speed:CL_vector"]
        y_interp, cl_interp = reshape_curve(y_interp, cl_interp)
        cl_med = np.interp((y_interp[0]+y_interp[-1])/2.0, y_interp, cl_interp)
        assert cl_med == pytest.approx(0.1216, abs=1e-4)

        # Research independent input value in .xml file
        ivc = get_indep_var_comp(list_inputs(ComputeHTPCLALPHAopenvsp(low_speed_aero=True)), __file__, XML_FILE)
        ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)  # compensate old version conversion error

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(ComputeHTPCLALPHAopenvsp(low_speed_aero=True), ivc)
        cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1")
        assert cl_alpha_htp == pytest.approx(0.6967, abs=1e-4)

        # Research independent input value in .xml file
        ivc = get_indep_var_comp(list_inputs(ComputeHTPCLCMopenvsp()), __file__, XML_FILE)
        ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)  # compensate old version conversion error

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(ComputeHTPCLCMopenvsp(result_folder_path=results_folder.name), ivc)
        alpha_interp = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:alpha", units="deg")
        cl_interp = problem["data:aerodynamics:horizontal_tail:low_speed:CL"]
        cm1_interp = problem["data:aerodynamics:horizontal_tail:low_speed:CM"]
        cm2_interp = problem["data:aerodynamics:wing:low_speed:CM"]
        cl = np.interp(5.0, alpha_interp, cl_interp)
        cm1 = np.interp(5.0, alpha_interp, cm1_interp)
        cm2 = np.interp(5.0, alpha_interp, cm2_interp)
        assert cl == pytest.approx(0.0541, abs=1e-4)
        assert cm1 == pytest.approx(-0.1559, abs=1e-4)
        assert cm2 == pytest.approx(-0.0037, abs=1e-4)
        assert pth.exists(pth.join(results_folder.name, 'ClCmHT'))

    # Remove existing result files
    results_folder.cleanup()


def _test_openvsp_comp_low_speed_new():
    """ Tests openvsp components @ low speed """

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    with Timer(name="Openvsp low-speed [NEW]: 1st run"):
        # Research independent input value in .xml file
        # noinspection PyTypeChecker
        ivc = get_indep_var_comp(list_inputs(ComputeAEROopenvsp(low_speed_aero=True)), __file__, XML_FILE)

        # Run problem and check obtained value(s) is/(are) correct
        # noinspection PyTypeChecker
        problem = run_system(ComputeAEROopenvsp(low_speed_aero=True, result_folder_path=results_folder.name), ivc)
        cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1")
        assert cl_alpha_htp == pytest.approx(0.6760, abs=1e-4)

    with Timer(name="Openvsp low-speed [NEW]: 2nd run"):
        # Run problem 2nd time
        # noinspection PyTypeChecker
        run_system(ComputeAEROopenvsp(low_speed_aero=True, result_folder_path=results_folder.name), ivc)


def _test_high_lift():
    """ Tests high-lift contribution """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeDeltaHighLift()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)  # correction to compensate old version conversion error
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_alpha", 4.569, units="rad**-1")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeDeltaHighLift(), ivc)
    delta_cl0_landing = problem["data:aerodynamics:flaps:landing:CL"]
    assert delta_cl0_landing == pytest.approx(0.7321, abs=1e-4)
    delta_clmax_landing = problem["data:aerodynamics:flaps:landing:CL_max"]
    assert delta_clmax_landing == pytest.approx(0.5788, abs=1e-4)
    delta_cm_landing = problem["data:aerodynamics:flaps:landing:CM"]
    assert delta_cm_landing == pytest.approx(-0.0988, abs=1e-4)
    delta_cd_landing = problem["data:aerodynamics:flaps:landing:CD"]
    assert delta_cd_landing == pytest.approx(0.0196, abs=1e-4)
    delta_cl0_takeoff = problem["data:aerodynamics:flaps:takeoff:CL"]
    assert delta_cl0_takeoff == pytest.approx(0.2805, abs=1e-4)
    delta_clmax_takeoff = problem["data:aerodynamics:flaps:takeoff:CL_max"]
    assert delta_clmax_takeoff == pytest.approx(0.1218, abs=1e-4)
    delta_cm_takeoff = problem["data:aerodynamics:flaps:takeoff:CM"]
    assert delta_cm_takeoff == pytest.approx(-0.0378, abs=1e-4)
    delta_cd_takeoff = problem["data:aerodynamics:flaps:takeoff:CD"]
    assert delta_cd_takeoff == pytest.approx(0.0034, abs=1e-4)
    cl_alpha_elev = problem.get_val("data:aerodynamics:elevator:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_elev == pytest.approx(0.6167, abs=1e-4)


def _test_max_cl():
    """ Tests maximum cl component with Openvsp and VLM results"""

    # Clear saved polar results (for wing and htp airfoils)
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('af', 'csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('af', 'csv')))
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_HTP_AIRFOIL.replace('af', 'csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_HTP_AIRFOIL.replace('af', 'csv')))

    with Timer(name="Maximum CL: run"):
        # Research independent input value in .xml file for Openvsp test
        ivc = get_indep_var_comp(list_inputs(ComputeMaxCL()), __file__, XML_FILE)
        y_openvsp = np.zeros(SPAN_MESH_POINT_OPENVSP)
        cl_openvsp = np.zeros(SPAN_MESH_POINT_OPENVSP)
        y_openvsp[0:39] = [0.04279, 0.12836, 0.21393, 0.2995, 0.38507, 0.47064, 0.55621,
                           0.68649, 0.862, 1.0385, 1.21588, 1.39404, 1.57287, 1.75226,
                           1.93212, 2.11231, 2.29274, 2.47328, 2.65384, 2.83429, 3.01452,
                           3.19443, 3.37389, 3.5528, 3.73106, 3.90855, 4.08517, 4.26082,
                           4.43539, 4.60879, 4.78093, 4.9517, 5.12103, 5.28882, 5.45499,
                           5.61947, 5.78218, 5.94305, 6.10201]
        cl_openvsp[0:39] = [0.0989, 0.09908, 0.09901, 0.09898, 0.09892, 0.09888, 0.09887,
                            0.09871, 0.09823, 0.09859, 0.09894, 0.09888, 0.09837, 0.098,
                            0.0979, 0.09763, 0.09716, 0.09671, 0.0961, 0.09545, 0.09454,
                            0.09377, 0.09295, 0.09209, 0.09087, 0.08965, 0.08812, 0.0866,
                            0.08465, 0.08284, 0.08059, 0.07817, 0.07494, 0.07178, 0.06773,
                            0.06279, 0.05602, 0.04639, 0.03265]
        ivc.add_output("data:aerodynamics:flaps:landing:CL_max", 0.5788)
        ivc.add_output("data:aerodynamics:flaps:takeoff:CL_max", 0.1218)
        ivc.add_output("data:aerodynamics:wing:low_speed:Y_vector", y_openvsp, units="m")
        ivc.add_output("data:aerodynamics:wing:low_speed:CL_vector", cl_openvsp)
        ivc.add_output("data:aerodynamics:wing:low_speed:CL0_clean", 0.0877)
        ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)  # correction to compensate old version ...
        ivc.add_output("data:aerodynamics:low_speed:unit_reynolds", 2613822, units="m**-1")  # correction to ...

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(ComputeMaxCL(), ivc)
        cl_max_clean = problem["data:aerodynamics:wing:low_speed:CL_max_clean"]
        assert cl_max_clean == pytest.approx(1.4992, abs=1e-4)
        cl_max_takeoff = problem["data:aerodynamics:aircraft:takeoff:CL_max"]
        assert cl_max_takeoff == pytest.approx(1.6211, abs=1e-4)
        cl_max_landing = problem["data:aerodynamics:aircraft:landing:CL_max"]
        assert cl_max_landing == pytest.approx(2.0780, abs=1e-2)

    # Research independent input value in .xml file for VLM test
    ivc = get_indep_var_comp(list_inputs(ComputeMaxCL()), __file__, XML_FILE)
    y_vlm = np.zeros(SPAN_MESH_POINT_OPENVSP)
    cl_vlm = np.zeros(SPAN_MESH_POINT_OPENVSP)
    y_vlm[0:17] = [0.09983333, 0.2995, 0.49916667, 0.918, 1.556,
                   2.194, 2.832, 3.47, 4.108, 4.746,
                   5.14475, 5.30425, 5.46375, 5.62325, 5.78275,
                   5.94225, 6.10175]
    cl_vlm[0:17] = [0.14430963, 0.14669887, 0.15259407, 0.16943047, 0.17317354,
                    0.17298787, 0.17043103, 0.1654374, 0.15695794, 0.14205057,
                    0.11684268, 0.10346649, 0.09243006, 0.08182086, 0.07048051,
                    0.05705357, 0.03870218]
    ivc.add_output("data:aerodynamics:flaps:landing:CL_max", 0.5788)
    ivc.add_output("data:aerodynamics:flaps:takeoff:CL_max", 0.1218)
    ivc.add_output("data:aerodynamics:wing:low_speed:Y_vector", y_vlm, units="m")
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_vector", cl_vlm)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL0_clean", 0.1475)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)  # correction to compensate old version conversion error
    ivc.add_output("data:aerodynamics:low_speed:unit_reynolds", 2613822, units="m**-1")  # correction to ...

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMaxCL(), ivc)
    cl_max_clean = problem["data:aerodynamics:wing:low_speed:CL_max_clean"]
    assert cl_max_clean == pytest.approx(1.4405, abs=1e-4)
    cl_max_takeoff = problem["data:aerodynamics:aircraft:takeoff:CL_max"]
    assert cl_max_takeoff == pytest.approx(1.5623, abs=1e-4)
    cl_max_landing = problem["data:aerodynamics:aircraft:landing:CL_max"]
    assert cl_max_landing == pytest.approx(2.0193, abs=1e-4)


def _test_l_d_max():
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


def _test_cnbeta():
    """ Tests cn beta fuselage """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeCnBetaFuselage()), __file__, XML_FILE)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.245)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnBetaFuselage(), ivc)
    cn_beta_fus = problem["data:aerodynamics:fuselage:cruise:CnBeta"]
    assert cn_beta_fus == pytest.approx(-0.0599, abs=1e-4)


def _test_high_speed_connection():
    """ Tests high speed components connection """

    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    register_wrappers()

    with Timer(name="High-speed complete [NEW]: apply VLM"):
        # Run problem with VLM and check obtained value(s) is/(are) correct
        # noinspection PyTypeChecker
        problem = run_system(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=False), input_vars,
                             check=True)
        coef_k = problem["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        assert coef_k == pytest.approx(0.0530, abs=1e-4)
        cl_alpha_wing = problem.get_val("data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1")
        assert cl_alpha_wing == pytest.approx(4.719, abs=1e-3)
        cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1")
        assert cl_alpha_htp == pytest.approx(0.6260, abs=1e-4)
        cl_alpha_vtp = problem.get_val("data:aerodynamics:vertical_tail:cruise:CL_alpha", units="rad**-1")
        assert cl_alpha_vtp == pytest.approx(2.8553, abs=1e-4)

    with Timer(name="High-speed complete [NEW]: apply OPENVSP"):
        # Run problem with OPENVSP and check change(s) is/(are) correct
        # noinspection PyTypeChecker
        problem = run_system(AerodynamicsHighSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=True), input_vars,
                             check=True)
        coef_k = problem["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        assert coef_k == pytest.approx(0.04822, abs=1e-4)
        cl_alpha_wing = problem.get_val("data:aerodynamics:wing:cruise:CL_alpha", units="rad**-1")
        assert cl_alpha_wing == pytest.approx(4.587, abs=1e-3)
        cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1")
        assert cl_alpha_htp == pytest.approx(0.6820, abs=1e-4)
        cl_alpha_vtp = problem.get_val("data:aerodynamics:vertical_tail:cruise:CL_alpha", units="rad**-1")
        assert cl_alpha_vtp == pytest.approx(2.8553, abs=1e-4)


def _test_low_speed_connection():
    """ Tests low speed components connection """

    # Clear saved polar results (for wing and htp airfoils)
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('af', 'csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_WING_AIRFOIL.replace('af', 'csv')))
    if pth.exists(pth.join(resources.__path__[0], DEFAULT_HTP_AIRFOIL.replace('af', 'csv'))):
        os.remove(pth.join(resources.__path__[0], DEFAULT_HTP_AIRFOIL.replace('af', 'csv')))

    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    register_wrappers()

    with Timer(name="Low-speed complete [NEW]: apply VLM"):
        # Run problem with VLM and check obtained value(s) is/(are) correct
        # noinspection PyTypeChecker
        problem = run_system(AerodynamicsLowSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=False), input_vars,
                             check=True)
        coef_k = problem["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
        assert coef_k == pytest.approx(0.0534, abs=1e-4)
        cl_alpha_wing = problem.get_val("data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1")
        assert cl_alpha_wing == pytest.approx(4.607, abs=1e-3)
        cl_max_clean = problem["data:aerodynamics:wing:low_speed:CL_max_clean"]
        assert cl_max_clean == pytest.approx(1.4557, abs=1e-4)
        cl_max_takeoff = problem["data:aerodynamics:aircraft:takeoff:CL_max"]
        assert cl_max_takeoff == pytest.approx(1.5775, abs=1e-4)
        cl_max_landing = problem["data:aerodynamics:aircraft:landing:CL_max"]
        assert cl_max_landing == pytest.approx(2.0345, abs=1e-4)
        cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1")
        assert cl_alpha_htp == pytest.approx(0.6202, abs=1e-4)

    with Timer(name="Low-speed complete [NEW]: apply OPENVSP"):
        # Run problem with OPENVSP and check change(s) is/(are) correct
        # noinspection PyTypeChecker
        problem = run_system(AerodynamicsLowSpeed(propulsion_id=ENGINE_WRAPPER, use_openvsp=True), input_vars,
                             check=True)
        coef_k = problem["data:aerodynamics:wing:low_speed:induced_drag_coefficient"]
        assert coef_k == pytest.approx(0.04821, abs=1e-4)
        cl_alpha_wing = problem.get_val("data:aerodynamics:wing:low_speed:CL_alpha", units="rad**-1")
        assert cl_alpha_wing == pytest.approx(4.5091, abs=1e-3)
        cl_max_clean = problem["data:aerodynamics:wing:low_speed:CL_max_clean"]
        assert cl_max_clean == pytest.approx(1.5319, abs=1e-4)
        cl_max_takeoff = problem["data:aerodynamics:aircraft:takeoff:CL_max"]
        assert cl_max_takeoff == pytest.approx(1.6537, abs=1e-4)
        cl_max_landing = problem["data:aerodynamics:aircraft:landing:CL_max"]
        assert cl_max_landing == pytest.approx(2.1108, abs=1e-4)
        cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1")
        assert cl_alpha_htp == pytest.approx(0.6760, abs=1e-4)


def test_v_n_diagram():

    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    input_vars.add_output("data:aerodynamics:aircraft:landing:CL_max", 1.9)
    input_vars.add_output("data:aerodynamics:wing:low_speed:CL_max_clean", 1.5)
    input_vars.add_output("data:aerodynamics:wing:low_speed:CL_min_clean", -1.5)
    input_vars.add_output("data:weight:aircraft:MTOW", 1700.0, units="kg")
    input_vars.add_output("data:aerodynamics:cruise:mach", 0.4)

    register_wrappers()

    # Run problem with VLM and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    run_system(ComputeVNopenvsp(propulsion_id=ENGINE_WRAPPER), input_vars, check=True)
