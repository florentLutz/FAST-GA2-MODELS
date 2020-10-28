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
import openmdao.api as om
import numpy as np
from platform import system
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from fastoad.io import VariableIO

from fastoad.models.aerodynamics.constants import POLAR_POINT_COUNT
from ...tests.testing_utilities import run_system
from ..components.cd0 import CD0
from ..external.vlm import ComputeOSWALDvlm, ComputeWingCLALPHAvlm
from ..external.xfoil import XfoilPolar
from ..external.openvsp import ComputeOSWALDopenvsp, ComputeWingCLALPHAopenvsp, ComputeHTPCLALPHAopenvsp, \
    ComputeHTPCLCMopenvsp
from ..components.compute_cnbeta_fuselage import ComputeCnBetaFuselage
from ..components.compute_cl_max import ComputeMaxCL
from ..components.high_lift_aero import ComputeDeltaHighLift
from ..components.compute_L_D_max import ComputeLDMax
from ..components.compute_reynolds import ComputeReynolds
from ..constants import SPAN_MESH_POINT_OPENVSP
from ..aerodynamics_high_speed import AerodynamicsHighSpeed
from ..aerodynamics_low_speed import AerodynamicsLowSpeed

from ...tests.xfoil_exe.get_xfoil import get_xfoil_path

RESULTS_FOLDER = pth.join(pth.dirname(__file__), "results")
ERROR_FILE = pth.join(pth.dirname(__file__), "external_code_comp_error.out")
xfoil_path = None if system() == "Windows" else get_xfoil_path()
XML_FILE = "beechcraft_76.xml"


def _create_tmp_directory() -> TemporaryDirectory:
    """Provide temporary directory for calculation."""

    for tmp_base_path in [None, pth.join(str(Path.home()), ".fast")]:
        if tmp_base_path is not None:
            os.makedirs(tmp_base_path, exist_ok=True)
        tmp_directory = tempfile.TemporaryDirectory(prefix="x", dir=tmp_base_path)
        break

    return tmp_directory


def get_indep_var_comp(var_names):
    """ Reads required input data and returns an IndepVarcomp() instance"""
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    ivc = reader.read(only=var_names).to_ivc()
    return ivc


def list_inputs(group):
    """ Reads input variables from a group and return as a list (run model with 0 value can lead to raise configuration
    errors in models)"""
    prob = om.Problem(model=group)
    prob.setup()
    prob.run_model()
    data = prob.model.list_inputs(out_stream=None)
    list_names = []
    for idx in range(len(data)):
        variable_name = data[idx][0].split('.')
        list_names.append(variable_name[len(variable_name) - 1])
    return list(dict.fromkeys(list_names))


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


def test_compute_reynolds():
    """ Tests high and low speed reynolds calculation """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model_hs", ComputeReynolds(), promotes=["*"])
    group.add_subsystem("my_model_ls", ComputeReynolds(low_speed_aero=True), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeReynolds(), ivc)
    mach = problem["data:aerodynamics:cruise:mach"]
    assert mach == pytest.approx(0.2457, abs=1e-4)
    reynolds = problem["data:aerodynamics:cruise:unit_reynolds"]
    assert reynolds == pytest.approx(4571770, abs=1)
    problem = run_system(ComputeReynolds(low_speed_aero=True), ivc)
    mach = problem["data:aerodynamics:low_speed:mach"]
    assert mach == pytest.approx(0.1179, abs=1e-4)
    reynolds = problem["data:aerodynamics:low_speed:unit_reynolds"]
    assert reynolds == pytest.approx(2746999, abs=1)


def test_cd0_high_speed():
    """ Tests drag coefficient @ high speed """

    # Generate input list from model
    group = om.Group()
    ivc = om.IndepVarComp()
    ivc.add_output("data:aerodynamics:cruise:mach", 0.2457)
    ivc.add_output("data:aerodynamics:cruise:unit_reynolds", 4571770)
    group.add_subsystem("constants", ivc, promotes=["*"])
    group.add_subsystem("my_model", CD0(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.2457)
    ivc.add_output("data:aerodynamics:cruise:unit_reynolds", 4571770)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CD0(), ivc)
    cd0_wing = problem["data:aerodynamics:wing:cruise:CD0"]
    assert cd0_wing == pytest.approx(0.00506, abs=1e-5)
    cd0_fus = problem["data:aerodynamics:fuselage:cruise:CD0"]
    assert cd0_fus == pytest.approx(0.00490, abs=1e-5)
    cd0_ht = problem["data:aerodynamics:horizontal_tail:cruise:CD0"]
    assert cd0_ht == pytest.approx(0.00123, abs=1e-5)
    cd0_vt = problem["data:aerodynamics:vertical_tail:cruise:CD0"]
    assert cd0_vt == pytest.approx(0.00077, abs=1e-5)
    cd0_nac = problem["data:aerodynamics:nacelles:cruise:CD0"]
    assert cd0_nac == pytest.approx(0.00202, abs=1e-5)
    cd0_lg = problem["data:aerodynamics:landing_gear:cruise:CD0"]
    assert cd0_lg == pytest.approx(0.0, abs=1e-5)
    cd0_other = problem["data:aerodynamics:other:cruise:CD0"]
    assert cd0_other == pytest.approx(0.00202, abs=1e-5)
    cd0_total = 1.25*(cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac + cd0_lg + cd0_other)
    assert cd0_total == pytest.approx(0.02003, abs=1e-5)


def test_cd0_low_speed():
    """ Tests drag coefficient @ low speed """

    # Generate input list from model
    group = om.Group()
    ivc = om.IndepVarComp()
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1179)
    ivc.add_output("data:aerodynamics:low_speed:unit_reynolds", 2746999)
    group.add_subsystem("constants", ivc, promotes=["*"])
    group.add_subsystem("my_model", CD0(low_speed_aero=True), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)  # correction to compensate old version conversion error
    ivc.add_output("data:aerodynamics:low_speed:unit_reynolds", 2613822)  # correction to ...

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CD0(low_speed_aero=True), ivc)
    cd0_wing = problem["data:aerodynamics:wing:low_speed:CD0"]
    assert cd0_wing == pytest.approx(0.00555, abs=1e-5)
    cd0_fus = problem["data:aerodynamics:fuselage:low_speed:CD0"]
    assert cd0_fus == pytest.approx(0.00547, abs=1e-5)
    cd0_ht = problem["data:aerodynamics:horizontal_tail:low_speed:CD0"]
    assert cd0_ht == pytest.approx(0.00136, abs=1e-5)
    cd0_vt = problem["data:aerodynamics:vertical_tail:low_speed:CD0"]
    assert cd0_vt == pytest.approx(0.00086, abs=1e-5)
    cd0_nac = problem["data:aerodynamics:nacelles:low_speed:CD0"]
    assert cd0_nac == pytest.approx(0.00221, abs=1e-5)
    cd0_lg = problem["data:aerodynamics:landing_gear:low_speed:CD0"]
    assert cd0_lg == pytest.approx(0.01900, abs=1e-5)
    cd0_other = problem["data:aerodynamics:other:low_speed:CD0"]
    assert cd0_other == pytest.approx(0.00202, abs=1e-5)
    cd0_total = 1.25*(cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_nac + cd0_lg + cd0_other)
    assert cd0_total == pytest.approx(0.04562, abs=1e-5)


def test_polar():
    """ Tests polar execution (XFOIL) @ high and low speed """

    # Define high-speed parameters (with .xml file and additional inputs)
    input_list = [
        "data:geometry:wing:thickness_ratio",
        "data:geometry:wing:MAC:length",
    ]
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("xfoil:mach", 0.245)
    ivc.add_output("xfoil:unit_reynolds", 4571770)
    ivc.add_output("xfoil:length", 1.549)  # group connection between data:geometry:wing:MAC:length and xfoil:length

    # Run problem and check obtained value(s) is/(are) correct
    xfoil_comp = XfoilPolar(
        alpha_start=0.0, alpha_end=25.0, iter_limit=20, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)
    cl = problem["xfoil:CL"]
    cdp = problem["xfoil:CDp"]
    cl, cdp = reshape_polar(cl, cdp)
    cdp_1 = np.interp(1.0, cl, cdp)
    assert cdp_1 == pytest.approx(0.0041, abs=1e-4)

    # Define low-speed parameters (with .xml file and additional inputs)
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("xfoil:mach", 0.1179)
    ivc.add_output("xfoil:unit_reynolds", 2746999)
    ivc.add_output("xfoil:length", 1.549)  # group connection between data:geometry:wing:MAC:length and xfoil:length

    # Run problem and check obtained value(s) is/(are) correct
    xfoil_comp = XfoilPolar(
        alpha_start=0.0, alpha_end=25.0, iter_limit=20, xfoil_exe_path=xfoil_path
    )
    problem = run_system(xfoil_comp, ivc)
    cl = problem["xfoil:CL"]
    cdp = problem["xfoil:CDp"]
    cl_max_2d = problem["xfoil:CL_max_2D"]
    assert cl_max_2d == pytest.approx(1.6999, abs=1e-4)
    cl, cdp = reshape_polar(cl, cdp)
    cdp_1 = np.interp(1.0, cl, cdp)
    assert cdp_1 == pytest.approx(0.0047, abs=1e-4)


def test_vlm_comp_high_speed():
    """ Tests vlm components @ high speed """

    # Input list from model (not generated because NaN values not supported by vspcript/vspaero)
    input_list = [
        "data:TLAR:v_cruise",
        "data:geometry:wing:area",
        "data:geometry:wing:span",
        "data:geometry:wing:aspect_ratio",
        "data:geometry:wing:kink:span_ratio",
        "data:geometry:wing:MAC:length",
        "data:geometry:wing:root:y",
        "data:geometry:wing:root:chord",
        "data:geometry:wing:tip:chord",
        "data:geometry:flap:span_ratio",
        "data:geometry:horizontal_tail:span",
        "data:geometry:horizontal_tail:root:chord",
        "data:geometry:horizontal_tail:tip:chord",
        "data:geometry:fuselage:maximum_width",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
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
    assert coef_k == pytest.approx(0.0531, abs=1e-4)
    problem = run_system(ComputeWingCLALPHAvlm(), ivc)
    cl0 = problem["data:aerodynamics:aircraft:cruise:CL0_clean"]
    assert cl0 == pytest.approx(0.1511, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:aircraft:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(4.819, abs=1e-3)


def test_vlm_comp_low_speed():
    """ Tests vlm components @ high speed """

    # Input list from model (not generated because NaN values not supported by vspcript/vspaero)
    input_list = [
        "data:geometry:wing:area",
        "data:geometry:wing:span",
        "data:geometry:wing:aspect_ratio",
        "data:geometry:wing:kink:span_ratio",
        "data:geometry:wing:MAC:length",
        "data:geometry:wing:root:y",
        "data:geometry:wing:root:chord",
        "data:geometry:wing:tip:chord",
        "data:geometry:flap:span_ratio",
        "data:geometry:horizontal_tail:span",
        "data:geometry:horizontal_tail:root:chord",
        "data:geometry:horizontal_tail:tip:chord",
        "data:geometry:fuselage:maximum_width",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
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
    assert coef_k == pytest.approx(0.0528, abs=1e-4)
    problem = run_system(ComputeWingCLALPHAvlm(low_speed_aero=True), ivc)
    cl0 = problem["data:aerodynamics:aircraft:low_speed:CL0_clean"]
    assert cl0 == pytest.approx(0.1475, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:aircraft:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(4.703, abs=1e-3)
    y_interp = problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", units="m")
    cl_interp = problem["data:aerodynamics:wing:low_speed:CL_vector"]
    y_interp, cl_interp = reshape_curve(y_interp, cl_interp)
    cl_med = np.interp((y_interp[0] + y_interp[-1]) / 2.0, y_interp, cl_interp)
    assert cl_med == pytest.approx(0.1683, abs=1e-4)


def test_openvsp_comp_high_speed():
    """ Tests openvsp components @ high speed """

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Input list from model (not generated because NaN values not supported by vspcript/vspaero)
    input_list = [
        "data:mission:sizing:main_route:cruise:altitude",
        "data:geometry:wing:MAC:leading_edge:x:local",
        "data:geometry:wing:MAC:length",
        "data:geometry:fuselage:maximum_width",
        "data:geometry:wing:root:y",
        "data:geometry:wing:root:chord",
        "data:geometry:wing:tip:y",
        "data:geometry:wing:tip:chord",
        "data:geometry:wing:sweep_0",
        "data:geometry:wing:MAC:at25percent:x",
        "data:geometry:wing:area",
        "data:geometry:wing:span",
        "data:geometry:fuselage:maximum_height",
        "data:geometry:horizontal_tail:sweep_25",
        "data:geometry:horizontal_tail:span",
        "data:geometry:horizontal_tail:root:chord",
        "data:geometry:horizontal_tail:tip:chord",
        "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
        "data:geometry:horizontal_tail:MAC:length",
        "data:geometry:horizontal_tail:MAC:at25percent:x:local",
        "data:geometry:horizontal_tail:z:from_wingMAC25",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.245)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeOSWALDopenvsp(result_folder_path=results_folder.name), ivc)
    coef_k = problem["data:aerodynamics:aircraft:cruise:induced_drag_coefficient"]
    assert coef_k == pytest.approx(0.0480, abs=1e-4)
    assert pth.exists(pth.join(results_folder.name, 'OSWALD'))
    problem = run_system(ComputeWingCLALPHAopenvsp(result_folder_path=results_folder.name), ivc)
    cl0 = problem["data:aerodynamics:aircraft:cruise:CL0_clean"]
    assert cl0 == pytest.approx(0.0906, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:aircraft:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(4.650, abs=1e-3)
    assert pth.exists(pth.join(results_folder.name, 'ClAlphaWING'))
    problem = run_system(ComputeHTPCLALPHAopenvsp(result_folder_path=results_folder.name), ivc)
    cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_htp == pytest.approx(0.7027, abs=1e-4)
    assert pth.exists(pth.join(results_folder.name, 'ClAlphaHT'))

    # Remove existing result files
    results_folder.cleanup()


def test_openvsp_comp_low_speed():
    """ Tests openvsp components @ low speed """

    # Create result temporary directory
    results_folder = _create_tmp_directory()

    # Input list from model (not generated because NaN values not supported by vspcript/vspaero)
    input_list = [
        "data:geometry:wing:MAC:leading_edge:x:local",
        "data:geometry:wing:MAC:length",
        "data:geometry:fuselage:maximum_width",
        "data:geometry:wing:root:y",
        "data:geometry:wing:root:chord",
        "data:geometry:wing:tip:y",
        "data:geometry:wing:tip:chord",
        "data:geometry:wing:sweep_0",
        "data:geometry:wing:MAC:at25percent:x",
        "data:geometry:wing:area",
        "data:geometry:wing:span",
        "data:geometry:fuselage:maximum_height",
        "data:geometry:horizontal_tail:sweep_25",
        "data:geometry:horizontal_tail:span",
        "data:geometry:horizontal_tail:root:chord",
        "data:geometry:horizontal_tail:tip:chord",
        "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
        "data:geometry:horizontal_tail:MAC:length",
        "data:geometry:horizontal_tail:MAC:at25percent:x:local",
        "data:geometry:horizontal_tail:z:from_wingMAC25",

    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)  # correction to compensate old version conversion error

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeOSWALDopenvsp(low_speed_aero=True), ivc)
    coef_k = problem["data:aerodynamics:aircraft:low_speed:induced_drag_coefficient"]
    assert coef_k == pytest.approx(0.0480, abs=1e-4)
    problem = run_system(ComputeWingCLALPHAopenvsp(low_speed_aero=True), ivc)
    cl0 = problem["data:aerodynamics:aircraft:low_speed:CL0_clean"]
    assert cl0 == pytest.approx(0.0889, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:aircraft:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(4.569, abs=1e-3)
    y_interp = problem.get_val("data:aerodynamics:wing:low_speed:Y_vector", units="m")
    cl_interp = problem["data:aerodynamics:wing:low_speed:CL_vector"]
    y_interp, cl_interp = reshape_curve(y_interp, cl_interp)
    cl_med = np.interp((y_interp[0]+y_interp[-1])/2.0, y_interp, cl_interp)
    assert cl_med == pytest.approx(0.0950, abs=1e-4)
    problem = run_system(ComputeHTPCLALPHAopenvsp(low_speed_aero=True), ivc)
    cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_htp == pytest.approx(0.696, abs=1e-3)
    problem = run_system(ComputeHTPCLCMopenvsp(result_folder_path=results_folder.name), ivc)
    alpha_interp = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:alpha", units="deg")
    cl_interp = problem["data:aerodynamics:horizontal_tail:low_speed:CL"]
    cm_interp = problem["data:aerodynamics:horizontal_tail:low_speed:CM"]
    cl = np.interp(10.0, alpha_interp, cl_interp)
    cm = np.interp(10.0, alpha_interp, cm_interp)
    assert cl == pytest.approx(0.1127, abs=1e-4)
    assert cm == pytest.approx(-0.3292, abs=1e-4)
    assert pth.exists(pth.join(results_folder.name, 'ClCmHT'))

    # Remove existing result files
    results_folder.cleanup()


def test_high_lift():
    """ Tests high-lift contribution """

    # Input list from model (not generated because NaN values not supported by interpolation function)
    input_list = [
        "data:geometry:wing:span",
        "data:geometry:wing:area",
        "data:geometry:horizontal_tail:area",
        "data:geometry:wing:taper_ratio",
        "data:geometry:fuselage:maximum_width",
        "data:geometry:wing:root:y",
        "data:geometry:wing:root:chord",
        "data:geometry:flap:chord_ratio",
        "data:geometry:flap:span_ratio",
        "data:geometry:flap_type",
        "data:aerodynamics:aircraft:low_speed:CL_alpha",
        "data:aerodynamics:low_speed:mach",
        "data:mission:sizing:landing:flap_angle",
        "data:mission:sizing:takeoff:flap_angle",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1149)  # correction to compensate old version conversion error
    ivc.add_output("data:aerodynamics:aircraft:low_speed:CL_alpha", 4.569)

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
    angle_interp = problem.get_val("data:aerodynamics:elevator:low_speed:angle", units="deg")
    cl_interp = problem["data:aerodynamics:elevator:low_speed:CL"]
    delta_cl_elevator = np.interp(10.0, angle_interp, cl_interp)
    assert delta_cl_elevator == pytest.approx(9.277, abs=1e-3)


def test_max_cl():
    """ Tests maximum cl component with Openvsp and VLM results"""

    # Input list from model (not generated because NaN values not supported by interpolation function)
    input_list = [
        "data:geometry:wing:root:y",
        "data:geometry:wing:tip:y",
    ]

    # Research independent input value in .xml file for Openvsp test
    ivc = get_indep_var_comp(input_list)
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
    ivc.add_output("data:aerodynamics:wing:low_speed:Y_vector", y_openvsp)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_vector", cl_openvsp)
    ivc.add_output("data:aerodynamics:wing:low_speed:root:CL_max_2D", 1.6999)
    ivc.add_output("data:aerodynamics:wing:low_speed:tip:CL_max_2D", 1.6999)
    ivc.add_output("data:aerodynamics:aircraft:low_speed:CL0_clean", 0.0877)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMaxCL(), ivc)
    cl_max_clean = problem["data:aerodynamics:wing:low_speed:CL_max_clean"]
    assert cl_max_clean == pytest.approx(1.5070, abs=1e-4)
    cl_max_takeoff = problem["data:aerodynamics:aircraft:takeoff:CL_max"]
    assert cl_max_takeoff == pytest.approx(1.6288, abs=1e-4)
    cl_max_landing = problem["data:aerodynamics:aircraft:landing:CL_max"]
    assert cl_max_landing == pytest.approx(2.0858, abs=1e-2)

    # Research independent input value in .xml file for VLM test
    ivc = get_indep_var_comp(input_list)
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
    ivc.add_output("data:aerodynamics:wing:low_speed:Y_vector", y_vlm)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_vector", cl_vlm)
    ivc.add_output("data:aerodynamics:wing:low_speed:root:CL_max_2D", 1.6999)
    ivc.add_output("data:aerodynamics:wing:low_speed:tip:CL_max_2D", 1.6999)
    ivc.add_output("data:aerodynamics:aircraft:low_speed:CL0_clean", 0.1475)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMaxCL(), ivc)
    cl_max_clean = problem["data:aerodynamics:wing:low_speed:CL_max_clean"]
    assert cl_max_clean == pytest.approx(1.4480, abs=1e-4)
    cl_max_takeoff = problem["data:aerodynamics:aircraft:takeoff:CL_max"]
    assert cl_max_takeoff == pytest.approx(1.5698, abs=1e-4)
    cl_max_landing = problem["data:aerodynamics:aircraft:landing:CL_max"]
    assert cl_max_landing == pytest.approx(2.0268, abs=1e-4)


def test_l_d_max():
    """ Tests best lift/drag component """

    # Define independent input value (openVSP)
    ivc = om.IndepVarComp()
    ivc.add_output("data:aerodynamics:aircraft:cruise:CL0_clean", 0.0906)
    ivc.add_output("data:aerodynamics:aircraft:cruise:CL_alpha", 4.650)
    ivc.add_output("data:aerodynamics:aircraft:cruise:CD0", 0.01603)
    ivc.add_output("data:aerodynamics:aircraft:cruise:induced_drag_coefficient", 0.0480)

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

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("cnbeta_fus", ComputeCnBetaFuselage(), promotes=["*"])
    input_list = list_inputs(group)
    # print(input_list)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.245)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnBetaFuselage(), ivc)
    cn_beta_fus = problem["data:aerodynamics:fuselage:cruise:CnBeta"]
    assert cn_beta_fus == pytest.approx(-0.0599, abs=1e-4)


def test_high_speed_connection():
    """ Tests high speed components connection """

    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    problem = run_system(AerodynamicsHighSpeed(), input_vars)
    cd0 = problem["data:aerodynamics:aircraft:cruise:CD0"]
    assert cd0 == pytest.approx(0.0200, abs=1e-4)
    coef_k = problem["data:aerodynamics:aircraft:cruise:induced_drag_coefficient"]
    assert coef_k == pytest.approx(0.0480, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:aircraft:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(4.650, abs=1e-3)
    cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_htp == pytest.approx(0.7030, abs=1e-4)
    cl_alpha_vtp = problem.get_val("data:aerodynamics:vertical_tail:cruise:CL_alpha", units="rad**-1")
    assert cl_alpha_vtp == pytest.approx(1.9564, abs=1e-4)


def test_low_speed_connection():
    """ Tests low speed components connection """

    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    problem = run_system(AerodynamicsLowSpeed(), input_vars)
    cl_max_clean = problem["data:aerodynamics:wing:low_speed:CL_max_clean"]
    assert cl_max_clean == pytest.approx(1.5172, abs=1e-4)
    cl_max_takeoff = problem["data:aerodynamics:aircraft:takeoff:CL_max"]
    assert cl_max_takeoff == pytest.approx(1.6391, abs=1e-4)
    cl_max_landing = problem["data:aerodynamics:aircraft:landing:CL_max"]
    assert cl_max_landing == pytest.approx(2.0961, abs=1e-4)
    cd0 = problem["data:aerodynamics:aircraft:low_speed:CD0"]
    assert cd0 == pytest.approx(0.0454, abs=1e-4)
    coef_k = problem["data:aerodynamics:aircraft:low_speed:induced_drag_coefficient"]
    assert coef_k == pytest.approx(0.0480, abs=1e-4)
    cl_alpha_wing = problem.get_val("data:aerodynamics:aircraft:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_wing == pytest.approx(4.570, abs=1e-3)
    cl_alpha_htp = problem.get_val("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1")
    assert cl_alpha_htp == pytest.approx(0.6967, abs=1e-4)

    # Remove error file
    if os.path.exists(ERROR_FILE):
        os.remove(ERROR_FILE)
