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
import shutil
import openmdao.api as om
import numpy as np

import pytest
from fastoad.io import VariableIO

from pytest import approx
from fastoad.models.aerodynamics.constants import POLAR_POINT_COUNT
from tests.testing_utilities import run_system
from ..components.cd0 import CD0
from ..external.vlm import ComputeOSWALDvlm, ComputeWingCLALPHAvlm
from ..external.xfoil import XfoilPolar
from ..external.openvsp import ComputeOSWALDopenvsp, ComputeWingCLALPHAopenvsp, ComputeHTPCLALPHAopenvsp, ComputeHTPCLCMopenvsp
from ..components.compute_cnbeta_fuselage import ComputeCnBetaFuselage
from ..components.compute_cnbeta_vt import ComputeCnBetaVT
from ..components.compute_cl_max import ComputeMaxCL
from ..components.high_lift_aero import ComputeDeltaHighLift
from ..components.compute_L_D_max import ComputeLDMax
from ..components.compute_reynolds import ComputeReynolds

OPENVSP_RESULTS = pth.join(pth.dirname(__file__), "results")

def get_indep_var_comp(var_names):
    """ Reads required input data and returns an IndepVarcomp() instance"""
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", "aerodynamics_inputs.xml"))
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


def reshape_data(alpha, cl, cd, cdp, cm):
    """ Reshape data from xfoil polar vectors """
    for idx in range(len(alpha)):
        if np.sum(alpha[idx:len(alpha)] == 0) == (len(alpha) - idx):
            alpha = alpha[0:idx]
            cl = cl[0:idx]
            cd = cd[0:idx]
            cdp = cdp[0:idx]
            cm = cm[0:idx]
            break

    return alpha, cl, cd, cdp, cm


def test_compute_reynolds():
    """ Tests high and low speed reynolds calculation """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeReynolds(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeReynolds(), ivc)
    mach = problem["data:aerodynamics:cruise:mach"]
    assert mach == pytest.approx(0.245, abs=1e-3)
    reynolds = problem["data:aerodynamics:wing:cruise:reynolds"]
    assert reynolds == pytest.approx(4571770, abs=1)
    problem = run_system(ComputeReynolds(low_speed_aero=True), ivc)
    reynolds = problem["data:aerodynamics:wing:low_speed:reynolds"]
    assert reynolds == pytest.approx(2329600, abs=1)


def test_cd0_high_speed():
    """ Tests drag coefficient @ high speed """

    # Generate input list from model
    group = om.Group()
    ivc = om.IndepVarComp()
    ivc.add_output("data:aerodynamics:cruise:mach", 0.245)
    ivc.add_output("data:aerodynamics:wing:cruise:reynolds", 4571770)
    group.add_subsystem("constants", ivc, promotes=["*"])
    group.add_subsystem("my_model", CD0(), promotes=["*"])
    input_list = list_inputs(group)
    #print(input_list)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.245)
    ivc.add_output("data:aerodynamics:wing:cruise:reynolds", 4571770)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CD0(), ivc)
    cd0_wing = problem["data:aerodynamics:wing:cruise:CD0"]
    assert cd0_wing == pytest.approx(0.00561, abs=1e-5)
    cd0_fus = problem["data:aerodynamics:fuselage:cruise:CD0"]
    assert cd0_fus == pytest.approx(0.00534, abs=1e-5)
    cd0_ht = problem["data:aerodynamics:horizontal_tail:cruise:CD0"]
    assert cd0_ht == pytest.approx(0.00137, abs=1e-5)
    cd0_vt = problem["data:aerodynamics:vertical_tail:cruise:CD0"]
    assert cd0_vt == pytest.approx(0.00086, abs=1e-5)
    cd0_nac = problem["data:aerodynamics:nacelles:cruise:CD0"]
    assert cd0_nac == pytest.approx(0.00216, abs=1e-5)
    cd0_lg = problem["data:aerodynamics:landing_gear:cruise:CD0"]
    assert cd0_lg == pytest.approx(0.0, abs=1e-5)
    cd0_other = problem["data:aerodynamics:other:cruise:CD0"]
    assert cd0_other == pytest.approx(0.00202, abs=1e-5)


def test_cd0_low_speed():
    """ Tests drag coefficient @ low speed """

    # Generate input list from model
    group = om.Group()
    ivc = om.IndepVarComp()
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1)
    ivc.add_output("data:aerodynamics:wing:low_speed:reynolds", 2329600)
    group.add_subsystem("constants", ivc, promotes=["*"])
    group.add_subsystem("my_model", CD0(low_speed_aero=True), promotes=["*"])
    input_list = list_inputs(group)
    #print(input_list)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.1)
    ivc.add_output("data:aerodynamics:wing:low_speed:reynolds", 2329600)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CD0(low_speed_aero=True), ivc)
    cd0_wing = problem["data:aerodynamics:wing:low_speed:CD0"]
    assert cd0_wing == pytest.approx(0.00636, abs=1e-5)
    cd0_fus = problem["data:aerodynamics:fuselage:low_speed:CD0"]
    assert cd0_fus == pytest.approx(0.00610, abs=1e-5)
    cd0_ht = problem["data:aerodynamics:horizontal_tail:low_speed:CD0"]
    assert cd0_ht == pytest.approx(0.00156, abs=1e-5)
    cd0_vt = problem["data:aerodynamics:vertical_tail:low_speed:CD0"]
    assert cd0_vt == pytest.approx(0.00098, abs=1e-5)
    cd0_nac = problem["data:aerodynamics:nacelles:low_speed:CD0"]
    assert cd0_nac == pytest.approx(0.00241, abs=1e-5)
    cd0_lg = problem["data:aerodynamics:landing_gear:low_speed:CD0"]
    assert cd0_lg == pytest.approx(0.01900, abs=1e-5)
    cd0_other = problem["data:aerodynamics:other:low_speed:CD0"]
    assert cd0_other == pytest.approx(0.00202, abs=1e-5)


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
    ivc.add_output("data:aerodynamics:cruise:mach", 0.8)
    alpha = np.zeros(POLAR_POINT_COUNT)
    cl = np.zeros(POLAR_POINT_COUNT)
    cdp = np.zeros(POLAR_POINT_COUNT)
    alpha[0:4] = np.array([0.0, 5.0, 10.0, 15.0])
    cl[0:4] = np.array([0.4, 0.8, 1.1, 1.35])
    cdp[0:4] = np.array([0.01, 0.012, 0.022, 0.06])
    ivc.add_output("data:aerodynamics:wing:cruise:alpha", alpha)
    ivc.add_output("data:aerodynamics:wing:cruise:CL", cl)
    ivc.add_output("data:aerodynamics:wing:cruise:CDp", cdp)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeOSWALDvlm(), ivc)
    coef_k = problem["data:aerodynamics:aircraft:cruise:induced_drag_coefficient"]
    assert coef_k == pytest.approx(0.068, abs=1e-2)
    problem = run_system(ComputeWingCLALPHAvlm(), ivc)
    cl0 = problem["data:aerodynamics:aircraft:cruise:CL0_clean"]
    assert cl0 == pytest.approx(0.4, abs=1e-3)
    cl_alpha_wing = problem["data:aerodynamics:aircraft:cruise:CL_alpha"]
    assert cl_alpha_wing == pytest.approx(9.31, abs=1e-2)


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
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.2)
    alpha = np.zeros(POLAR_POINT_COUNT)
    cl = np.zeros(POLAR_POINT_COUNT)
    cdp = np.zeros(POLAR_POINT_COUNT)
    alpha[0:4] = np.array([0.0, 5.0, 10.0, 15.0])
    cl[0:4] = np.array([0.4, 0.8, 1.1, 1.35])
    cdp[0:4] = np.array([0.01, 0.012, 0.022, 0.06])
    ivc.add_output("data:aerodynamics:wing:low_speed:alpha", alpha)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL", cl)
    ivc.add_output("data:aerodynamics:wing:low_speed:CDp", cdp)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeOSWALDvlm(low_speed_aero=True), ivc)
    coef_k = problem["data:aerodynamics:aircraft:low_speed:induced_drag_coefficient"]
    assert coef_k == pytest.approx(0.068, abs=1e-2)
    problem = run_system(ComputeWingCLALPHAvlm(low_speed_aero=True), ivc)
    cl0 = problem["data:aerodynamics:aircraft:low_speed:CL0_clean"]
    assert cl0 == pytest.approx(0.4, abs=1e-3)
    cl_alpha_wing = problem["data:aerodynamics:aircraft:low_speed:CL_alpha"]
    assert cl_alpha_wing == pytest.approx(5.70, abs=1e-2)


def test_openvsp_comp_high_speed():
    """ Tests openvsp components @ high speed """

    # Remove existing result files
    if pth.exists(OPENVSP_RESULTS):
        shutil.rmtree(OPENVSP_RESULTS)

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
    ivc.add_output("data:aerodynamics:cruise:mach", 0.8)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeOSWALDopenvsp(result_folder_path=OPENVSP_RESULTS), ivc)
    coef_k = problem["data:aerodynamics:aircraft:cruise:induced_drag_coefficient"]
    assert coef_k == pytest.approx(0.38, abs=1e-2)
    assert pth.exists(pth.join(OPENVSP_RESULTS, 'OSWALD'))
    problem = run_system(ComputeWingCLALPHAopenvsp(result_folder_path=OPENVSP_RESULTS), ivc)
    cl0 = problem["data:aerodynamics:aircraft:cruise:CL0_clean"]
    assert cl0 == pytest.approx(0.016, abs=1e-3)
    cl_alpha_wing = problem["data:aerodynamics:aircraft:cruise:CL_alpha"]
    assert cl_alpha_wing == pytest.approx(0.79, abs=1e-2)
    assert pth.exists(pth.join(OPENVSP_RESULTS, 'ClAlphaWING'))
    problem = run_system(ComputeHTPCLALPHAopenvsp(result_folder_path=OPENVSP_RESULTS), ivc)
    cl_alpha_htp = problem["data:aerodynamics:horizontal_tail:cruise:CL_alpha"]
    assert cl_alpha_htp == pytest.approx(0.0022, abs=1e-3)
    assert pth.exists(pth.join(OPENVSP_RESULTS, 'ClAlphaHT'))

def test_openvsp_comp_low_speed():
    """ Tests openvsp components @ low speed """

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
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.2)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeOSWALDopenvsp(low_speed_aero=True), ivc)
    coef_k = problem["data:aerodynamics:aircraft:low_speed:induced_drag_coefficient"]
    assert coef_k == pytest.approx(0.38, abs=1e-2)
    problem = run_system(ComputeWingCLALPHAopenvsp(low_speed_aero=True), ivc)
    cl0 = problem["data:aerodynamics:aircraft:low_speed:CL0_clean"]
    assert cl0 == pytest.approx(0.0117, abs=1e-3)
    cl_alpha_wing = problem["data:aerodynamics:aircraft:low_speed:CL_alpha"]
    assert cl_alpha_wing == pytest.approx(0.58, abs=1e-2)
    problem = run_system(ComputeHTPCLALPHAopenvsp(low_speed_aero=True), ivc)
    cl_alpha_htp = problem["data:aerodynamics:horizontal_tail:low_speed:CL_alpha"]
    assert cl_alpha_htp == pytest.approx(0.0034, abs=1e-3)
    problem = run_system(ComputeHTPCLCMopenvsp(result_folder_path=OPENVSP_RESULTS), ivc)
    alpha_interp = problem["data:aerodynamics:horizontal_tail:low_speed:alpha"]
    cl_interp = problem["data:aerodynamics:horizontal_tail:low_speed:CL"]
    cm_interp = problem["data:aerodynamics:horizontal_tail:low_speed:CM"]
    cl = np.interp(10.0, alpha_interp, cl_interp)
    cm = np.interp(10.0, alpha_interp, cm_interp)
    assert cl == pytest.approx(0.00055, abs=1e-4)
    assert cm == pytest.approx(0.052, abs=1e-2)
    assert pth.exists(pth.join(OPENVSP_RESULTS, 'ClCmHT'))


def test_polar_low_speed():
    """ Tests polar component @ low speed (high speed tested separately) """

    # Define independent input value
    ivc = om.IndepVarComp()
    ivc.add_output("data:aerodynamics:wing:low_speed:reynolds", 18000000)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.20)
    ivc.add_output("data:geometry:wing:thickness_ratio", 0.1284)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(XfoilPolar(low_speed_aero=True), ivc)
    cl_max_2d = problem["data:aerodynamics:wing:low_speed:CL_max_2D"]
    assert cl_max_2d == pytest.approx(1.94, abs=1e-2)
    alpha_interp = problem["data:aerodynamics:wing:low_speed:alpha"]
    cl_interp = problem["data:aerodynamics:wing:low_speed:CL"]
    cd_interp = problem["data:aerodynamics:wing:low_speed:CD"]
    cdp_interp = problem["data:aerodynamics:wing:low_speed:CDp"]
    cm_interp = problem["data:aerodynamics:wing:low_speed:CM"]
    alpha_interp, cl_interp, cd_interp, cdp_interp, cm_interp = reshape_data(alpha_interp, cl_interp, cd_interp, cdp_interp, cm_interp)
    cl_20 = np.interp(20.0, alpha_interp, cl_interp)
    assert cl_20 == pytest.approx(1.92, abs=1e-2)
    cd_20 = np.interp(20.0, alpha_interp, cd_interp)
    assert cd_20 == pytest.approx(0.032, abs=1e-3)
    cdp_20 = np.interp(20.0, alpha_interp, cdp_interp)
    assert cdp_20 == pytest.approx(0.029, abs=1e-3)
    cm_20 = np.interp(20.0, alpha_interp, cm_interp)
    assert cm_20 == pytest.approx(0.028, abs=1e-3)


def test_cnbeta():
    """ Tests cn beta components (fuselage/vt) """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("cnbeta_fus", ComputeCnBetaFuselage(), promotes=["*"])
    group.add_subsystem("cnbeta_vt", ComputeCnBetaVT(), promotes=["*"])
    input_list = list_inputs(group)
    # print(input_list)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.80)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCnBetaFuselage(), ivc)
    cn_beta_fus = problem["data:aerodynamics:fuselage:cruise:CnBeta"]
    assert cn_beta_fus == pytest.approx(-0.015, abs=1e-3)
    problem = run_system(ComputeCnBetaVT(), ivc)
    cn_beta_vt = problem["data:aerodynamics:vertical_tail:cruise:CnBeta"]
    assert cn_beta_vt == pytest.approx(4.50, abs=1e-2)


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
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.20)
    ivc.add_output("data:aerodynamics:aircraft:low_speed:CL_alpha", 8.30)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeDeltaHighLift(), ivc)
    cl_landing = problem["data:aerodynamics:flaps:landing:CL"]
    assert cl_landing == pytest.approx(0.043, abs=1e-3)
    cm_landing = problem["data:aerodynamics:flaps:landing:CM"]
    assert cm_landing == pytest.approx(-0.006, abs=1e-3)
    cd_landing = problem["data:aerodynamics:flaps:landing:CD"]
    assert cd_landing == pytest.approx(0.007, abs=1e-3)
    cl_takeoff = problem["data:aerodynamics:flaps:takeoff:CL"]
    assert cl_takeoff == pytest.approx(0.559, abs=1e-3)
    cm_takeoff = problem["data:aerodynamics:flaps:takeoff:CM"]
    assert cm_takeoff == pytest.approx(-0.083, abs=1e-3)
    cd_takeoff = problem["data:aerodynamics:flaps:takeoff:CD"]
    assert cd_takeoff == pytest.approx(0.001, abs=1e-3)
    angle_interp = problem["data:aerodynamics:elevator:low_speed:angle"]
    cl_interp = problem["data:aerodynamics:elevator:low_speed:CL"]
    cl_elevator = np.interp(10.0, angle_interp, cl_interp)
    assert cl_elevator == pytest.approx(0.675, abs=1e-3)


def test_max_cl():
    """ Tests maximum cl component """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeMaxCL(), promotes=["*"])
    input_list = list_inputs(group)
    # print(input_list)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:aerodynamics:wing:low_speed:CL_max_2D", 1.94)
    ivc.add_output("data:aerodynamics:flaps:landing:CL", 0.043)
    ivc.add_output("data:aerodynamics:flaps:takeoff:CL", 0.012)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMaxCL(), ivc)
    cl_max_takeoff = problem["data:aerodynamics:aircraft:takeoff:CL_max"]
    assert cl_max_takeoff == pytest.approx(1.59, abs=1e-2)
    cl_max_landing = problem["data:aerodynamics:aircraft:landing:CL_max"]
    assert cl_max_landing == pytest.approx(1.62, abs=1e-2)

def test_L_D_max():
    """ Tests best lift/drag component """

    # Define independent input value
    ivc = om.IndepVarComp()
    ivc.add_output("data:aerodynamics:aircraft:cruise:CL0_clean", 0.4)
    ivc.add_output("data:aerodynamics:aircraft:cruise:CL_alpha", 9.31)
    ivc.add_output("data:aerodynamics:aircraft:cruise:CD0", 0.011)
    ivc.add_output("data:aerodynamics:aircraft:cruise:induced_drag_coefficient", 0.068)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLDMax(), ivc)
    l_d_max = problem["data:aerodynamics:aircraft:cruise:L_D_max"]
    assert l_d_max == pytest.approx(18.2, abs=1e-1)
    optimal_cl = problem["data:aerodynamics:aircraft:cruise:optimal_CL"]
    assert optimal_cl == pytest.approx(0.40, abs=1e-2)
    optimal_cd = problem["data:aerodynamics:aircraft:cruise:optimal_CD"]
    assert optimal_cd == pytest.approx(0.022, abs=1e-3)
    optimal_alpha = problem["data:aerodynamics:aircraft:cruise:optimal_alpha"]
    assert optimal_alpha == pytest.approx(0.013, abs=1e-3)