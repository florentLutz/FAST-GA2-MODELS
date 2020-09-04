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
import openmdao.api as om
import numpy as np

import pytest
from fastoad.io import VariableIO

from pytest import approx
from tests.testing_utilities import run_system
from ..components.cd0 import CD0
from ..external.openvsp import ComputeOSWALDopenvsp, ComputeWingCLALPHAopenvsp, ComputeHTPCLALPHAopenvsp, ComputeHTPCLCMopenvsp
from ..aerodynamics_high_speed import AerodynamicsHighSpeed
from ..aerodynamics_low_speed import AerodynamicsLowSpeed

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


def test_cd0_high_speed():
    """ Tests drag coefficient @ high speed """

    # Generate input list from model
    group = om.Group()
    ivc = om.IndepVarComp()
    ivc.add_output("data:aerodynamics:cruise:mach", 0.8)
    ivc.add_output("data:aerodynamics:wing:cruise:reynolds", 1e8)
    group.add_subsystem("constants", ivc, promotes=["*"])
    group.add_subsystem("my_model", CD0(), promotes=["*"])
    input_list = list_inputs(group)
    print(input_list)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.8)
    ivc.add_output("data:aerodynamics:wing:cruise:reynolds", 1e8)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CD0(), ivc)
    cd0_wing = problem["data:aerodynamics:wing:cruise:CD0"]
    assert cd0_wing == pytest.approx(0.0035, abs=1e-4)
    cd0_fus = problem["data:aerodynamics:fuselage:cruise:CD0"]
    assert cd0_fus == pytest.approx(0.0040, abs=1e-4)
    cd0_ht = problem["data:aerodynamics:horizontal_tail:cruise:CD0"]
    assert cd0_ht == pytest.approx(0.0013, abs=1e-4)
    cd0_vt = problem["data:aerodynamics:vertical_tail:cruise:CD0"]
    assert cd0_vt == pytest.approx(0.0010, abs=1e-4)
    cd0_nac = problem["data:aerodynamics:nacelles:cruise:CD0"]
    assert cd0_nac == pytest.approx(0.0010, abs=1e-4)
    cd0_lg = problem["data:aerodynamics:landing_gear:cruise:CD0"]
    assert cd0_lg == pytest.approx(0.0, abs=1e-4)
    cd0_other = problem["data:aerodynamics:other:cruise:CD0"]
    assert cd0_other == pytest.approx(0.0007, abs=1e-4)


def test_cd0_low_speed():
    """ Tests drag coefficient @ low speed """

    # Generate input list from model
    group = om.Group()
    ivc = om.IndepVarComp()
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.2)
    ivc.add_output("data:aerodynamics:wing:low_speed:reynolds", 1e7)
    group.add_subsystem("constants", ivc, promotes=["*"])
    group.add_subsystem("my_model", CD0(low_speed_aero=True), promotes=["*"])
    input_list = list_inputs(group)
    print(input_list)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.2)
    ivc.add_output("data:aerodynamics:wing:low_speed:reynolds", 1e7)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(CD0(low_speed_aero=True), ivc)
    cd0_wing = problem["data:aerodynamics:wing:low_speed:CD0"]
    assert cd0_wing == pytest.approx(0.0048, abs=1e-4)
    cd0_fus = problem["data:aerodynamics:fuselage:low_speed:CD0"]
    assert cd0_fus == pytest.approx(0.0064, abs=1e-4)
    cd0_ht = problem["data:aerodynamics:horizontal_tail:low_speed:CD0"]
    assert cd0_ht == pytest.approx(0.0019, abs=1e-4)
    cd0_vt = problem["data:aerodynamics:vertical_tail:low_speed:CD0"]
    assert cd0_vt == pytest.approx(0.0014, abs=1e-4)
    cd0_nac = problem["data:aerodynamics:nacelles:low_speed:CD0"]
    assert cd0_nac == pytest.approx(0.0014, abs=1e-4)
    cd0_lg = problem["data:aerodynamics:landing_gear:low_speed:CD0"]
    assert cd0_lg == pytest.approx(0.0024, abs=1e-4)
    cd0_other = problem["data:aerodynamics:other:low_speed:CD0"]
    assert cd0_other == pytest.approx(0.0007, abs=1e-4)


def test_openvsp_comp_high_speed():
    """ Tests openvsp components @ high speed """

    # Generate input list from model
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
        "data:mission:sizing:cruise:altitude",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:aerodynamics:cruise:mach", 0.8)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeOSWALDopenvsp(), ivc)
    coef_k = problem["data:aerodynamics:aircraft:cruise:induced_drag_coefficient"]
    assert coef_k == pytest.approx(0.35, abs=1e-2)
    problem = run_system(ComputeWingCLALPHAopenvsp(), ivc)
    cl0 = problem["data:aerodynamics:aircraft:cruise:CL0_clean"]
    assert cl0 == pytest.approx(0.016, abs=1e-3)
    cl_alpha = problem["data:aerodynamics:aircraft:cruise:CL_alpha"]
    assert cl_alpha == pytest.approx(0.83, abs=1e-2)

def test_openvsp_comp_low_speed():
    """ Tests openvsp components @ low speed """

    # Generate input list from model
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
        "data:geometry: horizontal_tail:MAC: length",
        "data:geometry:horizontal_tail:MAC:at25percent:x:local",
        "data:geometry:horizontal_tail:height",

    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:aerodynamics:low_speed:mach", 0.2)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTPCLCMopenvsp(), ivc)
    alpha_interp = problem["data:aerodynamics:horizontal_tail:low_speed:alpha"]
    cl_interp = problem["data:aerodynamics:horizontal_tail:low_speed:CL"]
    cm_interp = problem["data:aerodynamics:horizontal_tail:low_speed:CL"]
    cl = np.interp(10.0, alpha_interp, cl_interp)
    cm = np.interp(10.0, alpha_interp, cm_interp)
    assert cl == pytest.approx(0.35, abs=1e-2)
    assert cm == pytest.approx(0.35, abs=1e-2)