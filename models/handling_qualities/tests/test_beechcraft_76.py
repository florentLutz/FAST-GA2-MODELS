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
import numpy as np

import pytest
from fastoad.io import VariableIO
from fastoad.module_management import OpenMDAOSystemRegistry

from ...tests.testing_utilities import run_system
from ..compute_static_margin import ComputeStaticMargin
from ..tail_sizing.compute_ht_area import _ComputeHTArea
from ..tail_sizing.compute_vt_area import _ComputeVTArea

XML_FILE = "beechcraft_76.xml"
ENGINE_WRAPPER = "fastga.wrapper.propulsion.basicIC_engine"


def get_indep_var_comp(var_names):
    """ Reads required input data and returns an IndepVarcomp() instance"""
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    ivc = reader.read(only=var_names).to_ivc()
    return ivc


def register_wrappers():
    path_split = pth.dirname(__file__).split('\\')
    drive = path_split[0]
    del path_split[0]
    del path_split[-1]
    del path_split[-1]
    path = drive + "\\" + pth.join(*path_split)
    OpenMDAOSystemRegistry.explore_folder(path)


def test_compute_vt_area():
    """ Tests computation of the vertical tail area """

    # Input list from model (not generated because of engine wrapper
    input_list = [
        "data:geometry:propulsion:engine:count",
        "data:geometry:wing:area",
        "data:geometry:wing:span",
        "data:geometry:wing:MAC:length",
        "data:weight:aircraft:CG:aft:MAC_position",
        "data:aerodynamics:fuselage:cruise:CnBeta",
        "data:aerodynamics:vertical_tail:cruise:CL_alpha",
        "data:TLAR:v_cruise",
        "data:TLAR:v_approach",
        "data:mission:sizing:main_route:cruise:altitude",
        "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
        "data:geometry:propulsion:nacelle:wet_area",
        "data:geometry:propulsion:nacelle:y"
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:weight:aircraft:CG:aft:MAC_position", 0.364924)
    ivc.add_output("data:aerodynamics:fuselage:cruise:CnBeta", -0.0599)
    ivc.add_output("data:propulsion:IC_engine:max_power", 130000, units="W")  # correct value to fit old version def.
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(_ComputeVTArea(propulsion_id=ENGINE_WRAPPER), ivc)
    vt_area = problem.get_val("data:geometry:vertical_tail:area", units="m**2")
    assert vt_area == pytest.approx(2.44, abs=1e-2)  # old-version obtained value 2.4m²


def test_compute_ht_area():
    """ Tests computation of the horizontal tail area """

    input_list = [
        "data:aerodynamics:aircraft:landing:CL_max",
        "data:aerodynamics:aircraft:takeoff:CL_max",
        "data:aerodynamics:wing:low_speed:CL_max_clean",
        "data:aerodynamics:aircraft:low_speed:CL0_clean",
        "data:aerodynamics:aircraft:low_speed:CL_alpha",
        "data:aerodynamics:flaps:landing:CL",
        "data:aerodynamics:flaps:takeoff:CL",
        "data:aerodynamics:flaps:landing:CM",
        "data:aerodynamics:flaps:takeoff:CM",
        "data:aerodynamics:horizontal_tail:low_speed:CL_alpha",
        "data:geometry:propulsion:engine:count",
        "data:geometry:propulsion:nacelle:height",
        "data:geometry:wing:MAC:at25percent:x",
        "data:geometry:wing:MAC:length",
        "data:geometry:wing:area",
        "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
        "data:mission:sizing:landing:elevator_angle",
        "data:mission:sizing:landing:thrust_rate",
        "data:mission:sizing:takeoff:elevator_angle",
        "data:mission:sizing:takeoff:thrust_rate",
        "data:propulsion:MTO_thrust",
        "data:weight:aircraft:MTOW",
        "data:weight:aircraft:MLW",
        "data:weight:aircraft:CG:aft:x",
        "settings:weight:aircraft:CG:range",
    ]

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:alpha",
                   np.array([0.0, 7.5, 15.0, 22.5, 30.0]), units="deg")
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:CL",
                   np.array([-0.00472, 0.08476, 0.16876, 0.23578, 0.28567]))
    ivc.add_output("data:aerodynamics:horizontal_tail:low_speed:CM",
                   np.array([0.01308, -0.24393, -0.50001, -0.71332, -0.88161]))
    ivc.add_output("data:aerodynamics:wing:low_speed:alpha",
                   np.array([0.0, 7.5, 15.0, 22.5, 30.0]), units="deg")
    ivc.add_output("data:aerodynamics:wing:low_speed:CM",
                   np.array([-0.01332, 0.02356, 0.10046, 0.20401, 0.31282]))
    ivc.add_output("data:aerodynamics:elevator:low_speed:CL_alpha", 0.6167, units="rad**-1")
    ivc.add_output("data:propulsion:IC_engine:max_power", 130000, units="W")  # correct value to fit old version def.
    ivc.add_output("data:propulsion:IC_engine:fuel_type", 1.0)
    ivc.add_output("data:propulsion:IC_engine:strokes_nb", 4.0)
    ivc.add_output("data:weight:airframe:landing_gear:main:CG:x", 3.97, units="m")  # correct value to fit old version

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    # noinspection PyTypeChecker
    problem = run_system(_ComputeHTArea(propulsion_id=ENGINE_WRAPPER), ivc)
    ht_area = problem.get_val("data:geometry:horizontal_tail:area", units="m**2")
    assert ht_area == pytest.approx(5.48, abs=1e-2)  # old-version obtained value 3.9m²


def test_compute_static_margin():
    """ Tests computation of static margin """

    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    input_vars.add_output("data:weight:aircraft:CG:aft:MAC_position", 0.20)

    problem = run_system(ComputeStaticMargin(), input_vars)
    static_margin = problem["data:handling_qualities:static_margin"]
    assert static_margin == pytest.approx(0.55, rel=1e-2)
