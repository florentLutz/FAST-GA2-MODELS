"""
Test module for geometry functions of cg components
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

import pytest
from fastoad.io import VariableIO

from ....tests.testing_utilities import run_system, get_indep_var_comp, list_inputs, Timer

from ..cg import CG
from ..cg_components.a_airframe import ComputeWingCG, ComputeFuselageCG, ComputeTailCG, ComputeFlightControlCG, \
    ComputeLandingGearCG
from ..cg_components.b_propulsion import ComputeEngineCG, ComputeFuelLinesCG, ComputeTankCG
from ..cg_components.c_systems import ComputePowerSystemsCG, ComputeLifeSupportCG, ComputeNavigationSystemsCG
from ..cg_components.d_furniture import ComputePassengerSeatsCG
from ..cg_components.payload import ComputePayloadCG
from ..cg_components.loadcase import ComputeCGLoadCase
from ..cg_components.ratio_aft import ComputeCGRatioAft
from ..cg_components.max_cg_ratio import ComputeMaxCGratio
from ..cg_components.update_mlg import UpdateMLG

XML_FILE = "beechcraft_76.xml"


def test_compute_cg_wing():
    """ Tests computation of wing center of gravity """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeWingCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingCG(), ivc)
    x_cg_a1 = problem.get_val("data:weight:airframe:wing:CG:x", units="m")
    assert x_cg_a1 == pytest.approx(3.68, abs=1e-2)


def test_compute_cg_fuselage():
    """ Tests computation of fuselage center of gravity """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeFuselageCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageCG(), ivc)
    x_cg_a2 = problem.get_val("data:weight:airframe:fuselage:CG:x", units="m")
    assert x_cg_a2 == pytest.approx(3.46593, abs=1e-2)


def test_compute_cg_tail():
    """ Tests computation of tail center(s) of gravity """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeTailCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeTailCG(), ivc)
    x_cg_a31 = problem.get_val("data:weight:airframe:horizontal_tail:CG:x", units="m")
    assert x_cg_a31 == pytest.approx(7.9769453, abs=1e-2)
    x_cg_a32 = problem.get_val("data:weight:airframe:vertical_tail:CG:x", units="m")
    assert x_cg_a32 == pytest.approx(7.9571, abs=1e-2)


def test_compute_cg_flight_control():
    """ Tests computation of flight control center of gravity """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeFlightControlCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFlightControlCG(), ivc)
    x_cg_a4 = problem.get_val("data:weight:airframe:flight_controls:CG:x", units="m")
    assert x_cg_a4 == pytest.approx(4.61, abs=1e-2)


def test_compute_cg_landing_gear():
    """ Tests computation of landing gear center(s) of gravity """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeLandingGearCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLandingGearCG(), ivc)
    x_cg_a52 = problem.get_val("data:weight:airframe:landing_gear:front:CG:x", units="m")
    assert x_cg_a52 == pytest.approx(1.71, abs=1e-2)


def test_compute_cg_engine():
    """ Tests computation of engine(s) center of gravity """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeEngineCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeEngineCG(), ivc)
    x_cg_b1 = problem.get_val("data:weight:propulsion:engine:CG:x", units="m")
    assert x_cg_b1 == pytest.approx(2.7, abs=1e-2)


def test_compute_cg_fuel_lines():
    """ Tests fuel lines center of gravity """

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs((ComputeFuelLinesCG())), __file__, XML_FILE)
    ivc.add_output("data:weight:propulsion:engine:CG:x", 2.7, units="m")
    ivc.add_output("data:weight:propulsion:tank:CG:x", 3.83, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuelLinesCG(), ivc)
    x_cg_b2 = problem.get_val("data:weight:propulsion:fuel_lines:CG:x", units="m")
    assert x_cg_b2 == pytest.approx(3.265, abs=1e-2)


def test_compute_cg_tank():
    """ Tests tank center of gravity """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeTankCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeTankCG(), ivc)
    x_cg_b3 = problem.get_val("data:weight:propulsion:tank:CG:x", units="m")
    assert x_cg_b3 == pytest.approx(3.83, abs=1e-2)


def test_compute_cg_power_systems():
    """ Tests computation of power systems center of gravity """

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputePowerSystemsCG()), __file__, XML_FILE)
    ivc.add_output("data:weight:propulsion:engine:CG:x", 2.7, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePowerSystemsCG(), ivc)
    x_cg_c12 = problem.get_val("data:weight:systems:power:electric_systems:CG:x", units="m")
    assert x_cg_c12 == pytest.approx(4.1185, abs=1e-2)
    x_cg_c13 = problem.get_val("data:weight:systems:power:hydraulic_systems:CG:x", units="m")
    assert x_cg_c13 == pytest.approx(4.1185, abs=1e-2)


def test_compute_cg_life_support_systems():
    """ Tests computation of life support systems center of gravity """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeLifeSupportCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeLifeSupportCG(), ivc)
    x_cg_c22 = problem.get_val("data:weight:systems:life_support:air_conditioning:CG:x", units="m")
    assert x_cg_c22 == pytest.approx(2.274, abs=1e-2)


def test_compute_cg_navigation_systems():
    """ Tests computation of navigation systems center of gravity """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeNavigationSystemsCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeNavigationSystemsCG(), ivc)
    x_cg_c3 = problem.get_val("data:weight:systems:navigation:CG:x", units="m")
    assert x_cg_c3 == pytest.approx(2.624, abs=1e-2)


def test_compute_cg_passenger_seats():
    """ Tests computation of passenger seats center of gravity """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputePassengerSeatsCG()), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePassengerSeatsCG(), ivc)
    x_cg_d2 = problem.get_val("data:weight:furniture:passenger_seats:CG:x", units="m")
    assert x_cg_d2 == pytest.approx(3.92, abs=1e-2)  # modified with new cabin definition


def test_compute_cg_payload():
    """ Tests computation of payload center(s) of gravity """

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputePayloadCG()), __file__, XML_FILE)
    ivc.add_output("data:weight:furniture:passenger_seats:CG:x", 4.13, units="m")   # use old fast-version

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputePayloadCG(), ivc)
    x_cg_pl = problem.get_val("data:weight:payload:PAX:CG:x", units="m")
    assert x_cg_pl == pytest.approx(4.13, abs=1e-1)
    x_cg_rear_fret = problem.get_val("data:weight:payload:rear_fret:CG:x", units="m")
    assert x_cg_rear_fret == pytest.approx(5.68, abs=1e-2)
    x_cg_front_fret = problem.get_val("data:weight:payload:front_fret:CG:x", units="m")
    assert x_cg_front_fret == pytest.approx(0.0, abs=1e-2)


def test_compute_cg_ratio_aft():
    """ Tests computation of center of gravity with aft estimation """

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeCGRatioAft()), __file__, XML_FILE)
    ivc.add_output("data:weight:airframe:wing:CG:x", 0.39, units="m")
    ivc.add_output("data:weight:airframe:fuselage:CG:x", 3.99, units="m")
    ivc.add_output("data:weight:airframe:horizontal_tail:CG:x", 7.91, units="m")
    ivc.add_output("data:weight:airframe:vertical_tail:CG:x", 8.05, units="m")
    ivc.add_output("data:weight:airframe:flight_controls:CG:x", 4.61, units="m")
    ivc.add_output("data:weight:airframe:landing_gear:main:CG:x", 3.62, units="m")
    ivc.add_output("data:weight:airframe:landing_gear:front:CG:x", 1.71, units="m")
    ivc.add_output("data:weight:propulsion:engine:CG:x", 2.7, units="m")
    ivc.add_output("data:weight:propulsion:fuel_lines:CG:x", 2.7, units="m")
    ivc.add_output("data:weight:systems:power:electric_systems:CG:x", 0.0, units="m")
    ivc.add_output("data:weight:systems:power:hydraulic_systems:CG:x", 0.0, units="m")
    ivc.add_output("data:weight:systems:life_support:air_conditioning:CG:x", 0.0, units="m")
    ivc.add_output("data:weight:systems:navigation:CG:x", 2.97, units="m")
    ivc.add_output("data:weight:furniture:passenger_seats:CG:x", 4.13, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeCGRatioAft(), ivc)
    empty_mass = problem.get_val("data:weight:aircraft_empty:mass", units="kg")
    assert empty_mass == pytest.approx(950.47, abs=1e-2)
    cg_x = problem.get_val("data:weight:aircraft_empty:CG:x", units="m")
    assert cg_x == pytest.approx(2.66, abs=1e-2)
    cg_mac_pos = problem["data:weight:aircraft:empty:CG:MAC_position"]
    assert cg_mac_pos == pytest.approx(-0.26, abs=1e-2)


def test_compute_cg_loadcase():
    """ Tests computation of center of gravity for all load case conf. """

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(ComputeCGLoadCase()), __file__, XML_FILE)
    ivc.add_output("data:weight:payload:PAX:CG:x", 4.13, units="m")  # use old fast-version for calculation
    ivc.add_output("data:weight:payload:rear_fret:CG:x", 5.68, units="m")
    ivc.add_output("data:weight:payload:front_fret:CG:x", 0.0, units="m")
    ivc.add_output("data:weight:aircraft_empty:CG:x", 2.66, units="m")
    ivc.add_output("data:weight:aircraft_empty:mass", 950.47, units="kg")
    ivc.add_output("data:weight:propulsion:tank:CG:x", 3.83, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    for case in range(1, 7):
        problem = run_system(ComputeCGLoadCase(load_case=case), ivc)
        cg_ratio_lc = problem.get_val("data:weight:aircraft:load_case_"+str(case)+":CG:MAC_position")
        if case == 1:
            assert cg_ratio_lc == pytest.approx(0.14, abs=1e-2)
        elif case == 2:
            assert cg_ratio_lc == pytest.approx(0.17, abs=1e-2)
        elif case == 3:
            assert cg_ratio_lc == pytest.approx(0.03, abs=1e-2)
        elif case == 4:
            assert cg_ratio_lc == pytest.approx(0.15, abs=1e-2)
        elif case == 5:
            assert cg_ratio_lc == pytest.approx(0.03, abs=1e-2)
        elif case == 6:
            assert cg_ratio_lc == pytest.approx(0.06, abs=1e-2)
        else:
            pass


def test_compute_max_cg_ratio():
    """ Tests computation of maximum center of gravity ratio """

    # Define the independent input values that should be filled if basic function is chosen
    ivc = om.IndepVarComp()
    ivc.add_output("data:weight:aircraft:empty:CG:MAC_position", -0.26)
    ivc.add_output("data:weight:aircraft:load_case_1:CG:MAC_position", 0.14)
    ivc.add_output("data:weight:aircraft:load_case_2:CG:MAC_position", 0.20)
    ivc.add_output("data:weight:aircraft:load_case_3:CG:MAC_position", 0.08)
    ivc.add_output("data:weight:aircraft:load_case_4:CG:MAC_position", 0.15)
    ivc.add_output("data:weight:aircraft:load_case_5:CG:MAC_position", 0.06)
    ivc.add_output("data:weight:aircraft:load_case_6:CG:MAC_position", 0.09)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMaxCGratio(), ivc)
    cg_ratio = problem.get_val("data:weight:aircraft:CG:aft:MAC_position")
    assert cg_ratio == pytest.approx(0.25, abs=1e-2)


def test_update_mlg():
    """ Tests computation of MLG update """

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(list_inputs(UpdateMLG()), __file__, XML_FILE)
    ivc.add_output("data:weight:aircraft:CG:aft:MAC_position", 0.25)
    ivc.add_output("data:weight:airframe:landing_gear:front:CG:x", 1.71, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(UpdateMLG(), ivc)
    cg_a51 = problem.get_val("data:weight:airframe:landing_gear:main:CG:x", units="m")
    assert cg_a51 == pytest.approx(4.19, abs=1e-2)


def test_complete_cg():
    """ Run computation of all models """

    with Timer(name="CG: loop"):
        # with data from file
        reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
        reader.path_separator = ":"
        input_vars = reader.read().to_ivc()

        # Run problem and check obtained value(s) is/(are) correct
        problem = run_system(CG(), input_vars, check=True)
        cg_global = problem.get_val("data:weight:aircraft:CG:aft:x", units="m")
        assert cg_global == pytest.approx(3.86577958, abs=1e-1)
        cg_ratio = problem.get_val("data:weight:aircraft:CG:aft:MAC_position")
        assert cg_ratio == pytest.approx(0.51906364, abs=1e-2)
