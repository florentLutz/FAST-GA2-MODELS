"""
Test takeoff module
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
import pandas as pd
import numpy as np
from openmdao.core.component import Component
from openmdao.core.group import Group
import pytest
from typing import Union

from fastoad.io import VariableIO
from fastoad.module_management.service_registry import RegisterPropulsion
from fastoad.model_base import FlightPoint, Atmosphere
from fastoad.model_base.propulsion import IOMPropulsionWrapper

from models.tests.testing_utilities import run_system, get_indep_var_comp, list_inputs
from models.weight.cg.cg_variation import InFlightCGVariation
from ..takeoff import TakeOffPhase, _v2, _vr_from_v2, _vloff_from_v2, _simulate_takeoff
from ..mission import _compute_taxi, _compute_climb, _compute_cruise, _compute_descent
from ..sizing import Sizing
from models.propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from models.propulsion.propulsion import IPropulsion

XML_FILE = "cirrus_sr22.xml"
ENGINE_WRAPPER = "test.wrapper.performances.cirrus.dummy_engine"


class DummyEngine(AbstractFuelPropulsion):

    def __init__(self):
        """
        Dummy engine model returning thrust in particular conditions defined for htp/vtp areas.

        """
        super().__init__()
        self.max_power = 231000.0
        self.max_thrust = 5417.0

    def compute_flight_points(self, flight_points: Union[FlightPoint, pd.DataFrame]):

        altitude = float(Atmosphere(np.array(flight_points.altitude)).get_altitude(altitude_in_feet=True))
        mach = np.array(flight_points.mach)
        thrust = np.array(flight_points.thrust)
        sigma = Atmosphere(altitude).density / Atmosphere(0.0).density
        max_power = self.max_power * (sigma - (1 - sigma) / 7.55)
        max_thrust = min(
            self.max_thrust * sigma**(1./3.),
            max_power * 0.8 / np.maximum(mach * Atmosphere(altitude).speed_of_sound, 1e-20)
        )
        if flight_points.thrust_rate is None:
            flight_points.thrust = min(max_thrust, float(thrust))
            flight_points.thrust_rate = float(thrust) / max_thrust
        else:
            flight_points.thrust = max_thrust * np.array(flight_points.thrust_rate)
        sfc_pmax = 8.5080e-08  # fixed whatever the thrust ratio, sfc for ONE 130kW engine !
        sfc = sfc_pmax * flight_points.thrust_rate * mach * Atmosphere(altitude).speed_of_sound
        flight_points.sfc = sfc

    def compute_weight(self) -> float:
        return 0.0

    def compute_dimensions(self) -> (float, float, float, float):
        return [0.0, 0.0, 0.0, 0.0]

    def compute_drag(self, mach, unit_reynolds, l0_wing):
        return 0.0


class DummyEngineWrapper(IOMPropulsionWrapper):
    def setup(self, component: Component):
        component.add_input("data:TLAR:v_cruise", np.nan, units="m/s")
        component.add_input("data:mission:sizing:main_route:cruise:altitude", np.nan, units="m")

    @staticmethod
    def get_model(inputs) -> IPropulsion:
        return DummyEngine()


RegisterPropulsion(ENGINE_WRAPPER)(DummyEngineWrapper)


def test_v2():
    """ Tests safety speed """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(_v2(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(_v2(propulsion_id=ENGINE_WRAPPER), ivc)
    v2 = problem.get_val("v2:speed", units="m/s")
    assert v2 == pytest.approx(43.18, abs=1e-2)
    alpha = problem.get_val("v2:angle", units="deg")
    assert alpha == pytest.approx(7.538, abs=1e-2)


def test_vloff():
    """ Tests lift-off speed """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(_vloff_from_v2(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE)
    ivc.add_output("v2:speed", 43.18, units='m/s')
    ivc.add_output("v2:angle", 7.538, units='deg')

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(_vloff_from_v2(propulsion_id=ENGINE_WRAPPER), ivc)
    vloff = problem.get_val("vloff:speed", units="m/s")
    assert vloff == pytest.approx(42.39, abs=1e-2)
    alpha = problem.get_val("vloff:angle", units="deg")
    assert alpha == pytest.approx(7.538, abs=1e-2)


def test_vr():
    """ Tests rotation speed """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(_vr_from_v2(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE)
    ivc.add_output("vloff:speed", 42.39, units='m/s')
    ivc.add_output("vloff:angle", 7.538, units='deg')

    # Run problem and check obtained value(s) is/(are) correct

    problem = run_system(_vr_from_v2(propulsion_id=ENGINE_WRAPPER), ivc)
    vr = problem.get_val("vr:speed", units="m/s")
    assert vr == pytest.approx(36.27, abs=1e-2)


def test_simulate_takeoff():
    """ Tests simulate takeoff """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(_simulate_takeoff(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE)
    ivc.add_output("vr:speed", 36.27, units='m/s')
    ivc.add_output("v2:angle", 7.538, units='deg')

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(_simulate_takeoff(propulsion_id=ENGINE_WRAPPER), ivc)
    vr = problem.get_val("data:mission:sizing:takeoff:VR", units='m/s')
    assert vr == pytest.approx(36.27, abs=1e-2)
    vloff = problem.get_val("data:mission:sizing:takeoff:VLOF", units='m/s')
    assert vloff == pytest.approx(42.53, abs=1e-2)
    v2 = problem.get_val("data:mission:sizing:takeoff:V2", units='m/s')
    assert v2 == pytest.approx(47.84, abs=1e-2)
    tofl = problem.get_val("data:mission:sizing:takeoff:TOFL", units='m')
    assert tofl == pytest.approx(341.61, abs=1)
    duration = problem.get_val("data:mission:sizing:takeoff:duration", units='s')
    assert duration == pytest.approx(20.5, abs=1e-1)
    fuel1 = problem.get_val("data:mission:sizing:takeoff:fuel", units='kg')
    assert fuel1 == pytest.approx(0.24, abs=1e-2)
    fuel2 = problem.get_val("data:mission:sizing:initial_climb:fuel", units='kg')
    assert fuel2 == pytest.approx(0.075, abs=1e-2)


def test_takeoff_phase_connections():
    """ Tests complete take-off phase connection with speeds """

    # load all inputs
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    ivc = reader.read().to_ivc()

    # Run problem and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(TakeOffPhase(propulsion_id=ENGINE_WRAPPER), ivc)
    vr = problem.get_val("data:mission:sizing:takeoff:VR", units='m/s')
    assert vr == pytest.approx(36.28, abs=1e-2)
    vloff = problem.get_val("data:mission:sizing:takeoff:VLOF", units='m/s')
    assert vloff == pytest.approx(42.52, abs=1e-2)
    v2 = problem.get_val("data:mission:sizing:takeoff:V2", units='m/s')
    assert v2 == pytest.approx(47.84, abs=1e-2)
    tofl = problem.get_val("data:mission:sizing:takeoff:TOFL", units='m')
    assert tofl == pytest.approx(341.49, abs=1)
    duration = problem.get_val("data:mission:sizing:takeoff:duration", units='s')
    assert duration == pytest.approx(20.5, abs=1e-1)
    fuel1 = problem.get_val("data:mission:sizing:takeoff:fuel", units='kg')
    assert fuel1 == pytest.approx(0.246, abs=1e-2)
    fuel2 = problem.get_val("data:mission:sizing:initial_climb:fuel", units='kg')
    assert fuel2 == pytest.approx(0.075, abs=1e-2)


def test_compute_taxi():
    """ Tests taxi in/out phase """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(_compute_taxi(propulsion_id=ENGINE_WRAPPER, taxi_out=True)),
                             __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(_compute_taxi(propulsion_id=ENGINE_WRAPPER, taxi_out=True), ivc)
    fuel_mass = problem.get_val("data:mission:sizing:taxi_out:fuel", units="kg")
    assert fuel_mass == pytest.approx(0.192, abs=1e-2)  # result strongly dependent on the defined Thrust limit

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(_compute_taxi(propulsion_id=ENGINE_WRAPPER, taxi_out=False)),
                             __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(_compute_taxi(propulsion_id=ENGINE_WRAPPER, taxi_out=False), ivc)
    fuel_mass = problem.get_val("data:mission:sizing:taxi_in:fuel", units="kg")
    assert fuel_mass == pytest.approx(0.192, abs=1e-2)  # result strongly dependent on the defined Thrust limit


def test_compute_climb():
    """ Tests climb phase """

    # Research independent input value in .xml file
    group = Group()
    group.add_subsystem("in_flight_cg_variation", InFlightCGVariation(), promotes=["*"])
    group.add_subsystem("descent", _compute_climb(propulsion_id=ENGINE_WRAPPER), promotes=["*"])
    ivc = get_indep_var_comp(list_inputs(group), __file__, XML_FILE)
    ivc.add_output("data:mission:sizing:taxi_out:fuel", 0.50, units="kg")
    ivc.add_output("data:mission:sizing:takeoff:fuel", 0.29, units="kg")
    ivc.add_output("data:mission:sizing:initial_climb:fuel", 0.07, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(group, ivc)
    v_cas = problem.get_val("data:mission:sizing:main_route:climb:v_cas", units="kn")
    assert v_cas == pytest.approx(71.5, abs=1)
    fuel_mass = problem.get_val("data:mission:sizing:main_route:climb:fuel", units="kg")
    assert fuel_mass == pytest.approx(5.107, abs=1e-1)
    distance = problem.get_val("data:mission:sizing:main_route:climb:distance", units="m") / 1000.0  # conversion to km
    assert distance == pytest.approx(16.648, abs=1e-2)
    duration = problem.get_val("data:mission:sizing:main_route:climb:duration", units="min")
    assert duration == pytest.approx(5.645, abs=1e-2)


def test_compute_cruise():
    """ Tests cruise phase """

    # Research independent input value in .xml file
    group = Group()
    group.add_subsystem("in_flight_cg_variation", InFlightCGVariation(), promotes=["*"])
    group.add_subsystem("cruise", _compute_cruise(propulsion_id=ENGINE_WRAPPER), promotes=["*"])
    ivc = get_indep_var_comp(list_inputs(group), __file__, XML_FILE)
    ivc.add_output("data:mission:sizing:taxi_out:fuel", 0.50, units="kg")
    ivc.add_output("data:mission:sizing:takeoff:fuel", 0.29, units="kg")
    ivc.add_output("data:mission:sizing:initial_climb:fuel", 0.07, units="kg")
    ivc.add_output("data:mission:sizing:main_route:climb:fuel", 5.56, units="kg")
    ivc.add_output("data:mission:sizing:main_route:climb:distance", 13.2, units="km")
    ivc.add_output("data:mission:sizing:main_route:descent:distance", 0.0, units="km")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(group, ivc)
    fuel_mass = problem.get_val("data:mission:sizing:main_route:cruise:fuel", units="kg")
    assert fuel_mass == pytest.approx(155.003, abs=1e-1)
    duration = problem.get_val("data:mission:sizing:main_route:cruise:duration", units="h")
    assert duration == pytest.approx(5.51, abs=1e-2)


def test_compute_descent():
    """ Tests descent phase """

    # Research independent input value in .xml file
    group = Group()
    group.add_subsystem("in_flight_cg_variation", InFlightCGVariation(), promotes=["*"])
    group.add_subsystem("descent", _compute_descent(propulsion_id=ENGINE_WRAPPER), promotes=["*"])
    ivc = get_indep_var_comp(list_inputs(group), __file__, XML_FILE)
    ivc.add_output("data:mission:sizing:taxi_out:fuel", 0.98, units="kg")
    ivc.add_output("data:mission:sizing:takeoff:fuel", 0.29, units="kg")
    ivc.add_output("data:mission:sizing:initial_climb:fuel", 0.07, units="kg")
    ivc.add_output("data:mission:sizing:main_route:climb:fuel", 5.56, units="kg")
    ivc.add_output("data:mission:sizing:main_route:cruise:fuel", 188.05, units="kg")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(group, ivc)
    fuel_mass = problem.get_val("data:mission:sizing:main_route:descent:fuel", units="kg")
    assert fuel_mass == pytest.approx(0.863, abs=1e-2)
    distance = problem.get_val("data:mission:sizing:main_route:descent:distance", units="m") / 1000  # conversion to km
    assert distance == pytest.approx(81.255, abs=1e-2)
    duration = problem.get_val("data:mission:sizing:main_route:descent:duration", units="min")
    assert duration == pytest.approx(28.277, abs=1e-2)


def test_loop_cruise_distance():
    """ Tests a distance computation loop matching the descent value/TLAR total range. """

    # Get the parameters from .xml
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    ivc = reader.read().to_ivc()

    # Run problem and check obtained value(s) is/(are) correct
    # noinspection PyTypeChecker
    problem = run_system(Sizing(propulsion_id=ENGINE_WRAPPER), ivc)
    m_total = problem.get_val("data:mission:sizing:fuel", units="kg")
    assert m_total == pytest.approx(175.204, abs=1)
    climb_distance = problem.get_val("data:mission:sizing:main_route:climb:distance", units="NM")
    cruise_distance = problem.get_val("data:mission:sizing:main_route:cruise:distance", units="NM")
    descent_distance = problem.get_val("data:mission:sizing:main_route:descent:distance", units="NM")
    total_distance = problem.get_val("data:TLAR:range", units="NM")
    error_distance = total_distance - (climb_distance + cruise_distance + descent_distance)
    assert error_distance == pytest.approx(0.0, abs=1e-1)
