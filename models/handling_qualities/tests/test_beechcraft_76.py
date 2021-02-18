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
import pytest
from typing import Union

from fastoad.io import VariableIO
from fastoad.module_management.service_registry import RegisterPropulsion
from fastoad import BundleLoader
from fastoad.base.flight_point import FlightPoint
from fastoad.constants import EngineSetting
from fastoad.models.propulsion.propulsion import IOMPropulsionWrapper

from ...tests.testing_utilities import run_system, register_wrappers, get_indep_var_comp, list_inputs
from ..compute_static_margin import ComputeStaticMargin
from ..tail_sizing.compute_ht_area import ComputeHTArea
from ..tail_sizing.compute_vt_area import ComputeVTArea
from ...propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from ...propulsion.propulsion import IPropulsion

XML_FILE = "beechcraft_76.xml"
ENGINE_WRAPPER = "test.wrapper.handling_qualities.beechcraft.dummy_engine"


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
        Dummy engine model returning nacelle dimensions height-width-length-wet_area.

        """
        super().__init__()
        self.prop_layout = prop_layout
        self.max_power = max_power
        self.design_altitude = design_altitude
        self.design_speed = design_speed
        self.fuel_type = fuel_type
        self.strokes_nb = strokes_nb

    def compute_flight_points(self, flight_points: Union[FlightPoint, pd.DataFrame]):
        if flight_points.engine_setting == EngineSetting.TAKEOFF:
            flight_points.thrust = 5800.0 / 2.0
        elif flight_points.engine_setting == EngineSetting.CLIMB:
            flight_points.thrust = 3110.0 / 2.0
        elif flight_points.engine_setting == EngineSetting.IDLE:
            flight_points.thrust = 605.0 / 2.0
        else:
            flight_points.thrust = 0.0
        flight_points['sfc'] = 0.0

    def compute_weight(self) -> float:
        return 0.0

    def compute_dimensions(self) -> (float, float, float, float, float, float):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def compute_drag(self, mach, unit_reynolds, l0_wing):
        return 0.0

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


def test_compute_vt_area():
    """ Tests computation of the vertical tail area """

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(ComputeVTArea(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE)
    ivc.add_output("data:weight:aircraft:CG:aft:MAC_position", 0.364924)
    ivc.add_output("data:aerodynamics:fuselage:cruise:CnBeta", -0.0599)

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    problem = run_system(ComputeVTArea(propulsion_id=ENGINE_WRAPPER), ivc)
    vt_area = problem.get_val("data:geometry:vertical_tail:area", units="m**2")
    assert vt_area == pytest.approx(2.44, abs=1e-2)  # old-version obtained value 2.4m²


def test_compute_ht_area():
    """ Tests computation of the horizontal tail area """

    # Research independent input value in .xml file
    # noinspection PyTypeChecker
    ivc = get_indep_var_comp(list_inputs(ComputeHTArea(propulsion_id=ENGINE_WRAPPER)), __file__, XML_FILE)

    # Run problem and check obtained value(s) is/(are) correct
    register_wrappers()
    # noinspection PyTypeChecker
    problem = run_system(ComputeHTArea(propulsion_id=ENGINE_WRAPPER), ivc)
    ht_area = problem.get_val("data:geometry:horizontal_tail:area", units="m**2")
    # FIXME: error on obtained value!
    assert ht_area == pytest.approx(6.00, abs=1e-2)  # old-version obtained value 3.9m²


def test_compute_static_margin():
    """ Tests computation of static margin """

    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    input_vars = reader.read().to_ivc()
    input_vars.add_output("data:weight:aircraft:CG:aft:MAC_position", 0.20)

    problem = run_system(ComputeStaticMargin(), input_vars)
    static_margin = problem["data:handling_qualities:static_margin"]
    assert static_margin == pytest.approx(0.58, rel=1e-2)
