"""
Test load_analysis module
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
import pytest
from typing import Union

from fastoad.io import VariableIO
from fastoad.utils.physics import Atmosphere
from fastoad.module_management.service_registry import RegisterPropulsion
from fastoad import BundleLoader
from fastoad.base.flight_point import FlightPoint
from fastoad.models.propulsion.propulsion import IOMPropulsionWrapper

from ...tests.testing_utilities import run_system, register_wrappers, get_indep_var_comp, list_inputs, Timer
from ...propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from ..aerostructural_loads import AerostructuralLoad
from ..private.wing_mass_estimation import AerostructuralLoadsAlternate
from ...propulsion.propulsion import IPropulsion

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
            self.max_thrust * sigma ** (1. / 3.),
            max_power * 0.8 / np.maximum(mach * Atmosphere(altitude).speed_of_sound, 1e-20)
        )
        if flight_points.thrust_rate is None:
            flight_points.thrust = min(max_thrust, float(thrust))
            flight_points.thrust_rate = float(thrust) / max_thrust
        else:
            flight_points.thrust = max_thrust * np.array(flight_points.thrust_rate)
        sfc_pmax = 8.5080e-08  # fixed whatever the thrust ratio, sfc for ONE 130kW engine !
        sfc = sfc_pmax * flight_points.thrust_rate * mach * Atmosphere(altitude).speed_of_sound

        flight_points['sfc'] = sfc

    def compute_weight(self) -> float:
        return 0.0

    def compute_dimensions(self) -> (float, float, float, float):
        return [0.0, 0.0, 0.0, 0.0]

    def compute_drag(self, mach, unit_reynolds, l0_wing):
        return 0.0

    def compute_sl_thrust(self) -> float:
        return 5417.0


@RegisterPropulsion(ENGINE_WRAPPER)
class DummyEngineWrapper(IOMPropulsionWrapper):
    def setup(self, component: Component):
        component.add_input("data:TLAR:v_cruise", np.nan, units="m/s")
        component.add_input("data:mission:sizing:main_route:cruise:altitude", np.nan, units="m")

    @staticmethod
    def get_model(inputs) -> IPropulsion:
        return DummyEngine()


BundleLoader().context.install_bundle(__name__).start()


def _test_compute_shear_stress():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerostructuralLoad()), __file__, XML_FILE)

    register_wrappers()
    problem = run_system(AerostructuralLoad(), ivc)
    shear_max_mass_condition = problem.get_val("data:loads:max_shear:mass", units="kg")
    assert shear_max_mass_condition == pytest.approx(1642.45, abs=1e-1)
    shear_max_lf_condition = problem.get_val("data:loads:max_shear:load_factor")
    assert shear_max_lf_condition == pytest.approx(3.996, abs=1e-2)
    shear_max_cg_position = problem.get_val("data:loads:max_shear:cg_position", units="m")
    assert shear_max_cg_position == pytest.approx(2.432, abs=1e-2)
    lift_shear_diagram = problem.get_val("data:loads:max_shear:lift_shear", units="N")
    lift_root_shear = lift_shear_diagram[0]
    assert lift_root_shear == pytest.approx(53047.77, abs=1)
    weight_shear_diagram = problem.get_val("data:loads:max_shear:weight_shear", units="N")
    weight_root_shear = weight_shear_diagram[0]
    assert weight_root_shear == pytest.approx(-13875.347, abs=1)


def _test_compute_root_bending_moment():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerostructuralLoad()), __file__, XML_FILE)

    register_wrappers()
    problem = run_system(AerostructuralLoad(), ivc)
    max_rbm_mass_condition = problem.get_val("data:loads:max_rbm:mass", units="kg")
    assert max_rbm_mass_condition == pytest.approx(1642.45, abs=1e-1)
    max_rbm_lf_condition = problem.get_val("data:loads:max_rbm:load_factor")
    assert max_rbm_lf_condition == pytest.approx(3.996, abs=1e-2)
    max_rbm_cg_position = problem.get_val("data:loads:max_rbm:cg_position", units="m")
    assert max_rbm_cg_position == pytest.approx(2.432, abs=1e-2)
    lift_rbm_diagram = problem.get_val("data:loads:max_rbm:lift_rbm", units="N*m")
    lift_rbm = lift_rbm_diagram[0]
    assert lift_rbm == pytest.approx(135064.087, abs=1)
    weight_rbm_diagram = problem.get_val("data:loads:max_rbm:weight_rbm", units="N*m")
    weight_rbm = weight_rbm_diagram[0]
    assert weight_rbm == pytest.approx(-32402.56, abs=1)


def test_compute_aerostructural_load_alternate():

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerostructuralLoadsAlternate()), __file__, XML_FILE)

    register_wrappers()
    problem = run_system(AerostructuralLoadsAlternate(), ivc)
    web_mass = problem.get_val("data:weight:airframe:wing:primary_structure:web_mass", units="kg")
    assert web_mass == pytest.approx(1.278, abs=1e-3)
    upper_flange_mass = problem.get_val("data:weight:airframe:wing:primary_structure:upper_flange_mass", units="kg")
    assert upper_flange_mass == pytest.approx(6.049, abs=1e-3)
    lower_flange_mass = problem.get_val("data:weight:airframe:wing:primary_structure:lower_flange_mass", units="kg")
    assert lower_flange_mass == pytest.approx(8.081, abs=1e-3)
    skin_mass = problem.get_val("data:weight:airframe:wing:primary_structure:skin_mass", units="kg")
    assert skin_mass == pytest.approx(101.647, abs=1e-3)
    ribs_mass = problem.get_val("data:weight:airframe:wing:primary_structure:ribs_mass", units="kg")
    assert ribs_mass == pytest.approx(10.190, abs=1e-3)
    misc_mass = problem.get_val("data:weight:airframe:wing:primary_structure:misc_mass", units="kg")
    assert misc_mass == pytest.approx(28.444, abs=1e-3)
    secondary_structure_mass = problem.get_val("data:weight:airframe:wing:secondary_structure:mass", units="kg")
    assert secondary_structure_mass == pytest.approx(51.897, abs=1e-3)
    wing_mass = problem.get_val("data:weight:airframe:wing:analytical_mass", units="kg")
    assert wing_mass == pytest.approx(207.589, abs=1e-3)
