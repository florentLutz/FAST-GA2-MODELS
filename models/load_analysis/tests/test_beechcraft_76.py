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

import pandas as pd
import numpy as np
from openmdao.core.component import Component
import pytest
from typing import Union

from fastoad.utils.physics import Atmosphere
from fastoad.module_management.service_registry import RegisterPropulsion
from fastoad import BundleLoader
from fastoad.base.flight_point import FlightPoint
from fastoad.models.propulsion.propulsion import IOMPropulsionWrapper

from ...tests.testing_utilities import run_system, register_wrappers, get_indep_var_comp, list_inputs
from ...propulsion.fuel_propulsion.base import AbstractFuelPropulsion
from ..private.wing_mass_estimation import AerostructuralLoadsAlternate
from ..aerostructural_loads import AerostructuralLoad
from ..structural_loads import StructuralLoads
from ..aerodynamic_loads import AerodynamicLoads
from ...propulsion.propulsion import IPropulsion

XML_FILE = "beechcraft_76.xml"
ENGINE_WRAPPER = "test.wrapper.performances.beechcraft.dummy_engine"


class DummyEngine(AbstractFuelPropulsion):

    def __init__(self):
        """
        Dummy engine model returning thrust in particular conditions defined for htp/vtp areas.

        """
        super().__init__()
        self.max_power = 130000.0
        self.max_thrust = 5800.0 / 2.0

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
        sfc_p_max = 7.96359441e-08  # fixed whatever the thrust ratio, sfc for ONE 130kW engine !
        sfc = sfc_p_max * flight_points.thrust_rate * mach * Atmosphere(altitude).speed_of_sound

        flight_points['sfc'] = sfc

    def compute_weight(self) -> float:
        return 0.0

    def compute_dimensions(self) -> (float, float, float, float):
        return [0.0, 0.0, 0.0, 0.0]

    def compute_drag(self, mach, unit_reynolds, l0_wing):
        return 0.0

    def compute_sl_thrust(self) -> float:
        return 5800.0


@RegisterPropulsion(ENGINE_WRAPPER)
class DummyEngineWrapper(IOMPropulsionWrapper):
    def setup(self, component: Component):
        component.add_input("data:TLAR:v_cruise", np.nan, units="m/s")
        component.add_input("data:mission:sizing:main_route:cruise:altitude", np.nan, units="m")

    @staticmethod
    def get_model(inputs) -> IPropulsion:
        return DummyEngine()


BundleLoader().context.install_bundle(__name__).start()


def test_compute_shear_stress():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerostructuralLoad()), __file__, XML_FILE)
    cl_vector_only_prop = [0.0161, 0.0017, 0.0021, 0.0026, 0.0034, 0.0042, 0.0052, 0.0063, 0.0082, 0.0117, 0.0426,
                           0.0739, 0.1035, 0.1072, 0.0931, 0.0327, 0.0136, 0.001, -0.0069, -0.0062, -0.0259, -0.0192,
                           -0.0147, -0.0117, -0.0097, -0.0082, -0.0071, -0.0065, -0.006, -0.0055, -0.0048, -0.0042,
                           -0.0033, -0.0027, -0.0021, -0.0017, -0.001, -0.0006, 0.0004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0]
    y_vector = [0.04278, 0.12834, 0.21389, 0.29945, 0.38501, 0.47056, 0.55612, 0.67982, 0.84215, 1.00539, 1.16945,
                1.33423, 1.49963, 1.66555, 1.8319, 1.99856, 2.16543, 2.33242, 2.49942, 2.66632, 2.83301, 2.99941,
                3.16539, 3.33087, 3.49574, 3.6599, 3.82325, 3.98571, 4.14717, 4.30755, 4.46676, 4.6247, 4.78131,
                4.9365, 5.0902, 5.24233, 5.39282, 5.5416, 5.68862, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0]
    ivc.add_output("data:aerodynamics:slipstream:wing:only_prop:CL_vector", cl_vector_only_prop)
    ivc.add_output("data:aerodynamics:slipstream:wing:prop_on:Y_vector", y_vector, units="m")
    ivc.add_output("data:aerodynamics:slipstream:wing:prop_on:velocity", 84.368, units="m/s")
    register_wrappers()
    problem = run_system(AerostructuralLoad(), ivc)
    shear_max_mass_condition = problem.get_val("data:loads:max_shear:mass", units="kg")
    assert shear_max_mass_condition == pytest.approx(1639.00, abs=1e-1)
    shear_max_lf_condition = problem.get_val("data:loads:max_shear:load_factor")
    assert shear_max_lf_condition == pytest.approx(4.047, abs=1e-2)
    shear_max_cg_position = problem.get_val("data:loads:max_shear:cg_position", units="m")
    assert shear_max_cg_position == pytest.approx(2.759, abs=1e-2)
    lift_shear_diagram = problem.get_val("data:loads:max_shear:lift_shear", units="N")
    lift_root_shear = lift_shear_diagram[0]
    assert lift_root_shear == pytest.approx(49139.9, abs=1)
    weight_shear_diagram = problem.get_val("data:loads:max_shear:weight_shear", units="N")
    weight_root_shear = weight_shear_diagram[0]
    assert weight_root_shear == pytest.approx(-23378.89, abs=1)


def test_compute_root_bending_moment():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerostructuralLoad()), __file__, XML_FILE)
    cl_vector_only_prop = [0.0161, 0.0017, 0.0021, 0.0026, 0.0034, 0.0042, 0.0052, 0.0063, 0.0082, 0.0117, 0.0426,
                           0.0739, 0.1035, 0.1072, 0.0931, 0.0327, 0.0136, 0.001, -0.0069, -0.0062, -0.0259, -0.0192,
                           -0.0147, -0.0117, -0.0097, -0.0082, -0.0071, -0.0065, -0.006, -0.0055, -0.0048, -0.0042,
                           -0.0033, -0.0027, -0.0021, -0.0017, -0.001, -0.0006, 0.0004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0]
    y_vector = [0.04278, 0.12834, 0.21389, 0.29945, 0.38501, 0.47056, 0.55612, 0.67982, 0.84215, 1.00539, 1.16945,
                1.33423, 1.49963, 1.66555, 1.8319, 1.99856, 2.16543, 2.33242, 2.49942, 2.66632, 2.83301, 2.99941,
                3.16539, 3.33087, 3.49574, 3.6599, 3.82325, 3.98571, 4.14717, 4.30755, 4.46676, 4.6247, 4.78131,
                4.9365, 5.0902, 5.24233, 5.39282, 5.5416, 5.68862, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0]
    ivc.add_output("data:aerodynamics:slipstream:wing:only_prop:CL_vector", cl_vector_only_prop)
    ivc.add_output("data:aerodynamics:slipstream:wing:prop_on:Y_vector", y_vector, units="m")
    ivc.add_output("data:aerodynamics:slipstream:wing:prop_on:velocity", 84.368, units="m/s")
    register_wrappers()
    problem = run_system(AerostructuralLoad(), ivc)
    max_rbm_mass_condition = problem.get_val("data:loads:max_rbm:mass", units="kg")
    assert max_rbm_mass_condition == pytest.approx(1639.00, abs=1e-1)
    max_rbm_lf_condition = problem.get_val("data:loads:max_rbm:load_factor")
    assert max_rbm_lf_condition == pytest.approx(4.047, abs=1e-2)
    max_rbm_cg_position = problem.get_val("data:loads:max_rbm:cg_position", units="m")
    assert max_rbm_cg_position == pytest.approx(2.759, abs=1e-2)
    lift_rbm_diagram = problem.get_val("data:loads:max_rbm:lift_rbm", units="N*m")
    lift_rbm = lift_rbm_diagram[0]
    assert lift_rbm == pytest.approx(130480.12, abs=1)
    weight_rbm_diagram = problem.get_val("data:loads:max_rbm:weight_rbm", units="N*m")
    weight_rbm = weight_rbm_diagram[0]
    assert weight_rbm == pytest.approx(-57044.03, abs=1)


def test_compute_aerostructural_load_alternate():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerostructuralLoadsAlternate()), __file__, XML_FILE)
    cl_vector_only_prop = [0.0161, 0.0017, 0.0021, 0.0026, 0.0034, 0.0042, 0.0052, 0.0063, 0.0082, 0.0117, 0.0426,
                           0.0739, 0.1035, 0.1072, 0.0931, 0.0327, 0.0136, 0.001, -0.0069, -0.0062, -0.0259, -0.0192,
                           -0.0147, -0.0117, -0.0097, -0.0082, -0.0071, -0.0065, -0.006, -0.0055, -0.0048, -0.0042,
                           -0.0033, -0.0027, -0.0021, -0.0017, -0.001, -0.0006, 0.0004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0]
    y_vector = [0.04278, 0.12834, 0.21389, 0.29945, 0.38501, 0.47056, 0.55612, 0.67982, 0.84215, 1.00539, 1.16945,
                1.33423, 1.49963, 1.66555, 1.8319, 1.99856, 2.16543, 2.33242, 2.49942, 2.66632, 2.83301, 2.99941,
                3.16539, 3.33087, 3.49574, 3.6599, 3.82325, 3.98571, 4.14717, 4.30755, 4.46676, 4.6247, 4.78131,
                4.9365, 5.0902, 5.24233, 5.39282, 5.5416, 5.68862, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0]
    ivc.add_output("data:aerodynamics:slipstream:wing:only_prop:CL_vector", cl_vector_only_prop)
    ivc.add_output("data:aerodynamics:slipstream:wing:prop_on:Y_vector", y_vector, units="m")
    ivc.add_output("data:aerodynamics:slipstream:wing:prop_on:velocity", 84.368, units="m/s")
    register_wrappers()
    problem = run_system(AerostructuralLoadsAlternate(), ivc)
    web_mass = problem.get_val("data:weight:airframe:wing:primary_structure:web_mass", units="kg")
    assert web_mass == pytest.approx(0.856, abs=1e-3)
    upper_flange_mass = problem.get_val("data:weight:airframe:wing:primary_structure:upper_flange_mass", units="kg")
    assert upper_flange_mass == pytest.approx(5.806, abs=1e-3)
    lower_flange_mass = problem.get_val("data:weight:airframe:wing:primary_structure:lower_flange_mass", units="kg")
    assert lower_flange_mass == pytest.approx(7.758, abs=1e-3)
    skin_mass = problem.get_val("data:weight:airframe:wing:primary_structure:skin_mass", units="kg")
    assert skin_mass == pytest.approx(75.718, abs=1e-3)
    ribs_mass = problem.get_val("data:weight:airframe:wing:primary_structure:ribs_mass", units="kg")
    assert ribs_mass == pytest.approx(17.675, abs=1e-3)
    misc_mass = problem.get_val("data:weight:airframe:wing:primary_structure:misc_mass", units="kg")
    assert misc_mass == pytest.approx(36.683, abs=1e-3)
    secondary_structure_mass = problem.get_val("data:weight:airframe:wing:secondary_structure:mass", units="kg")
    assert secondary_structure_mass == pytest.approx(48.048, abs=1e-3)
    wing_mass = problem.get_val("data:weight:airframe:wing:analytical_mass", units="kg")
    assert wing_mass == pytest.approx(192.547, abs=1e-3)


def test_compute_mass_distribution():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(StructuralLoads()), __file__, XML_FILE)
    load_factor_shear = 4.0
    ivc.add_output("data:loads:max_shear:load_factor", load_factor_shear)
    load_factor_rbm = 4.0
    ivc.add_output("data:loads:max_rbm:load_factor", load_factor_rbm)

    register_wrappers()
    problem = run_system(StructuralLoads(), ivc)
    point_mass_array = problem.get_val("data:loads:structure:ultimate:force_distribution:point_mass", units="N/m")
    point_mass_result = np.array([-0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -23056.08, -23056.08, -23056.08,
                                  -23056.08, -23056.08, -23056.08, -0., -0., -0., -0., -0., -0., -0., -0., -146585.61,
                                  -146585.61, -146585.61, -146585.61, -146585.61, -0., -0., -0., -0., -0., -0., -0.,
                                  -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
                                  -0., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0.])
    assert np.max(np.abs(point_mass_array - point_mass_result)) <= 1e-1
    wing_mass_array = problem.get_val("data:loads:structure:ultimate:force_distribution:wing", units="N/m")
    wing_mass_result = np.array([-617.88, -614.61, -614.61, -614.61, -614.61, -614.61, -614.61,
                                 -614.61, -614.61, -614.61, -614.61, -614.61, -614.61, -614.61,
                                 -614.61, -614.61, -614.61, -614.61, -614.61, -614.61, -614.61,
                                 -614.61, -614.61, -614.61, -614.61, -614.61, -614.61, -614.61,
                                 -614.61, -614.61, -614.61, -614.61, -614.61, -614.61, -614.61,
                                 -614.61, -614.61, -614.61, -614.61, -614.61, -614.61, -614.61,
                                 -614.61, -614.61, -614.61, -614.61, -614.61, -614.61, -614.61,
                                 -614.61, -614.61, -614.61, -614.61, -614.61, -617.88, 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert np.max(np.abs(wing_mass_result - wing_mass_array)) <= 1e-1
    fuel_mass_array = problem.get_val("data:loads:structure:ultimate:force_distribution:fuel", units="N/m")
    fuel_mass_result = np.array([-169.47, -168.57, -168.57, -168.57, -168.57, -168.57, -168.57,
                                 -168.57, -168.57, -168.57, -168.57, -168.57, -842.87, -842.87,
                                 -842.87, -842.87, -842.87, -842.87, -842.87, -842.87, -421.44,
                                 -421.44, -421.44, -421.44, -421.44, -421.44, -421.44, -421.44,
                                 -421.44, -421.44, -421.44, -421.44, -421.44, -842.87, -842.87,
                                 -842.87, -842.87, -842.87, -842.87, -842.87, -842.87, -842.87,
                                 -842.87, -842.87, -842.87, -842.87, -842.87, -842.87, -842.87,
                                 -842.87, -842.87, -842.87, -842.87, -842.87, -847.36, 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    assert np.max(np.abs(fuel_mass_result - fuel_mass_array)) <= 1e-1


def test_compute_structure_shear():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(StructuralLoads()), __file__, XML_FILE)
    load_factor_shear = 4.0
    ivc.add_output("data:loads:max_shear:load_factor", load_factor_shear)
    load_factor_rbm = 4.0
    ivc.add_output("data:loads:max_rbm:load_factor", load_factor_rbm)

    register_wrappers()
    problem = run_system(StructuralLoads(), ivc)
    point_mass_array = problem.get_val("data:loads:structure:ultimate:shear:point_mass", units="N")
    point_mass_result = np.array([-1997.25, -1997.25, -1997.25, -1997.25, -1997.25, -1997.25, -1997.25, -1997.25,
                                  -1997.25, -1997.25, -1994.37, -1927.95, -1861.52, -1841.15, -1795.10, -1728.68,
                                  -1725.80, -1725.80, -1725.80, -1725.80, -1725.80, -1725.80, -1725.80, -1725.80,
                                  -1707.48, -1285.19, -862.90, -440.61, -18.32, 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    point_mass_result *= load_factor_shear
    assert np.max(np.abs(point_mass_array - point_mass_result)) <= 1e-1
    wing_mass_array = problem.get_val("data:loads:structure:ultimate:shear:wing", units="N")
    wing_mass_result = np.array([-3541.36, -3514.99, -3462.41, -3409.83, -3357.24, -3304.66,
                                 -3252.08, -3199.49, -3123.46, -3040.65, -3040.03, -3032.95,
                                 -3025.87, -3023.69, -3018.78, -3011.7, -3011.09, -2923.37,
                                 -2822.53, -2721.26, -2619.6, -2517.63, -2415.39, -2352.07,
                                 -2351.45, -2344.37, -2337.29, -2330.21, -2323.12, -2322.51,
                                 -2312.96, -2210.4, -2107.77, -2005.13, -1902.55, -1800.1,
                                 -1697.83, -1595.82, -1494.11, -1392.78, -1291.89, -1191.49,
                                 -1091.64, -992.41, -893.84, -795.99, -698.92, -602.66,
                                 -507.28, -412.82, -319.32, -226.82, -135.38, -45.02,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert np.max(np.abs(wing_mass_result - wing_mass_array)) <= 1e-1
    fuel_mass_array = problem.get_val("data:loads:structure:ultimate:shear:fuel", units="N")
    fuel_mass_result = np.array([-3873.94, -3866.71, -3852.29, -3837.87, -3823.44, -3809.02,
                                 -3794.6, -3780.17, -3759.32, -3736.61, -3736.44, -3734.5,
                                 -3728.67, -3725.69, -3718.95, -3709.24, -3708.4, -3588.1,
                                 -3449.82, -3310.93, -3206.37, -3136.44, -3066.34, -3022.92,
                                 -3022.5, -3017.64, -3012.79, -3007.93, -3003.07, -3002.65,
                                 -2996.1, -2925.78, -2855.4, -2749.83, -2609.16, -2468.66,
                                 -2328.4, -2188.5, -2049.03, -1910.06, -1771.7, -1634.01,
                                 -1497.08, -1360.99, -1225.81, -1091.62, -958.49, -826.49,
                                 -695.69, -566.14, -437.91, -311.07, -185.66, -61.75,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert np.max(np.abs(fuel_mass_result - fuel_mass_array)) <= 1e-1


def test_compute_structure_bending():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(StructuralLoads()), __file__, XML_FILE)
    load_factor_shear = 4.0
    ivc.add_output("data:loads:max_shear:load_factor", load_factor_shear)
    load_factor_rbm = 4.0
    ivc.add_output("data:loads:max_rbm:load_factor", load_factor_rbm)

    register_wrappers()
    problem = run_system(StructuralLoads(), ivc)
    point_mass_array = problem.get_val("data:loads:structure:ultimate:root_bending:point_mass", units="N*m")
    point_mass_result = np.array([-14433.78, -14092.01, -13408.47, -12725.01, -12041.47, -11357.93, -10674.47, -9990.93,
                                  -9002.69, -7926.18, -7918.19, -7827.79, -7740.46, -7714.28, -7656.19, -7574.97,
                                  -7568.07, -6582.79, -5450.24, -4312.73, -3170.94, -2025.55, -877.2, -166., -159.1,
                                  -90.13, -40.62, -10.58, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    assert np.max(np.abs(point_mass_array - point_mass_result)) <= 1e-1
    wing_mass_array = problem.get_val("data:loads:structure:ultimate:root_bending:wing", units="N*m")
    wing_mass_result = np.array([-10202.22, -10051.29, -9752.8, -9458.84, -9169.34, -8884.34,
                                 -8603.88, -8327.88, -7936.81, -7521.51, -7518.47, -7483.47,
                                 -7448.57, -7437.87, -7413.74, -7378.99, -7375.98, -6952.47,
                                 -6481.14, -6024.39, -5582.7, -5156.51, -4746.21, -4500.62,
                                 -4498.27, -4471.22, -4444.24, -4417.35, -4390.54, -4388.22,
                                 -4352.19, -3974.79, -3614.24, -3270.82, -2944.72, -2636.12,
                                 -2345.1, -2071.76, -1816.1, -1578.12, -1357.76, -1154.93,
                                 -969.47, -801.22, -649.97, -515.45, -397.39, -295.47,
                                 -209.35, -138.64, -82.95, -41.86, -14.91, -1.65,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    assert np.max(np.abs(wing_mass_result - wing_mass_array)) <= 1e-1
    fuel_mass_array = problem.get_val("data:loads:structure:ultimate:root_bending:fuel", units="N*m")
    fuel_mass_result = np.array([-12950.68, -12785.11, -12454.89, -12125.95, -11798.2, -11471.68,
                                 -11146.43, -10822.39, -10356.07, -9851.04, -9847.3, -9804.25,
                                 -9761.23, -9748.06, -9718.32, -9675.52, -9671.81, -9151.1,
                                 -8573.78, -8016.77, -7480.67, -6954.47, -6438.55, -6124.88,
                                 -6121.86, -6087.06, -6052.31, -6017.62, -5982.99, -5979.98,
                                 -5933.37, -5439.28, -4956.58, -4485.6, -4038.39, -3615.18,
                                 -3216.07, -2841.21, -2490.6, -2164.23, -1862.03, -1583.87,
                                 -1329.53, -1098.8, -891.36, -706.88, -544.99, -405.21,
                                 -287.1, -190.13, -113.76, -57.4, -20.45, -2.26,
                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    assert np.max(np.abs(fuel_mass_result - fuel_mass_array)) <= 1e-1


def test_compute_lift_distribution():
    # Research independent input value in .xml file
    ivc = get_indep_var_comp(list_inputs(AerodynamicLoads()), __file__, XML_FILE)
    load_factor_shear = 4.0
    ivc.add_output("data:loads:max_shear:load_factor", load_factor_shear)
    ivc.add_output("data:loads:max_shear:cg_position", 2.759, units="m")
    ivc.add_output("data:loads:max_shear:mass", 1638.94, units="kg")
    load_factor_rbm = 4.0
    ivc.add_output("data:loads:max_rbm:load_factor", load_factor_rbm)
    ivc.add_output("data:loads:max_rbm:cg_position", 2.759, units="m")
    ivc.add_output("data:loads:max_rbm:mass", 1638.94, units="kg")
    cl_vector_only_prop = [0.0161, 0.0017, 0.0021, 0.0026, 0.0034, 0.0042, 0.0052, 0.0063, 0.0082, 0.0117, 0.0426,
                           0.0739, 0.1035, 0.1072, 0.0931, 0.0327, 0.0136, 0.001, -0.0069, -0.0062, -0.0259, -0.0192,
                           -0.0147, -0.0117, -0.0097, -0.0082, -0.0071, -0.0065, -0.006, -0.0055, -0.0048, -0.0042,
                           -0.0033, -0.0027, -0.0021, -0.0017, -0.001, -0.0006, 0.0004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0]
    y_vector = [0.04278, 0.12834, 0.21389, 0.29945, 0.38501, 0.47056, 0.55612, 0.67982, 0.84215, 1.00539, 1.16945,
                1.33423, 1.49963, 1.66555, 1.8319, 1.99856, 2.16543, 2.33242, 2.49942, 2.66632, 2.83301, 2.99941,
                3.16539, 3.33087, 3.49574, 3.6599, 3.82325, 3.98571, 4.14717, 4.30755, 4.46676, 4.6247, 4.78131,
                4.9365, 5.0902, 5.24233, 5.39282, 5.5416, 5.68862, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0]
    ivc.add_output("data:aerodynamics:slipstream:wing:only_prop:CL_vector", cl_vector_only_prop)
    ivc.add_output("data:aerodynamics:slipstream:wing:prop_on:Y_vector", y_vector, units="m")
    ivc.add_output("data:aerodynamics:slipstream:wing:prop_on:velocity", 84.368, units="m/s")

    register_wrappers()
    problem = run_system(AerodynamicLoads(), ivc)
    lift_array = problem.get_val("data:loads:aerodynamic:ultimate:force_distribution", units="N/m")
    lift_result = np.array([6184.99, 6152.22, 6083.25, 6079.98, 6076.25, 6076.38, 6078.89, 6078.57, 6052.07, 6040.45,
                            6040.37, 6039.37, 6038.38, 6038.08, 6039.97, 6042.7, 6042.93, 6076.74, 6233.1, 6381.43,
                            6502.82, 6507.21, 6440.2, 6247.01, 6245.14, 6223.53, 6201.92, 6180.32, 6158.71, 6156.83,
                            6127.69, 6009.4, 5924.52, 5854.2, 5825.73, 5682.57, 5678.86, 5659.57, 5630.51, 5579.86,
                            5531.5, 5460.7, 5391.25, 5307.02, 5227.54, 5130.96, 5025.3, 4894.92, 4751.16, 4543.09,
                            4279.26, 3905.43, 3332.92, 2513.76, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0.])

    assert np.max(np.abs(lift_array - lift_result)) <= 1e-1