"""Simple module for complete mission."""
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

import numpy as np
import warnings
import math
import openmdao.api as om

from fastoad import BundleLoader
from fastoad.base.flight_point import FlightPoint
from fastoad.constants import EngineSetting
from fastoad.models.propulsion.fuel_propulsion.base import FuelEngineSet
from fastoad.utils.physics import Atmosphere
from scipy.constants import g
from .takeoff import SAFETY_HEIGHT

TIME_STEP = 1.0  # For time dependent simulation


class Mission(om.Group):
    
    def setup(self):
        self.add_subsystem("taxi_out", _compute_taxi(), promotes=["*"])
        self.add_subsystem("taxi_in", _compute_taxi(taxi_out=False), promotes=["*"])
        self.add_subsystem("climb", _compute_climb(), promotes=["*"])
        self.add_subsystem("cruise", _compute_cruise(), promotes=["*"])
        self.add_subsystem("descent", _compute_descent(), promotes=["*"])


class _compute_taxi(om.ExplicitComponent):
    """
    Compute the fuel consumption for taxi based on speed and duration.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)
        self.options.declare("taxi_out", default=True, types=bool)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:propulsion:engine:count", np.nan)
        if self.options["taxi_out"]:
            self.add_input("data:mission:sizing:taxi_out:thrust_rate", np.nan)
            self.add_input("data:mission:sizing:taxi_out:duration", np.nan, units="s")
            self.add_input("data:mission:sizing:taxi_out:speed", np.nan, units="m/s")
            self.add_output("data:mission:sizing:taxi_out:fuel", units='kg')
        else:
            self.add_input("data:mission:sizing:taxi_in:thrust_rate", np.nan)
            self.add_input("data:mission:sizing:taxi_in:duration", np.nan, units="s")
            self.add_input("data:mission:sizing:taxi_in:speed", np.nan, units="m/s")
            self.add_output("data:mission:sizing:taxi_in:fuel", units='kg')

        self.declare_partials("*", "*", method="fd") 

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:engine:count"]
        )
        if self.options["taxi_out"]:
            thrust_rate = inputs["data:mission:sizing:taxi_out:thrust_rate"]
            duration = inputs["data:mission:sizing:taxi_out:duration"]
            mach = inputs["data:mission:sizing:taxi_out:speed"]/Atmosphere(0.0).speed_of_sound
        else:
            thrust_rate = inputs["data:mission:sizing:taxi_in:thrust_rate"]
            duration = inputs["data:mission:sizing:taxi_in:duration"]
            mach = inputs["data:mission:sizing:taxi_in:speed"] / Atmosphere(0.0).speed_of_sound

        # FIXME: no specific settings for taxi (to be changed in fastoad\constants.py)
        flight_point = FlightPoint(
            mach=mach, altitude=0.0, engine_setting=EngineSetting.TAKEOFF,
            thrust_rate=thrust_rate
        )
        propulsion_model.compute_flight_points(flight_point)
        fuel_mass = propulsion_model.get_consumed_mass(flight_point, duration)

        if self.options["taxi_out"]:
            outputs["data:mission:sizing:taxi_out:fuel"] = fuel_mass
        else:
            outputs["data:mission:sizing:taxi_in:fuel"] = fuel_mass


class _compute_climb(om.ExplicitComponent):
    """
    Compute the fuel consumption on climb segment with constant VCAS and fixed thrust ratio.
    The hypothesis of small alpha/gamma angles is done.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:propulsion:engine:count", np.nan)
        self.add_input("data:mission:sizing:main_route:cruise:altitude", np.nan, units="m")
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:geometry:wing:area", np.nan, units="m**2")
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_out:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:holding:fuel", 0.0, units="kg")
        self.add_input("data:mission:sizing:takeoff:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:climb:thrust_rate", np.nan)

        self.add_output("data:mission:sizing:main_route:climb:fuel", units="kg")
        self.add_output("data:mission:sizing:main_route:climb:distance", units="m")
        self.add_output("data:mission:sizing:main_route:climb:duration", units="s")
        self.add_output("data:mission:sizing:main_route:climb:v_cas", units="m/s")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:engine:count"]
        )
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        coef_k = inputs["data:aerodynamics:aircraft:cruise:induced_drag_coefficient"]
        wing_area = inputs["data:geometry:wing:area"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        m_to = inputs["data:mission:sizing:taxi_out:fuel"]
        m_ho = inputs["data:mission:sizing:holding:fuel"]
        m_tk = inputs["data:mission:sizing:takeoff:fuel"]
        m_ic = inputs["data:mission:sizing:initial_climb:fuel"]
        thrust_rate = inputs["data:mission:sizing:main_route:climb:thrust_rate"]

        # Define initial conditions
        altitude_t = SAFETY_HEIGHT  # conversion to m
        distance_t = 0.0
        time_t = 0.0
        mass_t = mtow - (m_to + m_ho + m_tk + m_ic)
        mass_fuel_t = 0.0
        atm_0 = Atmosphere(0.0)

        # FIXME: VCAS strategy is specific to ICE-propeller configuration, should be an input
        cl = math.sqrt(3*cd0/coef_k)
        atm = Atmosphere(altitude_t, altitude_in_feet=False)
        v_cas = math.sqrt((mass_t * g) / (0.5 * atm.density * wing_area * cl))

        while altitude_t < cruise_altitude:

            # Define air properties
            atm = Atmosphere(altitude_t, altitude_in_feet=False)
            v_tas = v_cas * math.sqrt(atm_0.density / atm.density)

            # Evaluate thrust and sfc
            mach = math.sqrt(
                5 * ((atm_0.pressure / atm.pressure * (
                        (1 + 0.2 * (v_cas / atm_0.speed_of_sound) ** 2) ** 3.5 - 1
                ) + 1) ** (1 / 3.5) - 1)
            )
            flight_point = FlightPoint(
                mach=mach, altitude=altitude_t, engine_setting=EngineSetting.CLIMB,
                thrust_rate=thrust_rate
            )  # with engine_setting as EngineSetting
            propulsion_model.compute_flight_points(flight_point)
            thrust = float(flight_point.thrust)

            # Calculates cl and drag considering constant climb rate
            cl = mass_t * g / (0.5 * atm.density * wing_area * v_tas**2)
            cd = cd0 + coef_k * cl**2

            # Calculate climb rate and height increase
            climb_rate = thrust / (mass_t * g) - cd / cl
            vz = v_tas * math.sin(climb_rate)
            vx = v_tas * math.cos(climb_rate)
            altitude_t += vz * TIME_STEP
            distance_t += vx * TIME_STEP

            # Estimate mass evolution and update time
            mass_fuel_t += propulsion_model.get_consumed_mass(flight_point, TIME_STEP)
            mass_t = mass_t - propulsion_model.get_consumed_mass(flight_point, TIME_STEP)
            time_t += TIME_STEP

        outputs["data:mission:sizing:main_route:climb:fuel"] = mass_fuel_t
        outputs["data:mission:sizing:main_route:climb:distance"] = distance_t
        outputs["data:mission:sizing:main_route:climb:duration"] = time_t
        outputs["data:mission:sizing:main_route:climb:v_cas"] = v_cas


class _compute_cruise(om.ExplicitComponent):
    """
    Compute the fuel consumption on cruise segment with constant VTAS and altitude.
    The hypothesis of small alpha/gamma angles is done.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:propulsion:engine:count", np.nan)
        self.add_input("data:TLAR:v_cruise", np.nan, units="m/s")
        self.add_input("data:TLAR:range", np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", np.nan, units="m")
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:geometry:wing:area", np.nan, units="m**2")
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_out:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:holding:fuel", 0.0, units="kg")
        self.add_input("data:mission:sizing:takeoff:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:climb:distance", np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:descent:distance", 0.0, units="m")

        self.add_output("data:mission:sizing:main_route:cruise:fuel", units="kg")
        self.add_output("data:mission:sizing:main_route:cruise:distance", units="m")
        self.add_output("data:mission:sizing:main_route:cruise:duration", units="s")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:engine:count"]
        )
        v_tas = inputs["data:TLAR:v_cruise"]
        cruise_distance = max(
            0.0,
            (
                inputs["data:TLAR:range"]
                - inputs["data:mission:sizing:main_route:climb:distance"]
                - inputs["data:mission:sizing:main_route:descent:distance"]
            )
        )
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        coef_k = inputs["data:aerodynamics:aircraft:cruise:induced_drag_coefficient"]
        wing_area = inputs["data:geometry:wing:area"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        m_to = inputs["data:mission:sizing:taxi_out:fuel"]
        m_ho = inputs["data:mission:sizing:holding:fuel"]
        m_tk = inputs["data:mission:sizing:takeoff:fuel"]
        m_ic = inputs["data:mission:sizing:initial_climb:fuel"]
        m_cl = inputs["data:mission:sizing:main_route:climb:fuel"]

        # Define specific time step ~1000 points for calculation
        time_step = (cruise_distance / v_tas) / 1000.0

        # Define initial conditions
        distance_t = 0.0
        time_t = 0.0
        mass_fuel_t = 0.0
        mass_t = mtow - (m_to + m_ho + m_tk + m_ic + m_cl)
        atm_0 = Atmosphere(0.0)
        atm = Atmosphere(cruise_altitude, altitude_in_feet=False)
        v_cas = v_tas / math.sqrt(atm_0.density / atm.density)

        while distance_t < cruise_distance:

            # Calculate Cl - Cd and corresponding drag
            cl = mass_t * g / (0.5 * atm.density * wing_area * v_tas ** 2)
            cd = cd0 + coef_k * cl ** 2
            drag = 0.5 * atm.density * wing_area * cd * v_tas ** 2

            # Evaluate sfc
            mach = math.sqrt(
                5 * ((atm_0.pressure / atm.pressure * (
                        (1 + 0.2 * (v_cas / atm_0.speed_of_sound) ** 2) ** 3.5 - 1
                ) + 1) ** (1 / 3.5) - 1)
            )
            flight_point = FlightPoint(
                mach=mach, altitude=cruise_altitude, engine_setting=EngineSetting.CRUISE,
                thrust_is_regulated=True, thrust=drag,
            )
            propulsion_model.compute_flight_points(flight_point)
            # If thrust exceed max thrust exit cruise calculation
            if float(flight_point.thrust_rate) > 1.0:
                warnings.warn("The cruise strategy exceeds propulsion power!")
                mass_fuel_t = 0.0
                time_t = 0.0
                break

            # Calculate distance increase
            distance_t += v_tas * min(time_step, (cruise_distance - distance_t) / v_tas)

            # Estimate mass evolution and update time
            mass_fuel_t += propulsion_model.get_consumed_mass(
                flight_point,
                min(time_step, (cruise_distance - distance_t) / v_tas)
            )
            mass_t = mass_t - propulsion_model.get_consumed_mass(
                flight_point,
                min(time_step, (cruise_distance - distance_t) / v_tas)
            )
            time_t += min(time_step, (cruise_distance - distance_t) / v_tas)

        outputs["data:mission:sizing:main_route:cruise:fuel"] = mass_fuel_t
        outputs["data:mission:sizing:main_route:cruise:distance"] = distance_t
        outputs["data:mission:sizing:main_route:cruise:duration"] = time_t


class _compute_descent(om.ExplicitComponent):
    """
    Compute the fuel consumption on descent segment with constant VCAS and descent
    rate.
    The hypothesis of small alpha angle is done.
    Warning: Descent rate is reduced if cd/cl < abs(desc_rate)!
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:propulsion:engine:count", np.nan)
        self.add_input("data:mission:sizing:main_route:cruise:altitude", np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:descent:descent_rate", np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:optimal_CL", np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:induced_drag_coefficient", np.nan)
        self.add_input("data:geometry:wing:area", np.nan, units="m**2")
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_out:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:holding:fuel", 0.0, units="kg")
        self.add_input("data:mission:sizing:takeoff:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:cruise:fuel", np.nan, units="kg")

        self.add_output("data:mission:sizing:main_route:descent:fuel", units="kg")
        self.add_output("data:mission:sizing:main_route:descent:distance", units="m")
        self.add_output("data:mission:sizing:main_route:descent:duration", units="s")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:engine:count"]
        )
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        descent_rate = -abs(inputs["data:mission:sizing:main_route:descent:descent_rate"])
        cl = inputs["data:aerodynamics:aircraft:cruise:optimal_CL"]
        cd0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        coef_k = inputs["data:aerodynamics:aircraft:cruise:induced_drag_coefficient"]
        wing_area = inputs["data:geometry:wing:area"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        m_to = inputs["data:mission:sizing:taxi_out:fuel"]
        m_ho = inputs["data:mission:sizing:holding:fuel"]
        m_tk = inputs["data:mission:sizing:takeoff:fuel"]
        m_ic = inputs["data:mission:sizing:initial_climb:fuel"]
        m_cl = inputs["data:mission:sizing:main_route:climb:fuel"]
        m_cr = inputs["data:mission:sizing:main_route:cruise:fuel"]

        # Define initial conditions
        gamma = math.asin(descent_rate)
        altitude_t = cruise_altitude
        distance_t = 0.0
        time_t = 0.0
        mass_fuel_t = 0.0
        mass_t = mtow - (m_to + m_ho + m_tk + m_ic + m_cl + m_cr)
        atm_0 = Atmosphere(0.0)
        warning = False
        # Calculate defined VCAS at the beginning of descent (cos(gamma)~1)
        v_cas = math.sqrt((mass_t * g) * math.cos(descent_rate) / (0.5 * atm_0.density * wing_area * cl))

        while altitude_t > SAFETY_HEIGHT:

            # Define air properties and calculate VTAS
            atm = Atmosphere(altitude_t, altitude_in_feet=False)
            v_tas = v_cas * math.sqrt(atm_0.density / atm.density)
            # Calculate lift and drag coefficients changes to maintain speed (cos(gamma)~1)
            cl = ((mass_t * g) * math.cos(descent_rate) / (0.5 * atm.density * wing_area * v_tas**2))
            cd = cd0 + coef_k * cl**2
            cl_cd = cl/cd
            drag = 0.5 * atm.density * wing_area * cd * v_tas**2
            # Evaluate mach
            mach = math.sqrt(
                5 * ((atm_0.pressure / atm.pressure * (
                        (1 + 0.2 * (v_cas / atm_0.speed_of_sound) ** 2) ** 3.5 - 1
                ) + 1) ** (1 / 3.5) - 1)
            )
            # Calculate necessary Thrust to maintain VCAS and descent rate
            # if T<0N, VCAS is maintained reducing gamma/descent rate and engine in IDLE condition
            thrust = drag + (mass_t * g) * math.sin(gamma)
            if thrust <= 0.0:
                flight_point = FlightPoint(
                    mach=mach, altitude=altitude_t, engine_setting=EngineSetting.IDLE,
                    thrust_rate=0.2)  # FIXME: define IDLE maybe?
                descent_rate = -1/cl_cd
                gamma = math.asin(descent_rate)
                warning = True
            else:
                # FIXME: DESCENT setting on engine does not exist, replaced by CRUISE for test
                flight_point = FlightPoint(
                    mach=mach, altitude=altitude_t, engine_setting=EngineSetting.CRUISE,
                    thrust_is_regulated=True, thrust=thrust,
                )
            propulsion_model.compute_flight_points(flight_point)

            # Calculate distance increase
            v_x = v_tas * math.cos(descent_rate)
            v_z = v_tas * math.sin(descent_rate)
            distance_t += v_x * TIME_STEP
            altitude_t += v_z * TIME_STEP

            # Estimate mass evolution and update time
            mass_fuel_t += propulsion_model.get_consumed_mass(flight_point, TIME_STEP)
            mass_t = mass_t - propulsion_model.get_consumed_mass(flight_point, TIME_STEP)
            time_t += TIME_STEP

        if warning:
            warnings.warn("Descent rate has been reduced!")

        outputs["data:mission:sizing:main_route:descent:fuel"] = mass_fuel_t
        outputs["data:mission:sizing:main_route:descent:distance"] = distance_t
        outputs["data:mission:sizing:main_route:descent:duration"] = time_t
