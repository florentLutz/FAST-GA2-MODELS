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
import pandas as pd
import math
import openmdao.api as om
from fastoad import BundleLoader
from ..propulsion.fuel_propulsion.base import FuelEngineSet
from fastoad.utils.physics import Atmosphere
from scipy.constants import g

TIME_STEP = 0.05 # For time dependent simulation

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
        """
        Computes thrust, SFC and thrust rate by direct call to engine model.
        Options:
          - propulsion_id: (mandatory) the identifier of the propulsion wrapper.
          - out_file: if provided, a csv file will be written at provided path with all computed
                      flight points. If path is relative, it will be resolved from working
                      directory
        """
        super().__init__(**kwargs)
        self.flight_points = None
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)
        self.options.declare("taxi_out", default=True, types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)
        self.taxi_out = self.options["taxi_out"]

        if self.taxi_out:
            self.add_input("data:mission:sizing:taxi_out:thrust_rate", np.nan)
            self.add_input("data:mission:sizing:taxi_out:duration", np.nan, units="s")
            self.add_input("data:mission:sizing:taxi_out:speed", np.nan, units="m/s")
        else:
            self.add_input("data:mission:sizing:taxi_in:thrust_rate", np.nan)
            self.add_input("data:mission:sizing:taxi_in:duration", np.nan, units="s")
            self.add_input("data:mission:sizing:taxi_in:speed", np.nan, units="m/s")

        self.add_input("data:mission:sizing:takeoff:friction_coefficient_no_brake", np.nan)
        self.add_input("data:aerodynamics:aircraft:takeoff:CD0_clean", np.nan)

        self.add_input("data:geometry:wing:area", np.nan, units='m**2')
        self.add_input("mass", np.nan, units='kg')
        self.add_input("duration", np.nan, units='s')
        self.add_input("speed", np.nan, units='m/s')

        self.add_output("data:mission:operational:taxi_out:fuel", units='kg')
        
        self.declare_partials("*", "*", method="fd") 

    def compute(self, inputs, outputs):

        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:engine:count"]
        )
        if self.taxi_out:
            thrust_rate = inputs["data:mission:sizing:taxi_out:thrust_rate"]
            duration = inputs["data:mission:sizing:taxi_out:duration"]
            speed = inputs["data:mission:sizing:taxi_out:speed"]
        else:
            thrust_rate = inputs["data:mission:sizing:taxi_in:thrust_rate"]
            duration = inputs["data:mission:sizing:taxi_in:duration"]
            speed = inputs["data:mission:sizing:taxi_in:speed"]

        atm =Atmosphere(0.0)
        flight_point = {
            'mach': [speed/atm.speed_of_sound],
            'altitude': [0.0],
            'thrust_is_regulated': [0],
            'thrust_rate': [thrust_rate],
            'thrust': [0.0],
        }
        flight_point = pd.DataFrame(data=flight_point)
        propulsion_model.compute_flight_points(flight_point)
        fuel_mass = (flight_point.sfc * flight_point.Thrust) * duration

        if self.taxi_out:
            outputs["data:mission:operational:taxi_out:fuel"] = fuel_mass
        else:
            outputs["data:mission:operational:taxi_in:fuel"] = fuel_mass

class _compute_climb(om.ExplicitComponent):
    """
    Compute the fuel consumption on climb segment with constant VCAS and fixed thrust ratio.
    The hypothesis of small alpha/gamma angles is done.
    """

    def __init__(self, **kwargs):
        """
        Computes thrust, SFC and thrust rate by direct call to engine model.
        Options:
          - propulsion_id: (mandatory) the identifier of the propulsion wrapper.
          - out_file: if provided, a csv file will be written at provided path with all computed
                      flight points. If path is relative, it will be resolved from working
                      directory
        """
        super().__init__(**kwargs)
        self.flight_points = None
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)
         
        self.add_input("data:mission:sizing:main_route:climb:thrust_rate", np.nan)
        self.add_input("data:mission:sizing:main_route:cruise:altitude", np.nan, units="m")
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:coef_k", np.nan)
        self.add_input("data:geometry:wing:area", np.nan, units="m**2")
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_out:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:holding:fuel", 0.0, units="kg")
        self.add_input("data:mission:sizing:takeoff:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:fuel", np.nan, units="kg")

        self.add_output("data:mission:sizing:main_route:climb:fuel", units="kg")
        self.add_output("data:mission:sizing:main_route:climb:distance", units="m")
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs):

        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:engine:count"]
        )
        thrust_rate = inputs["data:mission:sizing:main_route:climb:thrust_rate"]
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        cd0_clean = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        coef_k = inputs["data:aerodynamics:aircraft:cruise:coef_k"]
        wing_area = inputs["data:geometry:wing:area"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        m_to = inputs["data:mission:sizing:taxi_out:fuel"]
        m_ho = inputs["data:mission:sizing:holding:fuel"]
        m_tk = inputs["data:mission:sizing:takeoff:fuel"]
        m_ic = inputs["data:mission:sizing:initial_climb:fuel"]

        # Define initial conditions
        altitude_t = 50 / 0.3048 # conversion to m
        distance_t = 0.0
        mass_t = mtow - (m_to + m_ho + m_tk + m_ic)
        mass_fuel_t = 0.0
        atm_0 = Atmosphere(0.0)

        # FIXME: VCAS strategy is specific to ICE-propeller configuration, should be an input
        cl = math.sqrt(3*cd0_clean/coef_k)
        atm = Atmosphere(altitude_t, altitude_in_feet=False)
        v_cas = math.sqrt((mass_t * g) / (0.5 * atm.density * wing_area * cl))

        while altitude_t < cruise_altitude:

            # Define air properties
            atm = Atmosphere(altitude_t, altitude_in_feet=False)
            v_tas = v_cas * math.sqrt(atm_0.density / atm.density)

            # Evaluate thrust and sfc
            mach = math.sqrt(5 * ((atm_0.pressure / atm.pressure \
                    * ((1 + 0.2 * (v_cas / atm_0.speed_of_sound) ** 2) \
                    ** 3.5 - 1) + 1) ** (1 / 3.5) - 1))
            flight_point = {
                'mach': [mach],
                'altitude': [0],
                'thrust_is_regulated': [altitude_t],
                'thrust_rate': [thrust_rate],
                'thrust': None,
            }
            flight_point = pd.DataFrame(data=flight_point)
            propulsion_model.compute_flight_points(flight_point)
            Thrust = flight_point.Thrust
            sfc = flight_point.sfc

            # Calculates cl and drag considering constant climb rate
            cl = mass_t * g / (0.5 * atm.density * wing_area * v_tas**2)
            cd = cd0_clean + coef_k * cl**2

            # Calculate climb rate and height increase
            climb_rate = Thrust / (mass_t * g) - cd / cl
            vz = v_tas * math.sin(climb_rate)
            vx = v_tas * math.cos(climb_rate)
            altitude_t += vz * TIME_STEP
            distance_t += vx * TIME_STEP

            # Estimate mass evolution
            mass_fuel_t += (sfc * Thrust) * TIME_STEP
            mass_t = mass_t - (sfc * Thrust) * TIME_STEP

        outputs["data:mission:sizing:main_route:climb:fuel"] = mass_fuel_t
        outputs["data:mission:sizing:main_route:climb:distance"] = distance_t


class _compute_cruise(om.ExplicitComponent):
    """
    Compute the fuel consumption on cruise segment with constant VTAS and altitude.
    The hypothesis of small alpha/gamma angles is done.
    """

    def __init__(self, **kwargs):
        """
        Computes thrust, SFC and thrust rate by direct call to engine model.
        Options:
          - propulsion_id: (mandatory) the identifier of the propulsion wrapper.
          - out_file: if provided, a csv file will be written at provided path with all computed
                      flight points. If path is relative, it will be resolved from working
                      directory
        """
        super().__init__(**kwargs)
        self.flight_points = None
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:TLAR:v_cruise", np.nan, units="m/s")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:cruise:distance", np.nan, units="m")
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:coef_k", np.nan)
        self.add_input("data:geometry:wing:area", np.nan, units="m**2")
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_out:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:holding:fuel", 0.0, units="kg")
        self.add_input("data:mission:sizing:takeoff:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:climb:fuel", np.nan, units="kg")

        self.add_output("data:mission:sizing:main_route:cruise:fuel", units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:engine:count"]
        )
        v_tas = inputs["data:TLAR:v_cruise"]
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        cruise_distance = inputs["data:mission:sizing:main_route:cruise:distance"]
        cd0_clean = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        coef_k = inputs["data:aerodynamics:aircraft:cruise:coef_k"]
        wing_area = inputs["data:geometry:wing:area"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        m_to = inputs["data:mission:sizing:taxi_out:fuel"]
        m_ho = inputs["data:mission:sizing:holding:fuel"]
        m_tk = inputs["data:mission:sizing:takeoff:fuel"]
        m_ic = inputs["data:mission:sizing:initial_climb:fuel"]
        m_cl = inputs["data:mission:sizing:main_route:climb:fuel"]

        # Define initial conditions
        distance_t = 0.0
        mass_fuel_t = 0.0
        mass_t = mtow - (m_to + m_ho + m_tk + m_ic + m_cl)
        atm_0 = Atmosphere(0.0)
        atm = Atmosphere(cruise_altitude, altitude_in_feet=False)
        v_cas = v_tas / math.sqrt(atm_0.density / atm.density)

        while distance_t < cruise_distance:

            # Calculate Cl - Cd and corresponding drag
            cl = mass_t * g / (0.5 * atm.density * wing_area * v_tas ** 2)
            cd = cd0_clean + coef_k * cl ** 2
            Drag = 0.5 * atm.density * wing_area * cd * v_tas ** 2

            # Evaluate sfc
            mach = math.sqrt(5 * ((atm_0.pressure / atm.pressure \
                                   * ((1 + 0.2 * (v_cas / atm_0.speed_of_sound) ** 2) \
                                      ** 3.5 - 1) + 1) ** (1 / 3.5) - 1))
            flight_point = {
                'mach': [mach],
                'altitude': [cruise_altitude],
                'thrust_is_regulated': [1],
                'thrust_rate': None,
                'thrust': [Drag],
            }
            flight_point = pd.DataFrame(data=flight_point)
            propulsion_model.compute_flight_points(flight_point)
            Thrust = flight_point.Thrust
            sfc = flight_point.sfc

            # Calculate distance increase
            distance_t += v_tas * TIME_STEP

            # Estimate mass evolution
            mass_fuel_t += (sfc * Thrust) * TIME_STEP
            mass_t = mass_t - (sfc * Thrust) * TIME_STEP

        outputs["data:mission:sizing:main_route:cruise:fuel"] = mass_fuel_t


class _compute_descent(om.ExplicitComponent):
    """
    Compute the fuel consumption on descent segment with constant VCAS and descent
    speed Vz (<0m/s).
    The hypothesis of small alpha/gamma angles is done.
    Warning: Descent rate is maintained even if cd/cl < abs(desc_rate)!
    """

    def __init__(self, **kwargs):
        """
        Computes thrust, SFC and thrust rate by direct call to engine model.
        Options:
          - propulsion_id: (mandatory) the identifier of the propulsion wrapper.
          - out_file: if provided, a csv file will be written at provided path with all computed
                      flight points. If path is relative, it will be resolved from working
                      directory
        """
        super().__init__(**kwargs)
        self.flight_points = None
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:mission:sizing:main_route:cruise:altitude", np.nan, units="m")
        self.add_input("data:mission:sizing:main_route:descent:descent_rate", np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:L_D_max", np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:optimal_CL", np.nan)
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

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:engine:count"]
        )
        cruise_altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
        descent_rate = inputs["data:mission:sizing:main_route:descent:descent_rate"]
        cl_cd = inputs["data:aerodynamics:aircraft:cruise:L_D_max"]
        cl = inputs["data:aerodynamics:aircraft:cruise:optimal_CL"]
        wing_area = inputs["data:geometry:wing:area"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        m_to = inputs["data:mission:sizing:taxi_out:fuel"]
        m_ho = inputs["data:mission:sizing:holding:fuel"]
        m_tk = inputs["data:mission:sizing:takeoff:fuel"]
        m_ic = inputs["data:mission:sizing:initial_climb:fuel"]
        m_cl = inputs["data:mission:sizing:main_route:climb:fuel"]
        m_cr = inputs["data:mission:sizing:main_route:cruise:fuel"]

        # Define initial conditions
        altitude_t = cruise_altitude
        distance_t = 0.0
        mass_fuel_t = 0.0
        mass_t = mtow - (m_to + m_ho + m_tk + m_ic + m_cl + m_cr)
        atm_0 = Atmosphere(0.0)


        while altitude_t > 0.0:

            # Define air properties and calculate VTAS
            atm = Atmosphere(altitude_t, altitude_in_feet=False)
            v_cas = math.sqrt((mass_t * g) / (0.5 * atm.density * wing_area * cl))
            v_tas = v_cas * math.sqrt(atm_0.density / atm.density)

            # Is necessary thrust is 0 (desc_rate not reachable, then constant thrust
            # rate applied to match IDLE
            if (1/cl_cd) < abs(descent_rate):
                flight_point = {
                    'mach': [mach],
                    'altitude': [altitude_t],
                    'thrust_is_regulated': [0],
                    'thrust_rate': [propulsion_model.idle_thrust_rate],
                    'thrust': None,
                }
                descent_rate = (1/cl_cd)
            else:
                Thrust = (1/cl_cd + descent_rate) * (mass_t * g)
                flight_point = {
                    'mach': [mach],
                    'altitude': [altitude_t],
                    'thrust_is_regulated': [1],
                    'thrust_rate': None,
                    'thrust': [Thrust],
                }

            # Evaluate sfc
            mach = math.sqrt(5 * ((atm_0.pressure / atm.pressure \
                                   * ((1 + 0.2 * (v_cas / atm_0.speed_of_sound) ** 2) \
                                      ** 3.5 - 1) + 1) ** (1 / 3.5) - 1))

            flight_point = pd.DataFrame(data=flight_point)
            propulsion_model.compute_flight_points(flight_point)
            Thrust = flight_point.Thrust
            sfc = flight_point.sfc

            # Calculate distance increase
            v_x = v_tas * math.cos(descent_rate)
            v_z = v_tas * math.sin(descent_rate)
            distance_t += v_x * TIME_STEP
            altitude_t += v_z * TIME_STEP

            # Estimate mass evolution
            mass_fuel_t += (sfc * Thrust) * TIME_STEP
            mass_t = mass_t - (sfc * Thrust) * TIME_STEP

        outputs["data:mission:sizing:main_route:descent:fuel"] = mass_fuel_t
        outputs["data:mission:sizing:main_route:descent:distance"] = distance_t