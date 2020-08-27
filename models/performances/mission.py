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
import math
import openmdao.api as om
from fastoad import BundleLoader
from fastoad.constants import FlightPhase
from fastoad.utils.physics import Atmosphere
from scipy.constants import g

CLIMB_MASS_RATIO = 0.97  # = mass at end of climb / mass at start of climb
ALPHA_RATE = 3.0 # Angular rotation speed 
TIME_STEP = 0.05 # For time dependent simulation

class TakeOffPhase(om.Group):
    
    def setup(self):
        self.add_subsystem("compute_v2", _v2(), promotes=["*"])
        self.add_subsystem("compute_vloff", _vloff(), promotes=["*"])
        self.add_subsystem("compute_vr", _vr(), promotes=["*"])
        self.add_subsystem("simulate_takeoff", _simulate_takeoff(), promotes=["*"])
        self.connect("v2:v2", "vloff:v2")
        self.connect("v2:alpha_v2", "vloff:alpha_v2")
        self.connect("vloff:vloff", "vr:vloff")
        self.connect("vloff:alpha_vloff", "vr:alpha_vloff")
        self.connect("vr:vr", "takeoff:vr_in")
        self.connect("v2:alpha_v2", "takeoff:alpha_v2")

class TakeOffSpeed(om.Group):
    
    def setup(self):
        self.add_subsystem("compute_v2", _v2(), promotes=["*"])
        self.add_subsystem("compute_vloff", _vloff(), promotes=["*"])
        self.add_subsystem("compute_vr", _vr(), promotes=["*"])
        self.connect("v2:v2", "vloff:v2")
        self.connect("v2:alpha_v2", "vloff:alpha_v2")
        self.connect("vloff:vloff", "vr:vloff")
        self.connect("vloff:alpha_vloff", "vr:alpha_vloff")

class _compute_taxi_out(om.ExplicitComponent):
    """
    Compute the fuel consumption for taxi out based on speed and duration.
    """

    def setup(self):
        
        self.add_input("data:mission:sizing:taxi_out:duration", np.nan, units='s')
        self.add_input("data:mission:sizing:taxi_out:speed", np.nan, units='m/s')
        self.add_input("data:mission:sizing:takeoff:friction_coefficient_no_brake", np.nan)
        self.add_input("data:aerodynamics:aircraft:takeoff:CD0_clean", np.nan)
        self.add_input("data:aerodynamics:high_lift_devices:takeoff:CD", np.nan)
        self.add_input("data:weight:aircraft:MTOW", np.nan, units='kg')
        self.add_input("data:geometry:wing:area", np.nan, units='m**2')
        
        self.add_output("data:mission:sizing:taxi_out:thrust_rate")
        self.add_output("data:mission:operational:taxi_out:fuel", units='kg')
        
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        dt = inputs["data:mission:sizing:taxi_out:duration"]
        V = inputs["data:mission:sizing:taxi_out:speed"]
        friction_coeff = inputs["data:mission:sizing:takeoff:friction_coefficient_no_brake"]
        CL0_clean = inputs["data:aerodynamics:aircraft:takeoff:CL0_clean"]
        CL0_high_lift = inputs["data:aerodynamics:high_lift_devices:takeoff:CL"]
        CD0_clean = inputs["data:aerodynamics:aircraft:takeoff:CD0_clean"]
        CD0_high_lift = inputs["data:aerodynamics:high_lift_devices:takeoff:CD"]
        TOW = inputs["data:weight:aircraft:MTOW"]
        wing_area = inputs["data:geometry:wing:area"]
        
        
        atm = Atmosphere(0.0, 15.0)
        CL0 = CL0_clean+CL0_high_lift
        CD0 = CD0_clean+CD0_high_lift
        Lift = 0.5 * atm.density * wing_area * CL0 * V**2
        Drag = 0.5 * atm.density * wing_area * CD0 * V**2
        Friction = (TOW * g - Lift) * friction_coeff
        Thrust = Drag + Friction
        
        
        outputs["v2:climb_rate"] = CLIMB_RATE
        outputs["v2:v2"] = V2
        outputs["v2:alpha_v2"] = ALPHA * 180.0/math.pi # conversion to degre

class _vloff(om.ExplicitComponent):
    """
    Search for Vloff speed to reach V2 safety speed at 35ft height 
    with a alpha angle <= alpha_V2.
    """
    def setup(self):
        
        self.add_input("data:aerodynamics:aircraft:takeoff:CL0_clean", np.nan)
        self.add_input("data:aerodynamics:high_lift_devices:takeoff:CL", np.nan)
        self.add_input("data:aerodynamics:aircraft:takeoff:CL_alpha", np.nan)
        self.add_input("data:aerodynamics:aircraft:takeoff:CD0_clean", np.nan)
        self.add_input("data:aerodynamics:high_lift_devices:takeoff:CD", np.nan)
        self.add_input("data:aerodynamics:aircraft:takeoff:coef_k", np.nan)
        self.add_input("data:geometry:wing:area", np.nan, units="m**2")
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("propulsion_max_power", np.nan, units='w')
        self.add_input("propulsion_mach", np.nan, units='w')
        self.add_input("propulsion_max_power", np.nan, units='w')
        self.add_input("vloff:v2", np.nan, units='m/s')
        self.add_input("vloff:alpha_v2", np.nan, units='rad')
        
        self.add_output("vloff:vloff", units='m/s')
        self.add_output("vloff:alpha_vloff", units='deg')
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        CL0_clean = inputs["data:aerodynamics:aircraft:takeoff:CL0_clean"]
        CL0_high_lift = inputs["data:aerodynamics:high_lift_devices:takeoff:CL"]
        CL_alpha = inputs["data:aerodynamics:aircraft:takeoff:CL_alpha"]
        CD0_clean = inputs["data:aerodynamics:aircraft:takeoff:CD0_clean"]
        CD0_high_lift = inputs["data:aerodynamics:high_lift_devices:takeoff:CD"]
        coef_k = inputs["data:aerodynamics:aircraft:takeoff:coef_k"]
        wing_area = inputs["data:geometry:wing:area"]
        TOW = inputs["data:weight:aircraft:MTOW"]
        prop_max_power = inputs["propulsion_max_power"]
        prop_mach = inputs["propulsion_mach"]
        V2 = inputs["vloff:v2"]
        alpha_V2 = inputs["vloff:alpha_v2"]
        
        CL0 = CL0_clean+CL0_high_lift
        CD0 = CD0_clean+CD0_high_lift
        alpha = np.linspace(alpha_V2, 0.0, num=50)
        vloff = np.zeros(np.size(alpha))
        VLOFF = V2
        ALPHA = alpha_V2
        for i in range(len(alpha)):
            atm = Atmosphere(0, 15.0)
            cl = CL0 + CL_alpha * alpha[i]
            # Estimation thrust with previous alpha
            if i == 0:
                v_estimated = math.sqrt(2 * TOW * g / (cl * wing_area * atm.density))
                thrust_estimated = np.interp(v_estimated / atm.speed_of_sound, prop_mach, prop_max_power)/v_estimated
            else:
                thrust_estimated = np.interp(VLOFF[i-1] / atm.speed_of_sound, prop_mach, prop_max_power)/VLOFF[i-1]
            
            vloff[i] = math.sqrt(2 * (TOW * g - thrust_estimated * math.sin(alpha[i])) \
                       / (cl * wing_area * atm.density))
            # Start calculation of flight till reaching 35ft
            alpha_t = alpha[i]
            gamma_t = 0.0
            v_t = vloff[i]
            height_t = 0.0
            distance_t = 0.0
            mass_t = TOW
            while height_t < 35.0:
                # Estimation of thrust
                atm = Atmosphere(height_t, 15.0)
                thrust_real = np.interp(v_t / atm.speed_of_sound, prop_mach, prop_max_power)/v_t
                # Acceleration in air speed reference axis
                cl = CL0 + CL_alpha * alpha_t
                cd = CD0 + coef_k * cl**2
                Lift = 0.5 * atm.density * wing_area * cl * v_t**2
                Drag = 0.5 * atm.density * wing_area * cd * v_t**2
                acc_x = (thrust_real * math.cos(alpha_t) - mass_t * g * math.sin(gamma_t) - Drag) / mass_t
                acc_z = (Lift + thrust_real * math.sin(alpha_t) - mass_t * g * math.cos(gamma_t)) / mass_t
                delta_gamma = math.atan((acc_x*TIME_STEP) / (v_t+acc_z*TIME_STEP))
                # Speed calculation and integration of distance/height
                v_t_new = math.sqrt((acc_x*TIME_STEP)**2 + (v_t+acc_z*TIME_STEP)**2)
                v_x_old = v_t * math.cos(gamma_t)
                v_z_old = v_t * math.sin(gamma_t)
                v_x_new = v_t_new * math.cos(gamma_t + delta_gamma)
                v_z_new = v_t_new * math.sin(gamma_t + delta_gamma)
                delta_height = (v_z_old+v_z_new)/2 * TIME_STEP
                delta_distance = (v_x_old+v_x_new)/2 * TIME_STEP
                # Update temporal values
                alpha_t = min(alpha_V2, alpha_t + ALPHA_RATE*math.pi/180.0*TIME_STEP)
                gamma_t = gamma_t + delta_gamma
                height_t = height_t + delta_height * 0.3048 # conversion to ft
                distance_t = distance_t + delta_distance 
                v_t = v_t_new
            # Save lowest speed leading to speed >=V2 with alpha_V2 @ 35ft
            if (v_t >= V2) and (alpha_t==alpha_V2) and vloff[i]<VLOFF:
                VLOFF = vloff[i]
                ALPHA = alpha[i]
            else:
                break
        
        outputs["vloff:vloff"] = VLOFF
        outputs["vloff:alpha_vloff"] = ALPHA * 180.0/math.pi # conversion to degre

class _Propulsion(om.ExplicitComponent): # FIXME: understand how engine model is choosen
    
    def __init__(self, **kwargs):
        """
        Computes thrust, SFC and thrust rate by direct call to engine model.
        """
        super().__init__(**kwargs) 
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper.setup(self)
        self.add_input("data:mission:sizing:cruise:altitude", np.nan, units="m")
        self.add_input("data:TLAR:cruise_mach", np.nan)
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:aerodynamics:aircraft:cruise:L_D_max", np.nan)
        self.add_input("data:geometry:propulsion:engine:count", 2)

        self.add_output("data:propulsion:SFC", units="kg/s/N", ref=1e-4)
        self.add_output("data:propulsion:thrust_rate", lower=0.0, upper=1.0)
        self.add_output("data:propulsion:thrust", units="N", ref=1e5)

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        engine_count = inputs["data:geometry:propulsion:engine:count"]
        ld_ratio = inputs["data:aerodynamics:aircraft:cruise:L_D_max"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        initial_cruise_mass = mtow 

        thrust = initial_cruise_mass / ld_ratio * g / engine_count
        sfc, thrust_rate, _ = self._engine_wrapper.get_engine(inputs).compute_flight_points(
            inputs["data:TLAR:cruise_mach"],
            inputs["data:mission:sizing:cruise:altitude"],
            FlightPhase.CRUISE,
            thrust=thrust,
        )
        outputs["data:propulsion:thrust"] = thrust
        outputs["data:propulsion:SFC"] = sfc
        outputs["data:propulsion:thrust_rate"] = thrust_rate
        
class _MissionEngine(om.ExplicitComponent):
    def __init__(self, **kwargs):
        """
        Computes thrust, SFC and thrust rate by direct call to engine model.
        """
        super().__init__(**kwargs)
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper.setup(self)
        self.add_input("data:mission:sizing:cruise:altitude", np.nan, units="m")
        self.add_input("data:TLAR:cruise_mach", np.nan)
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("data:aerodynamics:aircraft:cruise:L_D_max", np.nan)
        self.add_input("data:geometry:propulsion:engine:count", 2)

        self.add_output("data:propulsion:SFC", units="kg/s/N", ref=1e-4)
        self.add_output("data:propulsion:thrust_rate", lower=0.0, upper=1.0)
        self.add_output("data:propulsion:thrust", units="N", ref=1e5)

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        engine_count = inputs["data:geometry:propulsion:engine:count"]
        ld_ratio = inputs["data:aerodynamics:aircraft:cruise:L_D_max"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        initial_cruise_mass = mtow * CLIMB_MASS_RATIO

        thrust = initial_cruise_mass / ld_ratio * g / engine_count
        sfc, thrust_rate, _ = self._engine_wrapper.get_engine(inputs).compute_flight_points(
            inputs["data:TLAR:cruise_mach"],
            inputs["data:mission:sizing:cruise:altitude"],
            FlightPhase.CRUISE,
            thrust=thrust,
        )
        outputs["data:propulsion:thrust"] = thrust
        outputs["data:propulsion:SFC"] = sfc
        outputs["data:propulsion:thrust_rate"] = thrust_rate