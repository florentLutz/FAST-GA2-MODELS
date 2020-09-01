"""Simple module for takeoff."""
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

ALPHA_LIMIT = 13.5 # Limit angle to touch tail on ground
ALPHA_RATE = 3.0 # Angular rotation speed 
SAFETY_HEIGHT = 50 # Height in feets to reach V2 speed
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

class _v2(om.ExplicitComponent):
    """
    Search for V2 safety speed and alpha angle to reach minimum climb rate with 
    take-off-weight and maximum thrust.
    """

    def setup(self):
        
        self.add_input("data:aerodynamics:aircraft:low_speed:CL0_clean", np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CL", np.nan)
        self.add_input("data:aerodynamics:aircraft:takeoff:CL_max", np.nan)
        self.add_input("data:aerodynamics:aircraft:low_speed:CL_alpha", np.nan)
        self.add_input("data:aerodynamics:aircraft:low_speed:CD0", np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CD", np.nan)
        self.add_input("data:aerodynamics:aircraft:low_speed:coef_k", np.nan)
        self.add_input("data:geometry:wing:area", np.nan, units="m**2")
        self.add_input("data:mission:sizing:climb:min_climb_rate", np.nan)
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("propulsion_max_power", np.nan, units='w') # FIXME: array
        self.add_input("propulsion_mach", np.nan) # FIXME: array
        
        self.add_output("v2:climb_rate")
        self.add_output("v2:v2", units='m/s')
        self.add_output("v2:alpha_v2", units='deg')
        
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        CL0_clean = inputs["data:aerodynamics:aircraft:low_speed:CL0_clean"]
        CL0_high_lift = inputs["data:aerodynamics:flaps:takeoff:CL"]
        CL_max_clean = inputs["data:aerodynamics:aircraft:takeoff:CL_max"]
        CL_alpha = inputs["data:aerodynamics:aircraft:low_speed:CL_alpha"]
        CD0_clean = inputs["data:aerodynamics:aircraft:low_speed:CD0"]
        CD0_high_lift = inputs["data:aerodynamics:flaps:takeoff:CD"]
        coef_k = inputs["data:aerodynamics:aircraft:low_speed:coef_k"]
        wing_area = inputs["data:geometry:wing:area"]
        min_climb_rate = inputs["data:mission:sizing:climb:min_climb_rate"]
        TOW = inputs["data:weight:aircraft:MTOW"]
        prop_max_power = inputs["propulsion_max_power"]
        prop_mach = inputs["propulsion_mach"]
        
        
        atm = Atmosphere(SAFETY_HEIGHT, 15.0)
        alpha_max = (CL_max_clean-CL0_clean)/CL_alpha
        gama_climb_rate = math.asin(min_climb_rate)
        CD0 = CD0_clean+CD0_high_lift
        CL0 = CL0_clean+CL0_high_lift
        alpha = np.linspace(0.0, min(alpha_max, ALPHA_LIMIT), num=100) * math.pi/180.0
        V = np.zeros(np.size(alpha))
        climb_rate = np.zeros(np.size(alpha))
        V2 = math.inf
        for i in range(len(alpha)):
            cl = CL0 + CL_alpha * alpha[i]
            cd = CD0 + coef_k * cl**2
            # Estimation thrust with previous alpha
            if i == 0:
                v_estimated = math.sqrt(2 * TOW * g * math.cos(gama_climb_rate)/ (cl * wing_area * atm.density))
                thrust_estimated = np.interp(v_estimated / atm.speed_of_sound, prop_mach, prop_max_power)/v_estimated
            else:
                thrust_estimated = np.interp(V[i-1] / atm.speed_of_sound, prop_mach, prop_max_power)/V[i-1]
            # Calculation of speed to compensate weight with lift-off (take into account thrust)
            V[i] = math.sqrt(2 * (TOW * g * math.cos(gama_climb_rate) - thrust_estimated * math.sin(alpha[i]))/ (cl * wing_area * atm.density))
            thrust_real = np.interp(V[i] / atm.speed_of_sound, prop_mach, prop_max_power)/V[i]
            drag = 0.5 * atm.density * wing_area * cd * V[i]**2
            # Estimation of climb rate
            climb_rate[i] = (thrust_real * math.cos(alpha[i]) - drag) / (TOW * g)
            # Save lowest speed with sufficient climb_rate
            if (climb_rate[i]>=min_climb_rate) and V[i]<V2:
                CLIMB_RATE = climb_rate[i]
                V2 = V[i]
                ALPHA = alpha[i]
        if max(climb_rate) < min_climb_rate:
            idx_max = np.where(climb_rate == max(climb_rate))
            CLIMB_RATE = max(climb_rate)
            V2 = V[idx_max]
            ALPHA = alpha[idx_max]
        
        outputs["v2:climb_rate"] = CLIMB_RATE
        outputs["v2:v2"] = V2
        outputs["v2:alpha_v2"] = ALPHA * 180.0/math.pi # conversion to degre

class _vloff(om.ExplicitComponent):
    """
    Search for Vloff speed to reach V2 safety speed at 35ft height 
    with an alpha angle <= alpha_V2 and maximum thrust.
    """
    def setup(self):
        
        self.add_input("data:aerodynamics:aircraft:low_speed:CL0_clean", np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CL", np.nan)
        self.add_input("data:aerodynamics:aircraft:low_speed:CL_alpha", np.nan)
        self.add_input("data:aerodynamics:aircraft:low_speed:CD0", np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CD", np.nan)
        self.add_input("data:aerodynamics:aircraft:low_speed:coef_k", np.nan)
        self.add_input("data:geometry:wing:area", np.nan, units="m**2")
        self.add_input("data:weight:aircraft:MTOW", np.nan, units="kg")
        self.add_input("propulsion_max_power", np.nan, units='w')
        self.add_input("propulsion_mach", np.nan)
        self.add_input("vloff:v2", np.nan, units='m/s')
        self.add_input("vloff:alpha_v2", np.nan, units='rad')
        
        self.add_output("vloff:vloff", units='m/s')
        self.add_output("vloff:alpha_vloff", units='deg')
        
        self.declare_partials("*", "*", method="fd")
        
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        CL0_clean = inputs["data:aerodynamics:aircraft:low_speed:CL0_clean"]
        CL0_high_lift = inputs["data:aerodynamics:flaps:takeoff:CL"]
        CL_alpha = inputs["data:aerodynamics:aircraft:low_speed:CL_alpha"]
        CD0_clean = inputs["data:aerodynamics:aircraft:low_speed:CD0"]
        CD0_high_lift = inputs["data:aerodynamics:flaps:takeoff:CD"]
        coef_k = inputs["data:aerodynamics:aircraft:low_speed:coef_k"]
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
            while height_t < SAFETY_HEIGHT:
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

class _vr(om.ExplicitComponent):
    """
    Search VR for given lift-off conditions error may be induced by estimated acceleration
    at t and not t-dt and approximation of speed V(t-dt)~V(t)-ACC(t)*dt.
    Time step has been reduced by 1/2 to limit integration error.
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
        self.add_input("data:mission:sizing:takeoff:friction_coefficient_no_brake", np.nan)
        self.add_input("propulsion_max_power", np.nan, units='w')
        self.add_input("propulsion_mach", np.nan, units='w')
        self.add_input("propulsion_max_power", np.nan, units='w')
        self.add_input("vr:vloff", np.nan, units='m/s')
        self.add_input("vr:alpha_vloff", np.nan, units='rad')
        
        self.add_output("vr:vr", units='m/s')
        
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
        friction_coeff = inputs["data:mission:sizing:takeoff:friction_coefficient_no_brake"]
        prop_max_power = inputs["propulsion_max_power"]
        prop_mach = inputs["propulsion_mach"]
        VLOFF = inputs["vr:vloff"]
        alpha_vloff = inputs["vloff:alpha_vloff"]
        
        # Start reverted calculation of flight from lift-off to 0° alpha angle
        CL0 = CL0_clean+CL0_high_lift
        CD0 = CD0_clean+CD0_high_lift
        alpha_t = alpha_vloff
        v_t = VLOFF
        atm = Atmosphere(0, 15.0)
        while alpha_t != 0.0:
            # Estimation of thrust
            thrust_real = np.interp(v_t / atm.speed_of_sound, prop_mach, prop_max_power)/v_t
            # Acceleration in air speed reference axis
            cl = CL0 + CL_alpha * alpha_t
            cd = CD0 + coef_k * cl**2
            Lift = 0.5 * atm.density * wing_area * cl * v_t**2
            Drag = 0.5 * atm.density * wing_area * cd * v_t**2
            Friction = (TOW * g - Lift) * friction_coeff
            acc_x = (thrust_real * math.cos(alpha_t) - Drag - Friction) / TOW
            # Speed and angle update
            v_t = v_t - acc_x * (TIME_STEP/2)
            alpha_t = max(0.0, alpha_t - ALPHA_RATE*math.pi/180.0*(TIME_STEP/2))
        
        outputs["vr:vr"] = v_t
        
class _simulate_takeoff(om.ExplicitComponent):
    """
    Simulate take-off from 0m/s speed to 35ft hight using input VR and alpha_v2
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
        self.add_input("data:mission:sizing:takeoff:friction_coefficient_no_brake", np.nan)
        self.add_input("propulsion_max_power", np.nan, units='w')
        self.add_input("propulsion_mach", np.nan, units='w')
        self.add_input("propulsion_max_power", np.nan, units='w')
        self.add_input("propulsion_consumption", np.nan, units='kg/s')
        self.add_input("takeoff:vr_in", np.nan, units='m/s')
        self.add_input("takeoff:alpha_v2", np.nan, units='rad')
        
        self.add_output("takeoff:vr", units='m/s')
        
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
        friction_coeff = inputs["data:mission:sizing:takeoff:friction_coefficient_no_brake"]
        prop_max_power = inputs["propulsion_max_power"]
        prop_mach = inputs["propulsion_mach"]
        fuel_consumption = inputs["propulsion_consumption"]
        VR = inputs["takeoff:vr_in"]
        alpha_v2 = inputs["takeoff:alpha_v2"]
        
        # Start calculation of flight from null speed to 35ft high
        CL0 = CL0_clean+CL0_high_lift
        CD0 = CD0_clean+CD0_high_lift
        alpha_t = 0.0
        gamma_t = 0.0
        v_t = 0.0
        height_t = 0.0
        distance_t = 0.0
        mass_t = TOW
        vloff = math.inf
        time_t = 0.0
        while alpha_t != 0.0:
            # Estimation of thrust
            atm = Atmosphere(height_t, 15.0)
            thrust_real = np.interp(v_t / atm.speed_of_sound, prop_mach, prop_max_power)/v_t
            # Acceleration in air speed reference axis
            cl = CL0 + CL_alpha * alpha_t
            cd = CD0 + coef_k * cl**2
            Lift = 0.5 * atm.density * wing_area * cl * v_t**2
            Drag = 0.5 * atm.density * wing_area * cd * v_t**2
            acc_z = (Lift + thrust_real * math.sin(alpha_t) - mass_t * g * math.cos(gamma_t)) / mass_t
            if acc_z>=0:
                friction_coeff = 0.0 # friction set to 0 when lift-off
                min(vloff, v_t)
            else:
                acc_z = 0.0
            Friction = (TOW * g - Lift) * friction_coeff
            acc_x = (thrust_real * math.cos(alpha_t) - mass_t * g * math.sin(gamma_t) - Drag - Friction) / mass_t
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
            if v_t>=VR:
                alpha_t = min(alpha_v2, alpha_t + ALPHA_RATE*math.pi/180.0*TIME_STEP)
            gamma_t = gamma_t + delta_gamma
            height_t = height_t + delta_height * 0.3048 # conversion to ft
            distance_t = distance_t + delta_distance
            mass_t = mass_t - np.interp(v_t / atm.speed_of_sound, prop_mach, fuel_consumption)*TIME_STEP
            v_t = v_t_new
            time_t = time_t + TIME_STEP
        
        outputs["takeoff:vloff"] = vloff
        outputs["takeoff:v2"] = v_t
        outputs["takeoff:climb_rate"] = math.sin(gamma_t)
        outputs["takeoff:distance"] = distance_t
        outputs["takeoff:consumption"] = TOW - mass_t
        outputs["takeoff:duration"] = time_t

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