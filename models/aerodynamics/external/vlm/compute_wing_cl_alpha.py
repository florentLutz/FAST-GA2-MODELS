"""
    Estimation of lift coefficient using VLM
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

import logging

import numpy as np
import math
from fastoad.models.aerodynamics.constants import POLAR_POINT_COUNT
from fastoad.utils.physics import Atmosphere
from vlm import VLM

_INPUT_AOAList = [2.0, 7.0] # ???: why such angles choosen ?
DEFAULT_ALPHA = 0.0

_LOGGER = logging.getLogger(__name__)


class ComputeWingCLALPHAvlm(VLM):
    """ Computes lift coefficient """
    
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)
        
    def setup(self):
        
        super().setup()
        
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units='m')
        self.add_input("data:geometry:wing:span", val=np.nan, units='m')
        self.add_input("vlm:alpha", val=np.nan)
        self.add_input("vlm:CL_clean", val=np.nan)
        nans_array = np.full(POLAR_POINT_COUNT, np.nan)
        if self.options["low_speed_aero"]:
            self.add_input("Mach_low_speed", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:alpha", val=nans_array)
            self.add_input("data:aerodynamics:wing:low_speed:CL", val=nans_array)
            self.add_output("data:aerodynamics:aircraft:low_speed:cl_0_clean")
            self.add_output("data:aerodynamics:aircraft:low_speed:cl_alpha")
        else:
            self.add_input("data:TLAR:v_cruise", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:alpha", val=nans_array)
            self.add_input("data:aerodynamics:wing:cruise:CL", val=nans_array)
            self.add_input("data:mission:sizing:cruise:altitude", val=np.nan, units='ft')
            self.add_output("data:aerodynamics:aircraft:cruise:cl_0_clean")
            self.add_output("data:aerodynamics:aircraft:cruise:cl_alpha")
        
        self.declare_partials("*", "*", method="fd")        
    
    def compute(self, inputs, outputs):

        # Get inputs
        b_f = inputs['data:geometry:fuselage:maximum_width']
        span = inputs['data:geometry:wing:span']
        if self.options["low_speed_aero"]:
            altitude = 0.0
            atm = Atmosphere(altitude)
            mach = inputs["Mach_low_speed"]
            CL_clean = inputs["data:aerodynamics:wing:low_speed:CL"]
            alpha = inputs["data:aerodynamics:wing:low_speed:alpha"]
        else:
            altitude = inputs["data:mission:sizing:cruise:altitude"]
            atm = Atmosphere(altitude)
            mach = inputs["data:TLAR:v_cruise"]/atm.speed_of_sound
            CL_clean = inputs["data:aerodynamics:wing:cruise:CL"]
            alpha = inputs["data:aerodynamics:wing:cruise:alpha"]
        
        atm = Atmosphere(altitude)
        V_inf = min(atm.speed_of_sound * mach, 0.1) # avoid V=0 m/s crashes
        super()._run()
        Cl, Cdi, Oswald, Cm = super().compute_wing(self, inputs, _INPUT_AOAList, V_inf, flaps_angle=0.0, use_airfoil=True)
        k_fus = 1 + 0.025*b_f/span - 0.025*(b_f/span)**2 # Fuselage correction
        beta = math.sqrt(1 - mach**2) # Prandtl-Glauert
        cl_alpha = (Cl[1] - Cl[0]) / ((_INPUT_AOAList[1]-_INPUT_AOAList[0])*math.pi/180) * k_fus / beta
        alpha_0 = alpha[0]**math.pi/180 - CL_clean[0] / cl_alpha
        cl_0 = -alpha_0 * cl_alpha

        if self.options["low_speed_aero"]:
            outputs['data:aerodynamics:aircraft:low_speed:cl_0_clean'] = cl_0
            outputs['data:aerodynamics:aircraft:low_speed:cl_alpha'] = cl_alpha
        else:
            outputs['data:aerodynamics:aircraft:cruise:cl_0_clean'] = cl_0
            outputs['data:aerodynamics:aircraft:cruise:cl_alpha'] = cl_alpha