"""
    FAST - Copyright (c) 2016 ONERA ISAE
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

import math
import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent
from fastoad.utils.physics import Atmosphere

class Cd0Wing(ExplicitComponent):
    
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        self.low_speed_aero = self.options["low_speed_aero"]

        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:wetted_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        if self.low_speed_aero:
            self.add_input("reynolds_low_speed", val=np.nan)
            self.add_input("Mach_low_speed", val=np.nan)
            self.add_output("data:aerodynamics:wing:low_speed:CD0")
        else:
            self.add_input("data:aerodynamics:wing:cruise:reynolds", val=np.nan)
            self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
            self.add_input("data:mission:sizing:cruise:altitude", val=np.nan, units="ft")
            self.add_output("data:aerodynamics:wing:cruise:CD0")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        y1_wing = inputs["data:geometry:fuselage:maximum_width"]/2.0
        y2_wing = inputs["data:geometry:wing:root:y"]
        span = inputs["data:geometry:wing:span"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        wet_area_wing = inputs["data:geometry:wing:wetted_area"]
        wing_area = inputs["data:geometry:wing:area"]
        if self.low_speed_aero:
            mach = inputs["Mach_low_speed"]
            reynolds = inputs["reynolds_low_speed"]
        else:
            altitude = inputs["data:mission:sizing:cruise:altitude"]
            atm = Atmosphere(altitude)
            mach = inputs["data:TLAR:v_cruise"]/atm.speed_of_sound
            reynolds = inputs["data:aerodynamics:wing:cruise:reynolds"]
        
        #Local Reynolds: re*length
        re = reynolds/l0_wing
        #Airfoil data: change if airfoil is redefined (NACA 23012)
        x_tmax = 0.3 # !!!: should be defined with parameter and data file
        thickness = 0.12 # !!!: should be defined with parameter and data file (data:geometry:wing:thickness_ratio?)
        #Root: 45% NLF
        x_trans = 0.45
        x0_turb = 36.9 * x_trans**0.625 * (1/(re*l2_wing))**0.375
        cf_root = 0.074 / (re*l2_wing)**0.2 * (1 - (x_trans - x0_turb))**0.8
        #Tip: 55% NLF
        x_trans = 0.55
        x0_turb = 36.9 * x_trans**0.625 * (1/(re*l4_wing))**0.375
        cf_tip = 0.074 / (re*l4_wing)**0.2 * (1 - (x_trans - x0_turb))**0.8
        
        cf_wing = (cf_root * (y2_wing-y1_wing) + 0.5*(span/2.0-y2_wing) * (cf_root+cf_tip)) / (span/2.0-y1_wing)
        ff = 1 + 0.6/x_tmax * thickness + 100 * thickness**4
        if mach>0.2:
            ff = ff * 1.34 * mach**0.18 * (math.cos(sweep_25*math.pi/180))**0.28
            
        cd0_wing = ff*cf_wing * wet_area_wing / wing_area        

        if self.low_speed_aero:
            outputs["data:aerodynamics:wing:low_speed:CD0"] = cd0_wing
        else:
            outputs["data:aerodynamics:wing:cruise:CD0"] = cd0_wing
