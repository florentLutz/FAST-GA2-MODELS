"""
Estimation of wing weight
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

import numpy as np
import openmdao.api as om
import math


class WingWeight(om.ExplicitComponent):
    """
    Wing weight estimation

    # TODO: Based on :????????????
    """

    def setup(self):
        
        self.add_input("data:mission:sizing:cs23:sizing_factor_ultimate", val=np.nan) # TODO: confirm it's a ratio!, to be added to xml variables
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan) # replace eps1
        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan) # replace el_aero
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:TLAR:limit_speed", val=np.nan, units="kn") # TODO: replace vne, to be added to xml variables
        
        self.add_output("data:weight:airframe:wing:mass", units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor_ultimate"]
        wing_area = inputs["data:geometry:wing:area"]*3.28084**2
        taper_ratio = inputs["data:geometry:wing:taper_ratio"]
        thickness_ratio = inputs["data:geometry:wing:root:thickness_ratio"]
        mtow = inputs["data:weight:aircraft:MTOW"]*2.20462 # Takeoff weight in lb
        aspect_ratio = inputs["data:geometry:wing:aspect_ratio"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        limit_speed = inputs["data:TLAR:limit_speed"]
        a1 = (
            96.948*((mtow*sizing_factor_ultimate/10**5)**0.65
            *(aspect_ratio/math.cos(sweep_25* math.pi/180))**0.57
            *(wing_area/100)**0.61*((1+taper_ratio)/(2*thickness_ratio))**0.36
            *(1+limit_speed/500)**0.5)**0.993
        ) # mass in lb
            
        outputs["data:weight:airframe:wing:mass"] = a1 /2.20462 # converted to kg
