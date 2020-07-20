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


class Cd0Fuselage(ExplicitComponent):
    def initialize(self):
        self.options.declare("reynolds", default=False, types=float)

    def setup(self):
        
        self.reynolds = self.options["reynolds"]

        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:wetted_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        
        self.add_output("cd0_fus")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        height = inputs["data:geometry:fuselage:maximum_height"]
        width = inputs["data:geometry:fuselage:maximum_width"]
        length = inputs["data:geometry:fuselage:length"]
        wet_area_fus = inputs["data:geometry:fuselage:wetted_area"]
        wing_area = inputs["data:geometry:wing:area"]
        
        #Local Reynolds: re*length
        re = self.reynolds/l0_wing
        #5% NLF
        x_trans = 0.05
        #Roots
        x0_turb = 36.9 * x_trans**0.625 * (1/(re*length))**0.375
        cf_fus = 0.074 / (re*length)**0.2 * (1 - (x_trans - x0_turb))**0.8        
        f = length/math.sqrt(4*height*width/math.pi) 
        ff_fus = 1 + 60/(f**3) + f/400
        
        #Fuselage
        cd0_fuselage = cf_fus * ff_fus * wet_area_fus / wing_area
        #Cockpit window (Gudmunsson p727)
        cd0_window = 0.002 * (height*width)/wing_area

        outputs["cd0_fus"] = cd0_fuselage + cd0_window
