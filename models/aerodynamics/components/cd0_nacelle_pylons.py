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


class Cd0NacelleAndPylons(ExplicitComponent):
    def initialize(self):
        self.options.declare("reynolds", default=False, types=float)
        self.options.declare("mach", default=False, types=float)

    def setup(self):
        
        self.reynolds = self.options["reynolds"]
        self.mach = self.options["mach"]

        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:height", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:width", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:length", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:wetted_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        
        self.add_output("cd0_nac")

        self.declare_partials(
                "cd0_nacelle_pylon",
                [
                        "data:geometry:wing:MAC:length",
                        "data:geometry:propulsion:nacelle:height",
                        "data:geometry:propulsion:nacelle:width",
                        "data:geometry:propulsion:nacelle:length",
                        "data:geometry:propulsion:nacelle:wetted_area",
                        "data:geometry:wing:area",
                ],
                method="fd",
        )

    def compute(self, inputs, outputs):
        
        engine_number = inputs["data:geometry:propulsion:engine:count"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        nac_height = inputs["data:geometry:propulsion:nacelle:height"]
        nac_width = inputs["data:geometry:propulsion:nacelle:width"]
        nac_length = inputs["data:geometry:propulsion:nacelle:length"]
        wet_area_nac = inputs["data:geometry:propulsion:nacelle:wetted_area"]
        wing_area = inputs["data:geometry:wing:area"]
        
        #Local Reynolds: re*length
        re = self.reynolds/l0_wing
        #Roskam method for wing-nacelle interaction factor (vol 6 page 3.62)
        cf_nac = 0.455 / ((1 + 0.144 * self.mach ** 2)**0.65 * (math.log10(re * nac_length)) ** (2.58)) #100% turbulent
        f = nac_length/math.sqrt(4*nac_height*nac_width/math.pi) 
        ff_nac = 1 + 0.35/f #Raymer (seen in Gudmunsson)
        if_nac = 0.036*nac_width*l0_wing/wing_area* 0.04
        engine_in_fus = engine_number % 2.0
        
        cd0 = (cf_nac * ff_nac * wet_area_nac / (wing_area) + if_nac) \
            * (engine_number - engine_in_fus)

        outputs["cd0_nac"] = cd0
