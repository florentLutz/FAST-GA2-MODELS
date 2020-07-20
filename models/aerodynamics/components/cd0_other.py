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

import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent


class Cd0Other(ExplicitComponent):

    def setup(self):

        self.add_input("data:configuration:LG_type", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        
        self.add_output("cd0_other")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        
        engine_number = inputs["data:geometry:propulsion:engine:count"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        wing_area = inputs["data:geometry:wing:area"]
        
        #COWLING (only if engine in fuselage): cx_cowl*wing_area assumed typical (Gudmunsson p739)
        engine_in_fus = engine_number % 2
        cd0_cowling = 0.0267/wing_area * engine_in_fus
        #Cooling (piston engine only)
        #Gudmunsson p715. Assuming cx_cooling*wing area/MTOW value of the book is typical
        cd0_cooling = 7.054E-6 / wing_area * mtow # FIXME: no type piston engine defined...
        #Gudmunnson p739. Sum of other components (not calculated here), cx_other*wing_area assumed typical
        cd0_components = 0.0253/(wing_area)        

        outputs["cd0_other"] = cd0_cowling + cd0_cooling + cd0_components
