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


class Cd0Wing(ExplicitComponent):
    def initialize(self):
        self.options.declare("reynolds", default=False, types=float)
        self.options.declare("mach", default=False, types=float)

    def setup(self):
        
        self.reynolds = self.options["reynolds"]
        self.mach = self.options["mach"]
        
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:wetted_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        self.add_output("cd0_wing", val=np.nan)

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
        
        #Local Reynolds: re*length
        re = self.reynolds/l0_wing
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
        if self.mach>0.2:
            ff = ff * 1.34 * self.mach**0.18 * (math.cos(sweep_25*math.pi/180))**0.28
            
        cd0 = ff*cf_wing * wet_area_wing / wing_area        

        outputs["cd0_wing"] = cd0
