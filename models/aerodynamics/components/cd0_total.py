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


class Cd0Total(ExplicitComponent):
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        
        self.add_input("cd0_wing", val=np.nan)
        self.add_input("cd0_fus", val=np.nan)
        self.add_input("cd0_ht", val=np.nan)
        self.add_input("cd0_vt", val=np.nan)
        self.add_input("cd0_nac", val=np.nan)
        self.add_input("cd0_lg", val=np.nan)
        self.add_input("cd0_other", val=np.nan)
        
        self.add_output("cd0_total")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):

        cd0_wing = inputs["cd0_wing"]
        cd0_fus = inputs["cd0_fus"]
        cd0_ht = inputs["cd0_ht"]
        cd0_vt = inputs["cd0_vt"]
        cd0_nac = inputs["cd0_nac"]
        cd0_lg = inputs["cd0_lg"]
        cd0_other = inputs["cd0_other"]

        #CRUD (other undesirable drag). Factor from Gudmunsson book
        crud_factor = 1.25

        cd0 = crud_factor * (cd0_wing + cd0_fus + cd0_ht + cd0_vt + cd0_lg + \
               cd0_nac + cd0_other)

        outputs["cd0_total"] = cd0
