"""
    Estimation of horizontal tail distance
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


class ComputeHTDistance(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Horizontal tail distance estimation """

    def setup(self):

        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:layout", val=np.nan)
        self.add_input("data:geometry:has_T_tail", val=np.nan)

        self.add_output("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", units="m")
        self.add_output("data:geometry:horizontal_tail:z:from_wingMAC25", units="m")

        self.declare_partials(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
            ["data:geometry:fuselage:length", "data:geometry:wing:MAC:length"],
            method="fd",
        )
        
        self.declare_partials("data:geometry:horizontal_tail:z:from_wingMAC25", "data:geometry:horizontal_tail:span", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
    
        fus_length = inputs["data:geometry:fuselage:length"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        span = inputs["data:geometry:vertical_tail:span"]
        engine_loc = inputs["data:geometry:propulsion:layout"]
        tail_type = inputs["data:geometry:has_T_tail"]

        if engine_loc == 1.0:
            lp_ht = 0.5 * fus_length
        elif engine_loc == 3.0:
            lp_ht = 3.0 * l0_wing
        else: # FIXME: no equation for configuration 2.0
            raise ValueError("Value of data:geometry:propulsion:layout can only be 0 or 3")
        
        if tail_type == 0.0:
            height_ht = 0
        else:
            height_ht = 0 + span

        outputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"] = lp_ht
        outputs["data:geometry:horizontal_tail:z:from_wingMAC25"] = height_ht
