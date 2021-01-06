"""
    Estimation of wing center of gravity
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
import math
from openmdao.core.explicitcomponent import ExplicitComponent


class ComputeWingCG(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Wing center of gravity estimation """

    def setup(self):

        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:virtual_chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")

        self.add_output("data:weight:airframe:wing:CG:x", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        x0_wing = inputs["data:geometry:wing:MAC:leading_edge:x:local"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        span = inputs["data:geometry:wing:span"]
        l1_wing = inputs["data:geometry:wing:root:virtual_chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]

        if sweep_25 < 5.0:
            x_cg_wing_rel = 0.4 * l0_wing + span * math.sin(sweep_25 * math.pi / 180.0)
        else:
            chord = np.interp(0.35, [0.0, 1.0], [l1_wing, l4_wing])
            x_cg_wing_rel = (0.35 * span / 2.0) * math.sin(sweep_25 * math.pi / 180.0) + 0.6 * chord

        x_cg_a1 = fa_length - 0.25 * l0_wing - x0_wing + x_cg_wing_rel
        
        outputs["data:weight:airframe:wing:CG:x"] = x_cg_a1
