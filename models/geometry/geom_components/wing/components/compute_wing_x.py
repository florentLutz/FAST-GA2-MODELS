"""
    Estimation of wing Xs
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


class ComputeWingX(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Wing Xs estimation """

    def setup(self):
        
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")

        self.add_output("data:geometry:wing:tip:leading_edge:x:local", units="m")
        
        self.declare_partials(
            "data:geometry:wing:tip:leading_edge:x:local",
            [
                "data:geometry:wing:root:chord",
                "data:geometry:wing:tip:chord",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs):

        l2_wing = inputs["data:geometry:wing:root:chord"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        sweep_pos = 0.25 # ???: is the design always fixed?

        x4_wing = sweep_pos * (l2_wing - l4_wing)

        outputs["data:geometry:wing:tip:leading_edge:x:local"] = x4_wing
