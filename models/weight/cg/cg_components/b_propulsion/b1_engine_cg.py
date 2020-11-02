"""
    Estimation of engine(s) center of gravity
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


class ComputeEngineCG(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Engine(s) center of gravity estimation """

    def setup(self):

        self.add_input("data:geometry:propulsion:layout", val=np.nan)
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:length", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:y", val=np.nan, units="m")

        self.add_output("data:weight:propulsion:engine:CG:x", units="m")

        self.declare_partials(
                "data:weight:propulsion:engine:CG:x",
                [
                        "data:geometry:wing:MAC:leading_edge:x:local",
                        "data:geometry:wing:MAC:length",
                        "data:geometry:wing:root:y",
                        "data:geometry:wing:root:chord",
                        "data:geometry:wing:tip:leading_edge:x:local",
                        "data:geometry:wing:tip:y",
                        "data:geometry:wing:tip:chord",
                        "data:geometry:wing:MAC:at25percent:x",
                        "data:geometry:propulsion:nacelle:length",
                        "data:geometry:propulsion:nacelle:y",
                ],
                method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        propulsion_loc = inputs["data:geometry:propulsion:layout"]
        x0_wing = inputs["data:geometry:wing:MAC:leading_edge:x:local"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        y2_wing = inputs["data:geometry:wing:root:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        nacelle_length = inputs["data:geometry:propulsion:nacelle:length"]
        y_nacell = inputs["data:geometry:propulsion:nacelle:y"]
        
        if propulsion_loc == 1.0:
            if y_nacell > y2_wing:  # Nacelle in the tapered part of the wing
                l_wing_nac = l4_wing + (l2_wing - l4_wing) * (y4_wing - y_nacell) / (y4_wing - y2_wing)
                delta_x_nacell = 0.05 * l_wing_nac
                x_nacell_cg = (
                        x4_wing * (y_nacell - y2_wing) / (y4_wing - y2_wing)
                        - delta_x_nacell - 0.2 * nacelle_length
                )
                x_cg_b1 = fa_length - 0.25 * l0_wing - (x0_wing - x_nacell_cg)
            else:  # Nacelle in the straight part of the wing
                l_wing_nac = l2_wing
                delta_x_nacell = 0.05 * l_wing_nac
                x_nacell_cg = -delta_x_nacell - 0.2 * nacelle_length
                x_cg_b1 = fa_length - 0.25 * l0_wing - (x0_wing - x_nacell_cg)
        elif propulsion_loc == 3.0:
            x_cg_b1 = nacelle_length / 2
        else:  # FIXME: no equation for configuration 2.0
            raise ValueError('Model only available for propulsion layout 1.0 or 3.0!')
            
        outputs["data:weight:propulsion:engine:CG:x"] = x_cg_b1
