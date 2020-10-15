"""
    Estimation of vertical induced yawing moment
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
import openmdao.api as om


class ComputeCnBetaVT(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Vertical tail yawing moment estimation """

    def setup(self):
        
        self.add_input("data:geometry:has_T_tail", val=np.nan)
        self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)

        self.add_output("data:aerodynamics:vertical_tail:cruise:CnBeta")

        self.declare_partials(
                "data:aerodynamics:vertical_tail:cruise:CnBeta",
                [
                    "data:aerodynamics:cruise:mach",
                    "data:geometry:vertical_tail:sweep_25",
                    "data:geometry:vertical_tail:aspect_ratio",
                ],
                method="fd",
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        tail_type = np.round(inputs["data:geometry:has_T_tail"])
        cruise_mach = inputs["data:aerodynamics:cruise:mach"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        k_ar_effective = 2.9 if tail_type == 1.00 else 1.55
        lambda_vt = inputs["data:geometry:vertical_tail:aspect_ratio"] * k_ar_effective
        
        beta = math.sqrt(1 - cruise_mach ** 2)
        cn_beta = (
            0.8
            * 2
            * math.pi
            * lambda_vt
            / (
                2
                + math.sqrt(
                    4
                    + lambda_vt ** 2
                    * beta ** 2
                    / 0.95 ** 2
                    * (1 + (math.tan(sweep_25_vt / 180.0 * math.pi)) ** 2 / beta ** 2)
                )
            )
        )

        outputs["data:aerodynamics:vertical_tail:cruise:CnBeta"] = cn_beta
