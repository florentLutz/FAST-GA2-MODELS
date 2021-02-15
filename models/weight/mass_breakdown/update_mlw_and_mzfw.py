"""
Main component for mass breakdown
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


class UpdateMLWandMZFW(ExplicitComponent):
    """
    Computes Maximum Landing Weight and Maximum Zero Fuel Weight from
    Overall Empty Weight and Maximum Payload.
    """

    def setup(self):
        self.add_input("data:weight:aircraft:OWE", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:max_payload", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:payload", val=np.nan, units="kg")
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="kn")

        self.add_output("data:weight:aircraft:MZFW", units="kg")
        self.add_output("data:weight:aircraft:ZFW", units="kg")
        self.add_output("data:weight:aircraft:MLW", units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        owe = inputs["data:weight:aircraft:OWE"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        max_pl = inputs["data:weight:aircraft:max_payload"]
        pl = inputs["data:weight:aircraft:payload"]
        cruise_ktas = inputs["data:TLAR:v_cruise"]
        mzfw = owe + max_pl
        zfw = owe + pl

        if cruise_ktas > 250:
            mlw = 1.06 * mzfw

        else:
            mlw = mtow

        outputs["data:weight:aircraft:MZFW"] = mzfw
        outputs["data:weight:aircraft:ZFW"] = zfw
        outputs["data:weight:aircraft:MLW"] = mlw
