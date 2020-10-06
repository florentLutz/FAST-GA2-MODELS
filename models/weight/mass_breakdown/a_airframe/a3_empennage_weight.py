"""
Estimation of empennage weight
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


class EmpennageWeight(om.ExplicitComponent):
    """
    Weight estimation for tail planes (only horizontal)

    # TODO: Based on :????????????
    """

    def setup(self):
        
        self.add_input("data:mission:sizing:cs23:sizing_factor_ultimate", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:geometry:horizontal_tail:wet_area", val=np.nan, units="ft**2")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="ft")
        self.add_input("data:geometry:horizontal_tail:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="ft")

        self.add_output("data:weight:airframe:vertical_tail:mass", units="lb")
        self.add_output("data:weight:airframe:horizontal_tail:mass", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor_ultimate"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        wet_area = inputs["data:geometry:horizontal_tail:wet_area"]
        span = inputs["data:geometry:horizontal_tail:span"]
        thickness_ratio = inputs["data:geometry:horizontal_tail:thickness_ratio"]
        chord = inputs["data:geometry:horizontal_tail:root:chord"]
        thickness = chord*thickness_ratio
        
        a32 = (
            98.5*((mtow*sizing_factor_ultimate/10**5)**0.87
            *(wet_area/100)**1.2
            *0.289*(span/thickness)**0.5)**0.458
        )# mass formula in lb

        outputs["data:weight:airframe:vertical_tail:mass"] = 0.0  # TODO: explain why not evaluated
        outputs["data:weight:airframe:horizontal_tail:mass"] = a32
