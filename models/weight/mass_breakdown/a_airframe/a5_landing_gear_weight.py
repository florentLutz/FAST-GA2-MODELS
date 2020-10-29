"""
Estimation of landing gear weight
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


class ComputeLandingGearWeight(om.ExplicitComponent):
    """
    Weight estimation for landing gears

    # TODO: Based on :????????????
    """

    def setup(self):
        
        self.add_input("data:mission:sizing:cs23:sizing_factor_ultimate", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:geometry:landing_gear:height", val=np.nan, units="ft")
        
        self.add_output("data:weight:airframe:landing_gear:main:mass", units="lb")
        self.add_output("data:weight:airframe:landing_gear:front:mass", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor_ultimate"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        height = inputs["data:geometry:landing_gear:height"]
        
        l_sm = height/3  # Shock strut length for MLG
        a5 = 0.054*l_sm**0.501*(mtow*sizing_factor_ultimate)**0.684  # mass formula in lb
        a51 = a5*2/3
        a52 = a5*1/3

        outputs["data:weight:airframe:landing_gear:main:mass"] = a51
        outputs["data:weight:airframe:landing_gear:front:mass"] = a52
