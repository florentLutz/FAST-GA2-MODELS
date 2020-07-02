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


class LandingGearWeight(om.ExplicitComponent):
    """
    Weight estimation for landing gears

    # TODO: Based on :????????????
    """

    def setup(self):
        
        self.add_input("data:mission:sizing:cs23:sizing_factor_ultimate", val=np.nan) # TODO: confirm it's a ratio!, to be added to xml variables        
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:geometry:landing_gear:height", val=np.nan, units="m")
        
        self.add_output("data:weight:airframe:landing_gear:main:mass", units="kg") # old weight_A51
        self.add_output("data:weight:airframe:landing_gear:front:mass", units="kg") # old weight_A52

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor_ultimate"]
        mtow = inputs["data:weight:aircraft:MTOW"]*2.20462 # Takeoff weight in lb
        height = inputs["data:geometry:landing_gear:height"]*3.28084 # conversion to feet
        
        l_sm = height/3 # Shock strut length for MLG
        a5 = 0.054*l_sm**0.501*(mtow*sizing_factor_ultimate)**0.684
        a51 = a5*2/3 # mass in lb
        a52 = a5*1/3 # mass in lb

        outputs["data:weight:airframe:landing_gear:main:mass"] = a51/ 2.20462 # converted to kg
        outputs["data:weight:airframe:landing_gear:front:mass"] = a52/ 2.20462 # converted to kg
