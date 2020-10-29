"""
Estimation of fuselage weight
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


class ComputeFuselageWeight(om.ExplicitComponent):
    """
    Fuselage weight estimation

    # TODO: Based on :????????????
    """

    def setup(self):
        
        self.add_input("data:mission:sizing:cs23:sizing_factor_ultimate", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="kn")
        
        self.add_output("data:weight:airframe:fuselage:mass", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor_ultimate"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        maximum_width = inputs["data:geometry:fuselage:maximum_width"]
        maximum_height = inputs["data:geometry:fuselage:maximum_height"]
        fus_length = inputs["data:geometry:fuselage:length"]
        cruise_speed = inputs["data:TLAR:v_cruise"]
        
        a2 = 200.0*(
                (mtow*sizing_factor_ultimate / (10.0**5.0))**0.286
                * (fus_length * 3.28084/10.0)**0.857
                * (maximum_width + maximum_height) * 3.28084/10.0
                * (cruise_speed/100.0)**0.338
        )**1.1  # mass formula in lb
            
        outputs["data:weight:airframe:fuselage:mass"] = a2
