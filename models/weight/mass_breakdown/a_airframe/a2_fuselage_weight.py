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
import math

from fastoad.utils.physics import Atmosphere

class ComputeFuselageWeight(om.ExplicitComponent):
    """
    Fuselage weight estimation

    Based on : Nicolai, Leland M., and Grant E. Carichner. Fundamentals of aircraft and airship design,
    Volume 1–Aircraft Design. American Institute of Aeronautics and Astronautics, 2010.

    Can also be found in : Gudmundsson, Snorri. General aviation aircraft design: Applied Methods and Procedures.
    Butterworth-Heinemann, 2013. Equation (6-25)
    """

    def setup(self):
        
        self.add_input("data:mission:sizing:cs23:sizing_factor_ultimate", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="lb")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:TLAR:v_limit", val=np.nan, units="kn")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")
        
        self.add_output("data:weight:airframe:fuselage:mass", units="lb")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        sizing_factor_ultimate = inputs["data:mission:sizing:cs23:sizing_factor_ultimate"]
        mtow = inputs["data:weight:aircraft:MTOW"]
        maximum_width = inputs["data:geometry:fuselage:maximum_width"]
        maximum_height = inputs["data:geometry:fuselage:maximum_height"]
        fus_length = inputs["data:geometry:fuselage:length"]
        limit_speed = inputs["data:TLAR:v_limit"]
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]

        rho_cruise = Atmosphere(cruise_alt).density
        rho_SL     = Atmosphere(    0.0   ).density

        limit_speed_KEAS = limit_speed * math.sqrt(rho_cruise / rho_SL)

        a2 = 200.0*(
                (mtow*sizing_factor_ultimate / (10.0**5.0))**0.286
                * (fus_length * 3.28084/10.0)**0.857
                * (maximum_width + maximum_height) * 3.28084/10.0
                * (limit_speed_KEAS/100.0)**0.338
        )**1.1  # mass formula in lb
            
        outputs["data:weight:airframe:fuselage:mass"] = a2
