"""
Estimation of passenger seats weight
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



class PassengerSeatsWeight(om.ExplicitComponent):
    """
    Weight estimation for passenger seats

    # TODO: Based on :????????????
    """

    def setup(self):
        
        self.add_input("data:TLAR:NPAX", val=np.nan)
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        
        self.add_output("data:weight:furniture:passenger_seats:mass", units="kg")

        self.declare_partials("data:weight:furniture:passenger_seats:mass", "data:weight:aircraft:MTOW", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        npax = inputs["data:TLAR:NPAX"] + 2.0 # includes 2 pilots seats
        mtow = inputs["data:weight:aircraft:MTOW"]*2.20462 # Takeoff weight in lb
        
        d2 = 0.412*npax**1.145*mtow**0.489 # mass in lb
        
        outputs["data:weight:furniture:passenger_seats:mass"] = d2 /2.20462 # converted to kg
