"""
    Estimation of passenger seats center of gravity
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


class ComputePassengerSeatsCG(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Passenger seats center of gravity estimation """

    def setup(self):
        self.add_input("data:TLAR:NPAX", val=np.nan)
        self.add_input("data:geometry:cabin:NPAX", val=np.nan)
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)
        self.add_input("data:geometry:cabin:seats:pilot:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:length", val=np.nan, units="m")

        self.add_output("data:weight:furniture:passenger_seats:CG:x", units="m")
        
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        npax = inputs["data:TLAR:NPAX"]
        npax1 = inputs["data:geometry:cabin:NPAX"]
        lav = inputs["data:geometry:fuselage:front_length"]
        count_by_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        l_pilot_seat = inputs["data:geometry:cabin:seats:pilot:length"]
        l_pass_seat = inputs["data:geometry:cabin:seats:passenger:length"]

        # Instruments length
        l_instr = 0.7
        # Seats and passengers gravity center (hypothesis of 2 pilots)
        nrows = int(npax1/count_by_row)
        x_cg_d2 = lav + l_instr + l_pilot_seat * 2./(npax + 2.)
        for idx in range(nrows):
            length = l_pilot_seat + (idx + 0.5)*l_pass_seat
            nb_pers = min(count_by_row, npax-idx*count_by_row)
            x_cg_d2 = x_cg_d2 + length*nb_pers/(npax + 2.)

        outputs["data:weight:furniture:passenger_seats:CG:x"] = x_cg_d2
