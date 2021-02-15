"""
Payload mass computation
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
from openmdao import api as om


class ComputePayload(om.ExplicitComponent):
    """ Computes payload from NPAX """

    def setup(self):
        self.add_input("data:TLAR:NPAX_des", val=np.nan)
        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
        self.add_input("data:geometry:cabin:luggage:max_luggage_weight", val=np.nan, units="kg")
        self.add_input("data:TLAR:m_lug_des", val=np.nan, units="kg")
        self.add_input(
            "settings:weight:aircraft:payload:design_mass_per_passenger",
            val=80.0,
            units="kg",
            desc="Design value of mass per passenger",
        )
        self.add_input(
            "settings:weight:aircraft:payload:max_mass_per_passenger",
            val=90.0,
            units="kg",
            desc="Maximum value of mass per passenger",
        )

        self.add_output("data:weight:aircraft:payload", units="kg")
        self.add_output("data:weight:aircraft:max_payload", units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        npax_des = inputs["data:TLAR:NPAX_des"] + 2.0  # addition of 2 pilots
        npax_max = inputs["data:geometry:cabin:seats:passenger:NPAX_max"] + 2.0  # addition of 2 pilots
        mass_per_pax = inputs["settings:weight:aircraft:payload:design_mass_per_passenger"]
        max_mass_per_pax = inputs["settings:weight:aircraft:payload:max_mass_per_passenger"]
        luggage_weight_des = inputs["data:TLAR:m_lug_des"]
        max_luggage_weight = inputs["data:geometry:cabin:luggage:max_luggage_weight"]

        outputs["data:weight:aircraft:payload"] = npax_des * mass_per_pax + (npax_des - 2.0) * luggage_weight_des
        outputs["data:weight:aircraft:max_payload"] = npax_max * max_mass_per_pax + max_luggage_weight
