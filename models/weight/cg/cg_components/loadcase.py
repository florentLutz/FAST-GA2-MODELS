"""
    Estimation of center of gravity for all load cases
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


class ComputeCGLoadCase(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Center of gravity estimation for all load cases"""

    def initialize(self):
        self.options.declare('load_case', default=1, types=int)
    
    def setup(self):
    
        self.load_case = self.options['load_case']
        self.add_input("data:TLAR:NPAX", val=np.nan)
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:weight:payload:PAX:CG:x", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:pilot:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:length", val=np.nan, units="m")
        self.add_input("data:weight:payload:rear_fret:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:payload:front_fret:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft_empty:CG:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft_empty:mass", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:MFW", val=np.nan, units="kg")
        self.add_input("data:weight:propulsion:tank:CG:x", val=np.nan, units="m")
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

        self.add_output("data:weight:aircraft:load_case_"+str(self.load_case)+":CG:MAC_position")

        self.declare_partials(
            "data:weight:aircraft:load_case_"+str(self.load_case)+":CG:MAC_position", 
            [
                "data:geometry:wing:MAC:length",
                "data:geometry:wing:MAC:at25percent:x",
                "data:weight:payload:PAX:CG:x",
                "data:geometry:fuselage:front_length",
                "data:geometry:cabin:seats:pilot:length",
                "data:geometry:cabin:seats:passenger:length",
                "data:weight:payload:rear_fret:CG:x",
                "data:weight:payload:front_fret:CG:x",
                "data:weight:aircraft_empty:CG:x",
                "data:weight:aircraft_empty:mass",
                "data:weight:aircraft:MFW",
                "data:weight:propulsion:tank:CG:x",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs):
         
        npax = inputs["data:TLAR:NPAX"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        cg_pax = inputs["data:weight:payload:PAX:CG:x"]
        lav = inputs["data:geometry:fuselage:front_length"]
        l_pilot_seat = inputs["data:geometry:cabin:seats:pilot:length"]
        l_pass_seat = inputs["data:geometry:cabin:seats:passenger:length"]
        cg_rear_fret = inputs["data:weight:payload:rear_fret:CG:x"]
        cg_front_fret = inputs["data:weight:payload:front_fret:CG:x"]
        x_cg_plane_aft = inputs["data:weight:aircraft_empty:CG:x"]
        m_empty = inputs["data:weight:aircraft_empty:mass"]
        mfw = inputs["data:weight:aircraft:MFW"]
        cg_tank = inputs["data:weight:propulsion:tank:CG:x"]
        design_mass_p_pax = inputs["settings:weight:aircraft:payload:design_mass_per_passenger"]
        max_mass_p_pax = inputs["settings:weight:aircraft:payload:max_mass_per_passenger"]

        if self.load_case == 1:  # all passengers, max fuel but no luggage
            weight_pax = (npax+2) * design_mass_p_pax
            weight_rear_fret = 0.0
            weight_front_fret = 0.0
        elif self.load_case == 2:  # all passengers, max fuel and 20kg luggage per pax
            weight_pax = (npax+2) * design_mass_p_pax
            weight_rear_fret = npax * 20.0
            weight_front_fret = 0.0
        elif self.load_case == 3:  # all passengers, no fuel and 20kg luggage per pax
            weight_pax = (npax+2) * design_mass_p_pax
            weight_rear_fret = npax * 20.0
            weight_front_fret = 0.0
            mfw = 0.0
        elif self.load_case == 4:  # all passengers (max mass), max fuel but no luggage
            weight_pax = (npax+2) * max_mass_p_pax
            weight_rear_fret = 0.0
            weight_front_fret = 0.0
        elif self.load_case == 5:  # only 1 pilot (over-write pax-CG), max fuel, no luggage
            l_instr = 0.7
            cg_pax = lav + l_instr + l_pilot_seat/2.0
            weight_pax = design_mass_p_pax
            weight_rear_fret = 0.0
            weight_front_fret = 0.0
        elif self.load_case == 6:  # only 1 pilot and 1 passenger (over-write pax-CG), max fuel, no luggage
            l_instr = 0.7
            cg_pax = lav + l_instr + l_pilot_seat*0.75 + l_pass_seat*0.25
            weight_pax = 160
            weight_rear_fret = 0.0
            weight_front_fret = 0.0
        else:
            raise ValueError('compute_cg_loadcase model only computes load case 1 to 6!')
        
        weight_pl = weight_pax + weight_rear_fret + weight_front_fret
        x_cg_pl = (
            weight_pax * cg_pax
            + weight_rear_fret * cg_rear_fret
            + weight_front_fret * cg_front_fret
        ) / weight_pl
        x_cg_plane_pl = (
                        m_empty * x_cg_plane_aft
                        + mfw * cg_tank
                        + weight_pl * x_cg_pl
        ) / (m_empty + mfw + weight_pl)  # forward
        cg_ratio_pl = (x_cg_plane_pl - fa_length + 0.25 * l0_wing) / l0_wing
        
        
        outputs["data:weight:aircraft:load_case_"+str(self.load_case)+":CG:MAC_position"] = cg_ratio_pl
