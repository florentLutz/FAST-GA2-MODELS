"""
    Estimation of other components center of gravities
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


class ComputeOthersCG(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Other components center of gravities estimation """

    def setup(self):
        
        self.add_input("data:geometry:cabin:NPAX", val=np.nan)
        self.add_input("data:geometry:propulsion:layout", val=np.nan)
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:weight:propulsion:engine:CG:x", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:propeller:depth", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)
        self.add_input("data:geometry:cabin:seats:passenger:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:pilot:length", val=np.nan, units="m")
        
        self.add_output("data:weight:airframe:fuselage:CG:x", units="m")
        self.add_output("data:weight:airframe:flight_controls:CG:x", units="m")
        self.add_output("data:weight:airframe:landing_gear:front:CG:x", units="m")
        self.add_output("data:weight:propulsion:fuel_lines:CG:x", units="m")
        self.add_output("data:weight:systems:power:electric_systems:CG:x", units="m")
        self.add_output("data:weight:systems:power:hydraulic_systems:CG:x", units="m")
        self.add_output("data:weight:systems:life_support:air_conditioning:CG:x", units="m")
        self.add_output("data:weight:systems:navigation:CG:x", units="m")
        self.add_output("data:weight:furniture:toilets:CG:x", units="m")
        self.add_output("data:weight:furniture:passenger_seats:CG:x", units="m")
        self.add_output("data:weight:payload:PAX:CG:x", units="m")
        self.add_output("data:weight:payload:rear_fret:CG:x", units="m")
        self.add_output("data:weight:payload:front_fret:CG:x", units="m")

        self.declare_partials(
            "data:weight:airframe:fuselage:CG:x",
            [
                "data:geometry:fuselage:length",
                "data:geometry:propulsion:propeller:depth",
            ],
            method="fd",
        )
        
        self.declare_partials(
            "data:weight:airframe:flight_controls:CG:x",
            [
                "data:geometry:wing:MAC:at25percent:x",
                "data:geometry:wing:MAC:length",
            ],
            method="fd",
        )
        
        self.declare_partials(
            "data:weight:airframe:landing_gear:front:CG:x","data:geometry:fuselage:front_length",
                method="fd",
        )
        
        self.declare_partials(
            "data:weight:propulsion:fuel_lines:CG:x","data:weight:propulsion:engine:CG:x",
                method="fd",
        )
        
        self.declare_partials(
            "data:weight:systems:power:electric_systems:CG:x","data:geometry:fuselage:front_length",
                method="fd",
        )
        
        self.declare_partials(
            "data:weight:systems:power:hydraulic_systems:CG:x","data:geometry:fuselage:front_length",
                method="fd",
        )
        
        self.declare_partials(
            "data:weight:systems:life_support:air_conditioning:CG:x","data:geometry:fuselage:front_length",
                method="fd",
        )
        
        self.declare_partials(
            "data:weight:systems:navigation:CG:x","data:geometry:fuselage:front_length",
                method="fd",
        )
        
        self.declare_partials(
            "data:weight:furniture:passenger_seats:CG:x",
            [
                "data:geometry:fuselage:front_length",
                "data:geometry:cabin:seats:passenger:length",
                "data:geometry:cabin:seats:pilot:length",
            ],
            method="fd",
        )
        
        self.declare_partials(
            "data:weight:payload:PAX:CG:x",
            [
                "data:geometry:fuselage:front_length",
                "data:geometry:cabin:seats:passenger:length",
                "data:geometry:cabin:seats:pilot:length",
            ],
            method="fd",
        )
        
        self.declare_partials(
            "data:weight:payload:rear_fret:CG:x",
            [
                "data:geometry:fuselage:front_length",
                "data:geometry:fuselage:length",
            ],
            method="fd",
        )
        
        self.declare_partials(
            "data:weight:payload:front_fret:CG:x",
            [
                "data:geometry:fuselage:front_length",
                "data:geometry:fuselage:length",
            ],
            method="fd",
        )

    def compute(self, inputs, outputs):
        
        npax1 = inputs["data:geometry:cabin:NPAX"]
        propulsion_loc = inputs["data:geometry:propulsion:layout"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        fus_length = inputs["data:geometry:fuselage:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        lav = inputs["data:geometry:fuselage:front_length"]
        x_cg_engine = inputs["data:weight:propulsion:engine:CG:x"]
        l_spinner = inputs["data:geometry:propulsion:propeller:depth"]
        count_by_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        l_pilot_seat = inputs["data:geometry:cabin:seats:pilot:length"]
        l_pass_seat = inputs["data:geometry:cabin:seats:passenger:length"]
        
        # Fuselage gravity center
        if propulsion_loc == 1.0:
            x_cg_a2 = 0.45 * fus_length
        elif propulsion_loc == 3.0: # nose mount
            x_cg_a2 = 0.33 * (fus_length - l_spinner)
        else: # FIXME: no equation for configuration 2.0
            raise ValueError('compute_cg_others model only computes propulsion layout equal to 1 or 3!')
        # NLG gravity center
        x_cg_a52 = lav * 0.75
        # Fuel lines gravity center
        x_cg_b2 = x_cg_engine
        # Flight control gravity center
        x_cg_a4 = fa_length + l0_wing #100% MAC
        # Electric system gravity center
        x_cg_c12 = 0 * lav
        # Hydraulic system gravity center
        x_cg_c13 = 0 * lav
        # Air conditioning system gravity center
        x_cg_c22 = 0 * lav
        # Instruments gravity center
        l_instr = 0.7
        x_cg_c3 = lav + l_instr
        # Seats and passengers gravity center (hypothesis of 2 pilots)
        nrows = int(npax1/count_by_row)
        x_cg_d2 = lav + l_instr + (l_pilot_seat + (l_pilot_seat + (nrows/2)*l_pass_seat)*npax1)/ (npax1 + 2)
        x_cg_pax = x_cg_d2
        # Fret center of gravity
        x_cg_f_fret = lav + 0.0 * fus_length # ???: should be defined somewhere in the CAB
        x_cg_r_fret = lav + 0.0 * fus_length # ???: should be defined somewhere in the CAB
        
        outputs["data:weight:airframe:fuselage:CG:x"] = x_cg_a2
        outputs["data:weight:airframe:flight_controls:CG:x"] = x_cg_a4
        outputs["data:weight:airframe:landing_gear:front:CG:x"] = x_cg_a52
        outputs["data:weight:propulsion:fuel_lines:CG:x"] = x_cg_b2
        outputs["data:weight:systems:power:electric_systems:CG:x"] = x_cg_c12
        outputs["data:weight:systems:power:hydraulic_systems:CG:x"] = x_cg_c13
        outputs["data:weight:systems:life_support:air_conditioning:CG:x"] = x_cg_c22
        outputs["data:weight:systems:navigation:CG:x"] = x_cg_c3
        outputs["data:weight:furniture:passenger_seats:CG:x"] = x_cg_d2
        outputs["data:weight:payload:PAX:CG:x"] = x_cg_pax
        outputs["data:weight:payload:rear_fret:CG:x"] = x_cg_f_fret
        outputs["data:weight:payload:front_fret:CG:x"] = x_cg_r_fret