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
        
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:weight:propulsion:engine:CG:x", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:propeller:diameter", val=np.nan, units="m")
        self.add_input("data:weight:furniture:passenger_seats:CG:x", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:layout", val=np.nan)
        self.add_input("data:geometry:cabin:NPAX1", val=np.nan)
        self.add_input("data:geometry:cabin:seats:economical:count_by_row", val=np.nan)
        self.add_input("data:geometry:cabin:seats:economical:length", val=np.nan, units="m")

        self.add_output("data:weight:airframe:fuselage:CG:x", units="m")
        self.add_output("data:weight:airframe:flight_controls:CG:x", units="m")
        self.add_output("data:weight:airframe:landing_gear:front:CG:x", units="m")
        self.add_output("data:weight:airframe:pylon:CG:x", units="m")
        self.add_output("data:weight:airframe:paint:CG:x", units="m")
        self.add_output("data:weight:propulsion:fuel_lines:CG:x", units="m")
        self.add_output("data:weight:propulsion:unconsumables:CG:x", units="m")
        self.add_output("data:weight:systems:power:auxiliary_power_unit:CG:x", units="m")
        self.add_output("data:weight:systems:power:electric_systems:CG:x", units="m")
        self.add_output("data:weight:systems:power:hydraulic_systems:CG:x", units="m")
        self.add_output("data:weight:systems:life_support:insulation:CG:x", units="m")
        self.add_output("data:weight:systems:life_support:air_conditioning:CG:x", units="m")
        self.add_output("data:weight:systems:life_support:de-icing:CG:x", units="m")
        self.add_output("data:weight:systems:life_support:cabin_lighting:CG:x", units="m")
        self.add_output("data:weight:systems:life_support:seats_crew_accommodation:CG:x", units="m")
        self.add_output("data:weight:systems:life_support:oxygen:CG:x", units="m")
        self.add_output("data:weight:systems:life_support:safety_equipment:CG:x", units="m")
        self.add_output("data:weight:systems:navigation:CG:x", units="m")
        self.add_output("data:weight:systems:transmission:CG:x", units="m")
        self.add_output("data:weight:systems:operational:radar:CG:x", units="m")
        self.add_output("data:weight:systems:operational:cargo_hold:CG:x", units="m")
        self.add_output("data:weight:furniture:food_water:CG:x", units="m")
        self.add_output("data:weight:furniture:security_kit:CG:x", units="m")
        self.add_output("data:weight:furniture:toilets:CG:x", units="m")
        self.add_output("data:weight:payload:PAX:CG:x", units="m")
        self.add_output("data:weight:payload:rear_fret:CG:x", units="m")
        self.add_output("data:weight:payload:front_fret:CG:x", units="m")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        fus_length = inputs["data:geometry:fuselage:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        lav = inputs["data:geometry:fuselage:front_length"]
        x_cg_engine = inputs["data:weight:propulsion:engine:CG:x"]
        l_spinner = 0.2*inputs["data:geometry:propulsion:propeller:diameter"]
        engine_loc = inputs["data:geometry:propulsion:layout"]
        npax1 = inputs["data:geometry:cabin:NPAX1"]
        count_by_row = inputs["data:geometry:cabin:seats:economical:count_by_row"]
        seats_length = inputs["data:geometry:cabin:seats:economical:length"]
        x_cg_b6 = inputs["weight:propulsion:generator:CG:x"] # ???: where is-it updated?
        
        # Fuselage gravity center
        if engine_loc == 1.0:
            x_cg_a2 = 0.45 * fus_length
        elif engine_loc == 3.0:
            x_cg_a2 = 0.33 * (fus_length - l_spinner)
        else: # FIXME: no equation for configuration 2.0
            raise ValueError('compute_cg_others model only computes propulsion layout equal to 1 or 3!')
        # NLG gravity center
        x_cg_a52 = lav * 0.75
        # Fuel lines gravity center
        x_cg_b2 = x_cg_engine
        # Flight control gravity center
        x_cg_a4 = fa_length + l0_wing #100percent MAC
        # Electric system gravity center
        x_cg_c12 = 0
        # Hydraulic system gravity center
        x_cg_c13 = 0
        # Air conditioning system gravity center
        x_cg_c22 = 0
        # Instruments gravity center
        l_instr = 0.7
        x_cg_c3 = lav + l_instr
        # Seats and passengers gravity center
        nrows = int(npax1/count_by_row)
        x_cg_d2 = lav + l_instr + 0.75 + (0.5* seats_length + (nrows-1)*(0.8 + 0.5*seats_length))/ nrows
        x_cg_pax = x_cg_d2
        
        outputs["data:weight:airframe:fuselage:CG:x"] = x_cg_a2
        outputs["data:weight:airframe:flight_controls:CG:x"] = x_cg_a4
        outputs["data:weight:airframe:landing_gear:front:CG:x"] = x_cg_a52
        outputs["data:weight:airframe:pylon:CG:x"] = 0.0 # A6
        outputs["data:weight:airframe:paint:CG:x"] = 0.0 #A7
        outputs["data:weight:propulsion:fuel_lines:CG:x"] = x_cg_b2
        outputs["data:weight:propulsion:unconsumables:CG:x"] = 0.0 #B3
        outputs["weight:propulsion:batteries:CG:x"] = 0.0 #B5
        outputs["weight:propulsion:IDC:CG:x"] = x_cg_b6 # ???: B7 here, strange formula to update ICE with generator
        outputs["weight:propulsion:propeller:CG:x"] = 0.0 # ???: B8
        outputs["data:weight:systems:power:auxiliary_power_unit:CG:x"] = 0.0 #C11
        outputs["data:weight:systems:power:electric_systems:CG:x"] = x_cg_c12
        outputs["data:weight:systems:power:hydraulic_systems:CG:x"] = x_cg_c13
        outputs["data:weight:systems:life_support:insulation:CG:x"] = 0.0 #C21
        outputs["data:weight:systems:life_support:air_conditioning:CG:x"] = x_cg_c22
        outputs["data:weight:systems:life_support:de-icing:CG:x"] = 0.0 #C23
        outputs["data:weight:systems:life_support:cabin_lighting:CG:x"] = 0.0 #C24
        outputs["data:weight:systems:life_support:seats_crew_accommodation:CG:x"] = 0.0 #C25
        outputs["data:weight:systems:life_support:oxygen:CG:x"] = 0.0 #C26
        outputs["data:weight:systems:life_support:safety_equipment:CG:x"] = 0.0 #C27
        outputs["data:weight:systems:navigation:CG:x"] = x_cg_c3
        outputs["data:weight:systems:transmission:CG:x"] = 0.0 #C4
        outputs["data:weight:systems:operational:radar:CG:x"] = 0.0 #C51
        outputs["data:weight:systems:operational:cargo_hold:CG:x"] = 0.0 #C52
        outputs["data:weight:systems:flight_kit:CG:x"] = 0.0 #C6
        outputs["data:weight:furniture:passenger_seats:CG:x"] = x_cg_d2
        outputs["ata:weight:furniture:food_water:CG:x"] = 0.0 #D3
        outputs["data:weight:furniture:security_kit:CG:x"] = 0.0 #D4
        outputs["data:weight:furniture:toilets:CG:x"] = 0.0 #D5
        outputs["data:weight:payload:PAX:CG:x"] = x_cg_pax
        outputs["data:weight:payload:rear_fret:CG:x"] = 0.0
        outputs["data:weight:payload:front_fret:CG:x"] = 0.0