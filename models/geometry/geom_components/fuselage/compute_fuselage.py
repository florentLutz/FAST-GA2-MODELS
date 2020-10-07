"""
    Estimation of geometry of fuselase part A - Cabin (Commercial)
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
import math
from openmdao.core.explicitcomponent import ExplicitComponent


class ComputeFuselageGeometryBasic(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Geometry of fuselage part A - Cabin (Commercial) estimation """

    def setup(self):
        
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")

        self.add_output("data:geometry:cabin:length", units="m")
        self.add_output("data:geometry:fuselage:wet_area", units="m**2")

        self.declare_partials("*", "*", method="fd") 

    def compute(self, inputs, outputs):
        
        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        fus_length = inputs["data:geometry:fuselage:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]
        
        # Cabine total length
        cabin_length = fus_length - (lav + lar)
        # Calculate wet area
        fus_dia = math.sqrt(b_f * h_f) # equivalent diameter of the fuselage
        cyl_length = fus_length - lav - lar
        wet_area_nose = 2.45 * fus_dia * lav
        wet_area_cyl = 3.1416 * fus_dia * cyl_length
        wet_area_tail = 2.3 * fus_dia * lar
        wet_area_fus = (wet_area_nose + wet_area_cyl + wet_area_tail)
        
        outputs["data:geometry:cabin:length"] = cabin_length
        outputs["data:geometry:fuselage:wet_area"] = wet_area_fus


class ComputeFuselageGeometryCabinSizing(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Geometry of fuselage part A - Cabin (Commercial) estimation """

    def setup(self):

        self.add_input("data:TLAR:NPAX", val=np.nan)
        self.add_input("data:geometry:cabin:seats:pilot:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:pilot:width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)
        self.add_input("data:geometry:cabin:aisle_width", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:layout", val=np.nan)
        self.add_input("data:geometry:propulsion:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:length", val=np.nan, units="m")

        self.add_output("data:geometry:cabin:NPAX", units="m")
        self.add_output("data:geometry:fuselage:length", units="m")
        self.add_output("data:geometry:fuselage:maximum_width", units="m")
        self.add_output("data:geometry:fuselage:maximum_height", units="m")
        self.add_output("data:geometry:fuselage:front_length", units="m")
        self.add_output("data:geometry:fuselage:rear_length", units="m")
        self.add_output("data:geometry:fuselage:PAX_length", units="m")
        self.add_output("data:geometry:cabin:length", units="m")
        self.add_output("data:geometry:fuselage:wet_area", units="m**2")
        self.add_output("data:geometry:fuselage:luggage_length", units="m")
        
        self.declare_partials("*", "*", method="fd") # FIXME: declara proper partials without int values

    def compute(self, inputs, outputs):
        
        npax = inputs["data:TLAR:NPAX"]
        l_pilot_seats = inputs["data:geometry:cabin:seats:pilot:length"]
        w_pilot_seats = inputs["data:geometry:cabin:seats:pilot:width"]
        l_pass_seats = inputs["data:geometry:cabin:seats:passenger:length"]
        w_pass_seats = inputs["data:geometry:cabin:seats:passenger:width"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        w_aisle = inputs["data:geometry:cabin:aisle_width"]
        engine_loc = inputs["data:geometry:propulsion:layout"]
        propulsion_length = inputs["data:geometry:propulsion:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        ht_lp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        vt_lp = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        ht_length = inputs["data:geometry:horizontal_tail:MAC:length"]
        vt_length = inputs["data:geometry:vertical_tail:MAC:length"]
        
        # Length of instrument panel
        l_instr = 0.7
        # Length of pax cabin
        npax_1 = math.ceil(npax/seats_p_row)*seats_p_row
        n_rows = npax_1 / seats_p_row
        lpax = l_pilot_seats + n_rows*l_pass_seats
        # Cabin width considered is for side by side seaters
        wcabin = max(2*w_pilot_seats, seats_p_row*w_pass_seats + w_aisle)
        r_i = wcabin / 2
        radius = 1.06 * r_i 
        # Cylindrical fuselage
        b_f = 2 * radius
        # 0.14m is the distance between both lobe centers of the fuselage
        h_f = b_f + 0.14
        # Lugage length
        l_lug = npax_1 * 0.20 / ( math.pi * radius**2)
        # Cabine total length
        cabin_length = l_instr + lpax + l_lug
        # Calculate nose length
        if engine_loc == 3.0: # engine in nose
            lav = propulsion_length
        else:
            lav = 1.7 * h_f 
        # Calculate fuselage length
        fus_length = fa_length + max(ht_lp+0.75*ht_length, vt_lp+0.75*vt_length)
        lar = fus_length - (lav + cabin_length)
        # Calculate wet area
        fus_dia = math.sqrt(b_f * h_f) # equivalent diameter of the fuselage
        cyl_length = fus_length - lav - lar
        wet_area_nose = 2.45 * fus_dia * lav
        wet_area_cyl = 3.1416 * fus_dia * cyl_length
        wet_area_tail = 2.3 * fus_dia * lar
        wet_area_fus = (wet_area_nose + wet_area_cyl + wet_area_tail)
        
        outputs["data:geometry:cabin:NPAX"] = npax_1
        outputs["data:geometry:fuselage:length"] = fus_length
        outputs["data:geometry:fuselage:maximum_width"] = b_f
        outputs["data:geometry:fuselage:maximum_height"] = h_f
        outputs["data:geometry:fuselage:front_length"] = lav
        outputs["data:geometry:fuselage:rear_length"] = lar
        outputs["data:geometry:fuselage:PAX_length"] = lpax
        outputs["data:geometry:cabin:length"] = cabin_length
        outputs["data:geometry:fuselage:wet_area"] = wet_area_fus
        outputs["data:geometry:fuselage:luggage_length"] = l_lug
