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
        self.add_output("data:geometry:fuselage:wetted_area", units="m**2")

        self.declare_partials("*", "*", method="fd") 

    def compute(self, inputs, outputs):
        
        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        fus_length = inputs["data:geometry:fuselage:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]
        
        # Cabine total length
        cabin_length = 0.81 * fus_length # ???: apparently the formula seems strange
        # Calculate wet area
        fus_dia = math.sqrt(b_f * h_f) # equivalent diameter of the fuselage
        cyl_length = fus_length - lav - lar
        wet_area_nose = 2.45 * fus_dia * lav
        wet_area_cyl = 3.1416 * fus_dia * cyl_length
        wet_area_tail = 2.3 * fus_dia * lar
        wet_area_fus = (wet_area_nose + wet_area_cyl + wet_area_tail)
        
        outputs["data:geometry:cabin:length"] = cabin_length
        outputs["data:geometry:fuselage:wetted_area"] = wet_area_fus


class ComputeFuselageGeometryCabinSizing(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Geometry of fuselage part A - Cabin (Commercial) estimation """

    def setup(self):

        self.add_input("data:TLAR:NPAX", val=np.nan)
        self.add_input("data:geometry:propulsion:layout", val=np.nan)
        self.add_input("data:geometry:propulsion:propeller:diameter", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:engine:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:length", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:length", val=np.nan, units="m")

        self.add_output("data:geometry:cabin:NPAX1")
        self.add_output("data:geometry:fuselage:length", units="m")
        self.add_output("data:geometry:fuselage:maximum_width", units="m")
        self.add_output("data:geometry:fuselage:maximum_height", units="m")
        self.add_output("data:geometry:fuselage:front_length", units="m")
        self.add_output("data:geometry:fuselage:rear_length", units="m")
        self.add_output("data:geometry:fuselage:PAX_length", units="m")
        self.add_output("data:geometry:cabin:length", units="m")
        self.add_output("data:geometry:fuselage:wetted_area", units="m**2")
        self.add_output("data:geometry:fuselage:luggage_length")
        
        self.declare_partials("*", "*", method="fd") 

    def compute(self, inputs, outputs):
        
        npax = inputs["data:TLAR:NPAX"]
        engine_loc = inputs["data:geometry:propulsion:layout"]
        propeller_dia = inputs["data:geometry:propulsion:propeller:diameter"]
        engine_length = inputs["data:geometry:propulsion:engine:length"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        ht_lp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        vt_lp = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        ht_length = inputs["data:geometry:horizontal_tail:length"]
        vt_length = inputs["data:geometry:vertical_tail:length"]
        
        # Length of instrument panel
        l_instr = 0.7
        # Length of pax cabin = Length between instrument pannel and first seats (0.75m) + Length of 2 seat row (0.8m)
        lpax = 0.75 + (npax/2)*0.8
        # Cabin width considered is for side by side seaters
        wcabin = 1.13
        r_i = wcabin / 2
        radius = 1.06 * r_i 
        # Cylindrical fuselage
        b_f = 2 * radius
        # 0.14m is the distance between both lobe centers of the fuselage
        h_f = b_f + 0.14
        # Lugage length
        l_lug = npax * 0.20 / ( math.pi * radius**2)
        # Cabine total length
        cabin_length = l_instr + lpax + l_lug
        # Calculate nose length
        if engine_loc == 3.0:
            l_spinner = 0.2 * propeller_dia
            l_engin_comp = 1.5 * engine_length
            lav = l_engin_comp + l_spinner
        else:
            lav = 1.7 * h_f 
        # Calculate fuselage length
        fus_length = fa_length + max(ht_lp + 0.75*ht_length, vt_lp + 0.75*vt_length )
        lar = fus_length - (lav + cabin_length) # !!!: I added this because never calculated
        # Calculate wet area
        fus_dia = math.sqrt(b_f * h_f) # equivalent diameter of the fuselage
        cyl_length = fus_length - lav - lar
        wet_area_nose = 2.45 * fus_dia * lav
        wet_area_cyl = 3.1416 * fus_dia * cyl_length
        wet_area_tail = 2.3 * fus_dia * lar
        wet_area_fus = (wet_area_nose + wet_area_cyl + wet_area_tail)
        
        outputs["data:geometry:cabin:NPAX1"] = npax
        outputs["data:geometry:fuselage:length"] = fus_length
        outputs["data:geometry:fuselage:maximum_width"] = b_f
        outputs["data:geometry:fuselage:maximum_height"] = h_f
        outputs["data:geometry:fuselage:front_length"] = lav
        outputs["data:geometry:fuselage:rear_length"] = lar
        outputs["data:geometry:fuselage:PAX_length"] = lpax
        outputs["data:geometry:cabin:length"] = cabin_length
        outputs["data:geometry:fuselage:wetted_area"] = wet_area_fus
        outputs["data:geometry:fuselage:luggage_length"] = l_lug
