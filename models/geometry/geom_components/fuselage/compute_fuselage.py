"""
    Estimation of geometry of fuselage part A - Cabin (Commercial)
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
from ....propulsion.fuel_propulsion.base import FuelEngineSet
from fastoad import BundleLoader


class ComputeFuselageGeometryBasic(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Geometry of fuselage - Cabin length defined with total fuselage length input (no sizing)
    """

    def setup(self):
        
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")

        self.add_output("data:geometry:cabin:length", units="m")
        self.add_output("data:geometry:fuselage:wet_area", units="m**2")

        self.declare_partials("*", "*", method="fd") 

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        b_f = inputs["data:geometry:fuselage:maximum_width"]
        h_f = inputs["data:geometry:fuselage:maximum_height"]
        fus_length = inputs["data:geometry:fuselage:length"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]
        
        # Cabin total length
        cabin_length = fus_length - (lav + lar)
        # Calculate wet area
        fus_dia = math.sqrt(b_f * h_f)  # equivalent diameter of the fuselage
        cyl_length = fus_length - lav - lar
        wet_area_nose = 2.45 * fus_dia * lav
        wet_area_cyl = 3.1416 * fus_dia * cyl_length
        wet_area_tail = 2.3 * fus_dia * lar
        wet_area_fus = (wet_area_nose + wet_area_cyl + wet_area_tail)
        
        outputs["data:geometry:cabin:length"] = cabin_length
        outputs["data:geometry:fuselage:wet_area"] = wet_area_fus


class ComputeFuselageGeometryCabinSizingFD(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Geometry of fuselage - Cabin is sized based on layout (seats, aisle...) and HTP/VTP position (Fixed tail Distance).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
        self.add_input("data:geometry:cabin:seats:pilot:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:pilot:width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)
        self.add_input("data:geometry:cabin:aisle_width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:luggage:luggage_mass_max", val=np.nan, units="kg")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:cabin:NPAX")
        self.add_output("data:geometry:plane:length", units="m")
        self.add_output("data:geometry:fuselage:length", val=10.0, units="m")
        self.add_output("data:geometry:fuselage:maximum_width", units="m")
        self.add_output("data:geometry:fuselage:maximum_height", units="m")
        self.add_output("data:geometry:fuselage:front_length", units="m")
        self.add_output("data:geometry:fuselage:rear_length", units="m")
        self.add_output("data:geometry:fuselage:PAX_length", units="m")
        self.add_output("data:geometry:cabin:length", units="m")
        self.add_output("data:geometry:fuselage:wet_area", units="m**2")
        self.add_output("data:geometry:fuselage:luggage_length", units="m")
        
        self.declare_partials("*", "*", method="fd")  # FIXME: declare proper partials without int values

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propulsion_model = FuelEngineSet(self._engine_wrapper.get_model(inputs), 1.0)
        npax_max = inputs["data:geometry:cabin:seats:passenger:NPAX_max"]
        l_pilot_seats = inputs["data:geometry:cabin:seats:pilot:length"]
        w_pilot_seats = inputs["data:geometry:cabin:seats:pilot:width"]
        l_pass_seats = inputs["data:geometry:cabin:seats:passenger:length"]
        w_pass_seats = inputs["data:geometry:cabin:seats:passenger:width"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        w_aisle = inputs["data:geometry:cabin:aisle_width"]
        luggage_mass_max = inputs["data:geometry:cabin:luggage:luggage_mass_max"]
        prop_layout = inputs["data:geometry:propulsion:layout"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        ht_lp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        vt_lp = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        ht_length = inputs["data:geometry:horizontal_tail:MAC:length"]
        vt_length = inputs["data:geometry:vertical_tail:MAC:length"]
        sweep_25_vt = inputs["data:geometry:vertical_tail:sweep_25"]
        b_v = inputs["data:geometry:vertical_tail:span"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        b_h = inputs["data:geometry:horizontal_tail:span"]

        # Length of instrument panel
        l_instr = 0.7
        # Length of pax cabin
        npax = math.ceil(float(npax_max)/float(seats_p_row)) * float(seats_p_row)
        n_rows = npax / float(seats_p_row)
        lpax = l_pilot_seats + n_rows * l_pass_seats
        # Cabin width considered is for side by side seats
        wcabin = max(2*w_pilot_seats, seats_p_row*w_pass_seats + w_aisle)
        r_i = wcabin / 2
        radius = 1.06 * r_i 
        # Cylindrical fuselage
        b_f = 2 * radius
        # 0.14m is the distance between both lobe centers of the fuselage
        h_f = b_f + 0.14
        # Luggage length (80% of internal radius section can be filled with luggage)
        luggage_density = 161.0  # In kg/m3
        l_lug = (luggage_mass_max / luggage_density) / (0.8 * math.pi * r_i ** 2)
        # Cabin total length
        cabin_length = l_instr + lpax + l_lug
        # Calculate nose length
        if prop_layout == 3.0:  # engine located in nose
            _, _, propulsion_length, _, _, spinner_length = propulsion_model.compute_dimensions()
            lav = propulsion_length + spinner_length
        else:
            lav = 1.40 * h_f
            # Used to be 1.7, supposedly as an A320 according to FAST legacy. Results on the BE76 tend to say it is
            # around 1.40, though it varies a lot depending on the airplane and its use
        # Calculate fuselage length
        fus_length = fa_length + max(ht_lp + 0.75 * ht_length, vt_lp + 0.75 * vt_length)
        plane_length = fa_length + max(ht_lp + 0.75 * ht_length + b_h / 2.0 * math.tan(sweep_25_ht * math.pi / 180),
                                       vt_lp + 0.75 * vt_length + b_v * math.tan(sweep_25_vt * math.pi / 180))
        lar = fus_length - (lav + cabin_length)
        # Calculate wet area
        fus_dia = math.sqrt(b_f * h_f)  # equivalent diameter of the fuselage
        cyl_length = fus_length - lav - lar
        wet_area_nose = 2.45 * fus_dia * lav
        wet_area_cyl = 3.1416 * fus_dia * cyl_length
        wet_area_tail = 2.3 * fus_dia * lar
        wet_area_fus = (wet_area_nose + wet_area_cyl + wet_area_tail)
        
        outputs["data:geometry:cabin:NPAX"] = npax
        outputs["data:geometry:fuselage:length"] = fus_length
        outputs["data:geometry:plane:length"] = plane_length
        outputs["data:geometry:fuselage:maximum_width"] = b_f
        outputs["data:geometry:fuselage:maximum_height"] = h_f
        outputs["data:geometry:fuselage:front_length"] = lav
        outputs["data:geometry:fuselage:rear_length"] = lar
        outputs["data:geometry:fuselage:PAX_length"] = lpax
        outputs["data:geometry:cabin:length"] = cabin_length
        outputs["data:geometry:fuselage:wet_area"] = wet_area_fus
        outputs["data:geometry:fuselage:luggage_length"] = l_lug


class ComputeFuselageGeometryCabinSizingFL(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """
    Geometry of fuselage - Cabin is sized based on layout (seats, aisle...) and additional rear length (Fixed Length).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)

        self.add_input("data:geometry:cabin:seats:passenger:NPAX_max", val=np.nan)
        self.add_input("data:geometry:cabin:seats:pilot:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:pilot:width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:length", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:seats:passenger:count_by_row", val=np.nan)
        self.add_input("data:geometry:cabin:aisle_width", val=np.nan, units="m")
        self.add_input("data:geometry:cabin:luggage:luggage_mass_max", val=np.nan, units="kg")
        self.add_input("data:geometry:fuselage:rear_length", units="m")

        self.add_output("data:geometry:cabin:NPAX")
        self.add_output("data:geometry:fuselage:length", val=10.0, units="m")
        self.add_output("data:geometry:fuselage:maximum_width", units="m")
        self.add_output("data:geometry:fuselage:maximum_height", units="m")
        self.add_output("data:geometry:fuselage:front_length", units="m")
        self.add_output("data:geometry:fuselage:PAX_length", units="m")
        self.add_output("data:geometry:cabin:length", units="m")
        self.add_output("data:geometry:fuselage:wet_area", units="m**2")
        self.add_output("data:geometry:fuselage:luggage_length", units="m")

        self.declare_partials("*", "*", method="fd")  # FIXME: declare proper partials without int values

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        propulsion_model = FuelEngineSet(self._engine_wrapper.get_model(inputs), 1.0)
        npax_max = inputs["data:geometry:cabin:seats:passenger:NPAX_max"]
        l_pilot_seats = inputs["data:geometry:cabin:seats:pilot:length"]
        w_pilot_seats = inputs["data:geometry:cabin:seats:pilot:width"]
        l_pass_seats = inputs["data:geometry:cabin:seats:passenger:length"]
        w_pass_seats = inputs["data:geometry:cabin:seats:passenger:width"]
        seats_p_row = inputs["data:geometry:cabin:seats:passenger:count_by_row"]
        w_aisle = inputs["data:geometry:cabin:aisle_width"]
        luggage_mass_max = inputs["data:geometry:cabin:luggage:luggage_mass_max"]
        prop_layout = inputs["data:geometry:propulsion:layout"]
        lar = inputs["data:geometry:fuselage:rear_length"]

        # Length of instrument panel
        l_instr = 0.7
        # Length of pax cabin
        # noinspection PyBroadException
        npax = math.ceil(float(npax_max) / float(seats_p_row)) * float(seats_p_row)
        n_rows = npax / float(seats_p_row)
        lpax = l_pilot_seats + n_rows * l_pass_seats
        # Cabin width considered is for side by side seats
        wcabin = max(2 * w_pilot_seats, seats_p_row * w_pass_seats + w_aisle)
        r_i = wcabin / 2
        radius = 1.06 * r_i
        # Cylindrical fuselage
        b_f = 2 * radius
        # 0.14m is the distance between both lobe centers of the fuselage
        h_f = b_f + 0.14
        # Luggage length (80% of internal radius section can be filled with luggage)
        luggage_density = 161.0  # In kg/m3
        l_lug = (luggage_mass_max / luggage_density) / (0.8 * math.pi * r_i ** 2)
        # Cabin total length
        cabin_length = l_instr + lpax + l_lug
        # Calculate nose length
        if prop_layout == 3.0:  # engine located in nose
            _, _, propulsion_length, _ = propulsion_model.compute_dimensions()
            lav = propulsion_length
        else:
            lav = 1.7 * h_f
            # Calculate fuselage length
        fus_length = lav + cabin_length + lar

        # Calculate wet area
        fus_dia = math.sqrt(b_f * h_f)  # equivalent diameter of the fuselage
        cyl_length = fus_length - lav - lar
        wet_area_nose = 2.45 * fus_dia * lav
        wet_area_cyl = 3.1416 * fus_dia * cyl_length
        wet_area_tail = 2.3 * fus_dia * lar
        wet_area_fus = (wet_area_nose + wet_area_cyl + wet_area_tail)

        outputs["data:geometry:cabin:NPAX"] = npax
        outputs["data:geometry:fuselage:length"] = fus_length
        outputs["data:geometry:fuselage:maximum_width"] = b_f
        outputs["data:geometry:fuselage:maximum_height"] = h_f
        outputs["data:geometry:fuselage:front_length"] = lav
        outputs["data:geometry:fuselage:PAX_length"] = lpax
        outputs["data:geometry:cabin:length"] = cabin_length
        outputs["data:geometry:fuselage:wet_area"] = wet_area_fus
        outputs["data:geometry:fuselage:luggage_length"] = l_lug
