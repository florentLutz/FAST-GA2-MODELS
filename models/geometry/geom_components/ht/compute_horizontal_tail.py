"""
    Estimation of horizontal tail chords and span
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

import math
import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


# TODO: is an OpenMDAO component required for this simple calculation ?
class ComputeHorizontalTailGeometry(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Horizontal tail geometry estimation """

    def setup(self):
        
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:volume_coefficient", val=np.nan)
        self.add_input("data:geometry:propulsion:layout", val=np.nan)
        self.add_input("data:geometry:has_T_tail", val=np.nan)
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")

        self.add_output("data:geometry:horizontal_tail:area", units="m**2")
        self.add_output("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", units="m")
        self.add_output("data:geometry:horizontal_tail:MAC:at25percent:x:local", units="m")
        self.add_output("data:geometry:horizontal_tail:MAC:y", units="m")
        self.add_output("data:geometry:horizontal_tail:sweep_25", units="deg")
        self.add_output("data:geometry:horizontal_tail:sweep_0", units="deg")
        self.add_output("data:geometry:horizontal_tail:sweep_100", units="deg")
        self.add_output("data:geometry:horizontal_tail:tip:chord", units="m")
        self.add_output("data:geometry:horizontal_tail:root:chord", units="m")
        self.add_output("data:geometry:horizontal_tail:MAC:length", units="m")
        self.add_output("data:geometry:horizontal_tail:wetted_area", units="m**2")
        self.add_output("data:geometry:horizontal_tail:aspect_ratio")
        self.add_output("data:geometry:horizontal_tail:height", units="m")
        
        self.declare_partials("", "", method="fd") 

    def compute(self, inputs, outputs):
        
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        fus_length = inputs["data:geometry:fuselage:length"]
        wing_area = inputs["data:geometry:wing:area"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        span = inputs["data:geometry:wing:span"]
        taper_ratio_ht = inputs["data:geometry:horizontal_tail:taper_ratio"]
        volume_coefficient_ht = inputs["data:geometry:horizontal_tail:volume_coefficient"]
        engine_loc = inputs["data:geometry:propulsion:layout"]
        tail_conf = inputs["data:geometry:has_T_tail"]
        span_vt = inputs["data:geometry:vertical_tail:span"]
        
        if engine_loc == 1.0:
            x_mac_ht = 0.5 * fus_length
        elif engine_loc == 3.0:
            x_mac_ht = 3*l0_wing
        else: # FIXME: no equation for configuration 2.0
            raise ValueError('compute_ht model only computes propulsion layout equal to 1 or 3!')
        
        if tail_conf == 1.0:
            sweep_25_ht = sweep_25 + 4.0
            lambda_ht = 5.9 * math.cos(sweep_25_ht / 180.0 * math.pi)**2
        else:
            sweep_25_ht = sweep_25 + 3.0
            lambda_ht = 5.5 * math.cos(sweep_25_ht / 180.0 * math.pi)**2
        
        area_ht = volume_coefficient_ht * wing_area * span / x_mac_ht 
        span_ht = math.sqrt(lambda_ht * area_ht)
        root_chord = area_ht * 2 / (1 + taper_ratio_ht) / span_ht
        tip_chord = root_chord * taper_ratio_ht
        temp = (root_chord * 0.25 + span_ht/2 \
                 * math.tan(sweep_25_ht * 180.0/math.pi) \
                 - tip_chord * 0.25)
        length_mac_ht = (root_chord**2 + root_chord * tip_chord + tip_chord**2) / \
                        (tip_chord + root_chord) * 2/3
        x0_ht = (temp * (root_chord + 2 * tip_chord))/(3 * (root_chord + tip_chord))
        y_mac_ht = (2 * span_ht/2 * (.5 * root_chord + tip_chord)) \
                    / (3 * (root_chord + tip_chord))
        
        if tail_conf == 1.0:
            wet_area_ht = 2 * 0.8 * area_ht * 1.05 #k_b coef from Gudmunnson p.707
            height_ht = 0 + span_vt
        else:
            wet_area_ht = 2 * area_ht * 1.05 #k_b coef from Gudmunnson p.707
            height_ht = 0 # Vertical distance between wing and HTP
        
        half_span = span_ht / 2.0
        sweep_0_ht = (math.pi / 2 - math.atan(half_span / (0.25 * root_chord - 0.25 \
                    * tip_chord + half_span * math.tan(sweep_25_ht / 180. * math.pi)))) \
                    * 180.0/ math.pi
        sweep_100_ht = (math.pi / 2 - math.atan(half_span / (half_span * math.tan(sweep_25_ht \
                        * math.pi/180.0 ) - 0.75 * root_chord + 0.75 * tip_chord))) \
                        * 180.0/ math.pi 
        
        outputs["data:geometry:horizontal_tail:area"] = area_ht
        outputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"] = x_mac_ht
        outputs["data:geometry:horizontal_tail:MAC:at25percent:x:local"] = x0_ht
        outputs["data:geometry:horizontal_tail:MAC:y"] = y_mac_ht
        outputs["data:geometry:horizontal_tail:sweep_25"] = sweep_25_ht
        outputs["data:geometry:horizontal_tail:sweep_0"] = sweep_0_ht
        outputs["data:geometry:horizontal_tail:sweep_100"] = sweep_100_ht
        outputs["data:geometry:horizontal_tail:tip:chord"] = tip_chord
        outputs["data:geometry:horizontal_tail:root:chord"] = root_chord
        outputs["data:geometry:horizontal_tail:MAC:length"] = length_mac_ht
        outputs["data:geometry:horizontal_tail:wetted_area"] = wet_area_ht
        outputs["data:geometry:horizontal_tail:aspect_ratio"] = lambda_ht
        outputs["data:geometry:horizontal_tail:height"] = height_ht # ???: does not appear in xml - used?
        
        