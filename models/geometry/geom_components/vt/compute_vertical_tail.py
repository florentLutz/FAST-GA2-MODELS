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
from fastoad.utils.physics import kinematic_viscosity


# TODO: is an OpenMDAO component required for this simple calculation ?
class ComputeVerticalTailGeometry(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Vertical tail geometry estimation """

    def setup(self):
        
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("ddata:mission:sizing:cruise:altitude", val=np.nan, units="ft")
        self.add_input("data:TLAR:cruise_speed", val=np.nan, units="kn")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:volume_coefficient", val=np.nan)
        self.add_input("data:geometry:propulsion:layout", val=np.nan)
        self.add_input("data:geometry:has_T_tail", val=np.nan)

        self.add_output("data:geometry:vertical_tail:area", units="m**2")
        self.add_output("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", units="m")
        self.add_output("data:geometry:vertical_tail:MAC:at25percent:x:local", units="m")
        self.add_output("data:geometry:vertical_tail:MAC:z", units="m")
        self.add_output("data:geometry:vertical_tail:sweep_25", units="deg")
        self.add_output("data:geometry:vertical_tail:sweep_0", units="deg")
        self.add_output("data:geometry:vertical_tail:sweep_100", units="deg")
        self.add_output("data:geometry:vertical_tail:tip:chord", units="m")
        self.add_output("data:geometry:vertical_tail:root:chord", units="m")
        self.add_output("data:geometry:vertical_tail:MAC:length", units="m")
        self.add_output("data:aerodynamics:vertical_tail:cruise:CL_alpha")
        self.add_output("data:geometry:vertical_tail:wetted_area", units="m**2")
        self.add_output("data:geometry:vertical_tail:aspect_ratio")
        
        self.declare_partials("", "", method="fd") 

    def compute(self, inputs, outputs):
        
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        cruise_speed = inputs["data:TLAR:cruise_speed"]
        Mach = cruise_speed*0.514444 / kinematic_viscosity(inputs["data:mission:sizing:cruise:altitude"]) 
        fus_length = inputs["data:geometry:fuselage:length"]
        wing_area = inputs["data:geometry:wing:area"]
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        span = inputs["data:geometry:wing:span"]
        taper_ratio_vt = inputs["data:geometry:vertical_tail:taper_ratio"]
        volume_coefficient_vt = inputs["data:geometry:vertical_tail:volume_coefficient"]
        engine_loc = inputs["data:geometry:propulsion:layout"]
        tail_conf = inputs["data:geometry:has_T_tail"]
        
        if engine_loc == 1.0:
            x_mac_vt = 0.5 * fus_length
        elif engine_loc == 3.0:
            x_mac_vt = 3*l0_wing
        else: # FIXME: no equation for configuration 2.0
            raise ValueError('compute_vt model only computes propulsion layout equal to 1 or 3!')
        
        if tail_conf == 1.0:
            sweep_25_vt = sweep_25 + 10.0
            lambda_vt = 1.2
        else:
            sweep_25_vt = sweep_25 + 10.0
            lambda_vt = 2.6 * math.cos(sweep_25_vt * math.pi/180.0)**2
        
        area_vt = volume_coefficient_vt * wing_area * span / x_mac_vt 
        span_vt = math.sqrt(lambda_vt * area_vt)
        root_chord = area_vt * 2 / (1 + taper_ratio_vt) / span_vt
        tip_chord = root_chord * taper_ratio_vt
        temp = (root_chord * 0.25 + span_vt \
                 * math.tan(sweep_25_vt * 180.0/math.pi) \
                 - tip_chord * 0.25)
        length_mac_vt = (root_chord**2 + root_chord * tip_chord + tip_chord**2) / \
                        (tip_chord + root_chord) * 2/3
        x0_vt = (temp * (root_chord + 2 * tip_chord))/(3 * (root_chord + tip_chord))
        z_mac_vt = (2 * span_vt * (.5 * root_chord + tip_chord)) \
                    / (3 * (root_chord + tip_chord))
        
        beta = math.sqrt(1 - Mach**2)
        if tail_conf == 1.0:
            k_ar_effective = 2.9
        else:
            k_ar_effective = 1.55
        
        lambda_vt *= k_ar_effective
        cl_alpha_vt = 0.8 * 2 * math.pi * lambda_vt / \
            (2 + math.sqrt(4 + lambda_vt**2 * beta**2 / 0.95 **
                           2 * (1 + (math.tan(sweep_25_vt / 180. * math.pi))**2 / beta**2)))
        wet_area_vt = 2.1 * area_vt
        sweep_0_vt = (math.pi / 2 - math.atan(span_vt / (0.25 * root_chord - 0.25 \
                    * tip_chord + span_vt * math.tan(sweep_25_vt / 180. * math.pi)))) \
                    * 180.0/ math.pi
        sweep_100_vt = (math.pi / 2 - math.atan(span_vt / (span_vt * math.tan(sweep_25_vt \
                        * math.pi/180.0 ) - 0.75 * root_chord + 0.75 * tip_chord))) \
                        * 180.0/ math.pi 
        
        outputs["data:geometry:vertical_tail:area"] = area_vt
        outputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"] = x_mac_vt
        outputs["data:geometry:vertical_tail:MAC:at25percent:x:local"] = x0_vt
        outputs["data:geometry:vertical_tail:MAC:z"] = z_mac_vt
        outputs["data:geometry:vertical_tail:sweep_25"] = sweep_25_vt
        outputs["data:geometry:vertical_tail:sweep_0"] = sweep_0_vt
        outputs["data:geometry:vertical_tail:sweep_100"] = sweep_100_vt
        outputs["data:geometry:vertical_tail:tip:chord"] = tip_chord
        outputs["data:geometry:vertical_tail:root:chord"] = root_chord
        outputs["data:geometry:vertical_tail:MAC:length"] = length_mac_vt
        outputs["data:aerodynamics:vertical_tail:cruise:CL_alpha"] = cl_alpha_vt
        outputs["data:geometry:vertical_tail:wetted_area"] = wet_area_vt
        outputs["data:geometry:vertical_tail:aspect_ratio"] = lambda_vt / k_ar_effective
        
        