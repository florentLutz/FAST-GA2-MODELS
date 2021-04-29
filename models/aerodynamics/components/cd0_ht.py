"""
    FAST - Copyright (c) 2016 ONERA ISAE
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
from models.geometry.profiles.get_profile import get_profile


class Cd0HorizontalTail(ExplicitComponent):
    
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)
        self.options.declare('htp_airfoil_file', default="naca0012.af", types=str, allow_none=True)

    def setup(self):

        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:horizontal_tail:wet_area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:thickness_ratio", val=np.nan)
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
            self.add_input("data:aerodynamics:low_speed:unit_reynolds", val=np.nan, units="m**-1")
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:CD0")
        else:
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:aerodynamics:cruise:unit_reynolds", val=np.nan, units="m**-1")
            self.add_output("data:aerodynamics:horizontal_tail:cruise:CD0")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        sweep_25_ht = inputs["data:geometry:horizontal_tail:sweep_25"]
        wet_area_ht = inputs["data:geometry:horizontal_tail:wet_area"]
        wing_area = inputs["data:geometry:wing:area"]
        thickness = inputs["data:geometry:horizontal_tail:thickness_ratio"]
        if self.options["low_speed_aero"]:
            mach = inputs["data:aerodynamics:low_speed:mach"]
            unit_reynolds = inputs["data:aerodynamics:low_speed:unit_reynolds"]
        else:
            mach = inputs["data:aerodynamics:cruise:mach"]
            unit_reynolds = inputs["data:aerodynamics:cruise:unit_reynolds"]

        # Sear max thickness position ratio
        profile = get_profile(file_name=self.options['htp_airfoil_file'])
        relative_thickness = profile.get_relative_thickness()
        index = int(np.where(relative_thickness['thickness'] == np.max(relative_thickness['thickness']))[0])
        x_tmax = relative_thickness['x'][index]
        # Root: 50% NLF
        x_trans = 0.5
        x0_turb = 36.9 * x_trans**0.625 * (1/(unit_reynolds*root_chord))**0.375
        cf_root = 0.074 / (unit_reynolds*root_chord)**0.2 * (1 - (x_trans - x0_turb))**0.8
        # Tip: 50% NLF
        x_trans = 0.5
        x0_turb = 36.9 * x_trans**0.625 * (1/(unit_reynolds*tip_chord))**0.375
        cf_tip = 0.074 / (unit_reynolds*tip_chord)**0.2 * (1 - (x_trans - x0_turb))**0.8
        # Global
        cf_ht = (cf_root + cf_tip) * 0.5
        ff = 1 + 0.6/x_tmax * thickness + 100 * thickness**4
        ff = ff*1.05  # Due to hinged elevator (Raymer)
        if mach > 0.2:
            ff = ff * 1.34 * mach**0.18 * (math.cos(sweep_25_ht*math.pi/180))**0.28
        interf = 1.05
        cd0 = ff*interf*cf_ht * wet_area_ht / wing_area

        if self.options["low_speed_aero"]:
            outputs["data:aerodynamics:horizontal_tail:low_speed:CD0"] = cd0
        else:
            outputs["data:aerodynamics:horizontal_tail:cruise:CD0"] = cd0
