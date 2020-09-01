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


import numpy as np
from fastoad.utils.physics import Atmosphere
from openmdao.core.explicitcomponent import ExplicitComponent


class ComputeReynolds(ExplicitComponent):
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)

    def setup(self):
        self.low_speed_aero = self.options["low_speed_aero"]

        if self.low_speed_aero:
            self.add_input("data:aerodynamics:low_speed:mach", val=0.2)
            self.add_output("data:aerodynamics:wing:low_speed:reynolds")
        else:
            self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
            self.add_input("data:mission:sizing:cruise:altitude", val=np.nan, units="ft")
            self.add_output("data:aerodynamics:cruise:mach")
            self.add_output("data:aerodynamics:wing:cruise:reynolds")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        if self.low_speed_aero:
            altitude = 0.0
            mach = inputs["data:aerodynamics:low_speed:mach"]
        else:
            altitude = inputs["data:mission:sizing:cruise:altitude"]
            mach = inputs["data:TLAR:v_cruise"]/Atmosphere(altitude).speed_of_sound
            
        reynolds = Atmosphere(altitude, altitude_in_feet=False).get_unitary_reynolds(mach)

        if self.low_speed_aero:
            outputs["data:aerodynamics:low_speed:mach"] = mach
            outputs["data:aerodynamics:wing:low_speed:reynolds"] = reynolds
        else:
            outputs["data:aerodynamics:cruise:mach"] = mach
            outputs["data:aerodynamics:wing:cruise:reynolds"] = reynolds
