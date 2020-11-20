"""
    Estimation of HTP lift coefficient using OPENVSP
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
from fastoad.utils.physics import Atmosphere
from .vlm import VLM

_INPUT_AOAList = [0.0, 4.0]


class ComputeHTPCLALPHAvlm(VLM):

    def initialize(self):
        super().initialize()
        self.options.declare("low_speed_aero", default=False, types=bool)
        
    def setup(self):
        
        super().setup()
        self.add_input("data:geometry:wing:area", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan)
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
            self.add_output("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", units="rad**-1")
        else:
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units='ft')
            self.add_output("data:aerodynamics:horizontal_tail:cruise:CL_alpha", units="rad**-1")
        
        self.declare_partials("*", "*", method="fd")
    
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # Get inputs
        wing_area = inputs['data:geometry:wing:area']
        htp_area = inputs['data:geometry:horizontal_tail:area']
        aspect_ratio = inputs['data:geometry:wing:aspect_ratio']
        if self.options["low_speed_aero"]:
            altitude = 0.0
            atm = Atmosphere(altitude)
            mach = inputs["data:aerodynamics:low_speed:mach"]
        else:
            altitude = inputs["data:mission:sizing:main_route:cruise:altitude"]
            atm = Atmosphere(altitude)
            mach = inputs["data:aerodynamics:cruise:mach"]
        v_inf = max(atm.speed_of_sound * mach, 0.01)  # avoid V=0 m/s crashes

        super()._run(inputs)
        beta = math.sqrt(1 - mach ** 2)  # Prandtl-Glauert
        cl_wing, _, _, _ = super().compute_wing(inputs, _INPUT_AOAList, v_inf, flaps_angle=0.0,
                                                         use_airfoil=True)
        # Calculate downwash angle based on Gudmundsson model (p.467)
        downwash_angle = 2.0 * np.array(cl_wing)/beta * 180.0 / (aspect_ratio * np.pi**2)
        HTP_AOAList = list(np.array(_INPUT_AOAList) - downwash_angle)
        result_cl, _ = super().compute_htp(inputs, HTP_AOAList, v_inf, use_airfoil=True)
        # Write value with wing Sref
        result_cl = np.array(result_cl)/beta * htp_area/wing_area
        # Calculate derivative
        cl_alpha = float((result_cl[1] - result_cl[0]) / ((_INPUT_AOAList[1] - _INPUT_AOAList[0]) * math.pi / 180))


        if self.options["low_speed_aero"]:
            outputs['data:aerodynamics:horizontal_tail:low_speed:CL_alpha'] = cl_alpha
        else:
            outputs['data:aerodynamics:horizontal_tail:cruise:CL_alpha'] = cl_alpha
