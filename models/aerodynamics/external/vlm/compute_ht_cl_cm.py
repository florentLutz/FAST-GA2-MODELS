"""
    Estimation of HTP low-speed lift and induced moment using VLM
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
from ...constants import HT_POINT_COUNT
from fastoad.utils.physics import Atmosphere
from .vlm import VLM

ALPHA_LIMIT = 30.0  # Limit angle for calculation
_INPUT_AOAList = list(np.linspace(0.0, ALPHA_LIMIT, HT_POINT_COUNT))


class ComputeHTPCLCMvlm(VLM):

    def initialize(self):
        super().initialize()
        
    def setup(self):

        super().setup()
        self.add_input("data:geometry:wing:area", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan)
        self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
        
        self.add_output("data:aerodynamics:horizontal_tail:low_speed:alpha", shape=len(_INPUT_AOAList), units="deg")
        self.add_output("data:aerodynamics:horizontal_tail:low_speed:CL", shape=len(_INPUT_AOAList))
        self.add_output("data:aerodynamics:horizontal_tail:low_speed:CM", shape=len(_INPUT_AOAList))
        self.add_output("data:aerodynamics:wing:low_speed:alpha", shape=len(_INPUT_AOAList), units="deg")
        self.add_output("data:aerodynamics:wing:low_speed:CM", shape=len(_INPUT_AOAList))

        self.declare_partials("*", "*", method="fd")
    
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # Get inputs
        wing_area = inputs['data:geometry:wing:area']
        htp_area = inputs['data:geometry:horizontal_tail:area']
        aspect_ratio = inputs['data:geometry:wing:aspect_ratio']
        mach = inputs["data:aerodynamics:low_speed:mach"]
        v_inf = max(Atmosphere(0.0).speed_of_sound * mach, 0.01)  # avoid V=0 m/s crashes

        super()._run(inputs)
        beta = math.sqrt(1 - mach ** 2)  # Prandtl-Glauert
        cl_wing, _, _, result_cm2 = super().compute_wing(inputs, _INPUT_AOAList, v_inf, flaps_angle=0.0,
                                                         use_airfoil=True)
        result_cm2 = np.array(result_cm2)/beta
        # Calculate downwash angle based on Gudmundsson model (p.467)
        downwash_angle = 2.0 * np.array(cl_wing)/beta * 180.0 / (aspect_ratio * np.pi**2)
        HTP_AOAList = list(np.array(_INPUT_AOAList) - downwash_angle)
        result_cl, result_cm1 = super().compute_htp(inputs, HTP_AOAList, v_inf, use_airfoil=True)
        # Write value with wing Sref
        result_cl = np.array(result_cl)/beta * htp_area / wing_area
        result_cm1 = np.array(result_cm1)/beta * htp_area / wing_area

        outputs['data:aerodynamics:horizontal_tail:low_speed:alpha'] = np.array(_INPUT_AOAList)
        outputs['data:aerodynamics:horizontal_tail:low_speed:CL'] = result_cl
        outputs['data:aerodynamics:horizontal_tail:low_speed:CM'] = result_cm1
        outputs['data:aerodynamics:wing:low_speed:alpha'] = np.array(_INPUT_AOAList)
        outputs['data:aerodynamics:wing:low_speed:CM'] = result_cm2
