"""
    Estimation of lift coefficient using VLM
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
import warnings

from fastoad.utils.physics import Atmosphere
from .vlm import VLM
from ...constants import SPAN_MESH_POINT_OPENVSP

_INPUT_AOAList = [0.0, 7.0]


class ComputeWingCLALPHAvlm(VLM):
    """ Computes lift coefficient """
    
    def initialize(self):
        super().initialize()
        self.options.declare("low_speed_aero", default=False, types=bool)
        
    def setup(self):
        
        super().setup()
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
            self.add_output("data:aerodynamics:aircraft:low_speed:CL0_clean")
            self.add_output("data:aerodynamics:aircraft:low_speed:CL_alpha", units="rad**-1")
            self.add_output("data:aerodynamics:wing:low_speed:Y_vector", shape=SPAN_MESH_POINT_OPENVSP, units="m")
            self.add_output("data:aerodynamics:wing:low_speed:CL_vector", shape=SPAN_MESH_POINT_OPENVSP)
        else:
            self.add_input("data:TLAR:v_cruise", val=np.nan, units='m/s')
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_output("data:aerodynamics:aircraft:cruise:CL0_clean")
            self.add_output("data:aerodynamics:aircraft:cruise:CL_alpha", units="rad**-1")
        
        self.declare_partials("*", "*", method="fd")        
    
    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        # Get inputs
        b_f = inputs['data:geometry:fuselage:maximum_width']
        span = inputs['data:geometry:wing:span']
        if self.options["low_speed_aero"]:
            mach = inputs["data:aerodynamics:low_speed:mach"]
            V_inf = max(Atmosphere(0.0).speed_of_sound * mach, 0.01)  # avoid V=0 m/s crashes
        else:
            mach = inputs["data:aerodynamics:cruise:mach"]
            V_inf = inputs["data:TLAR:v_cruise"]
        
        super()._run(inputs)
        Cl, Cdi, Oswald, Cm = super().compute_wing(inputs, _INPUT_AOAList, V_inf, flaps_angle=0.0, use_airfoil=True)
        k_fus = 1 + 0.025*b_f/span - 0.025*(b_f/span)**2  # Fuselage correction
        beta = math.sqrt(1 - mach**2)  # Prandtl-Glauert
        cl_alpha = (Cl[1] - Cl[0]) / ((_INPUT_AOAList[1]-_INPUT_AOAList[0])*math.pi/180) * k_fus / beta
        cl_0 = Cl[0] / beta
        y_vector, cl_vector = super().get_cl_curve(_INPUT_AOAList[0], V_inf)
        cl_vector = list(np.array(cl_vector) / beta)
        real_length = min(SPAN_MESH_POINT_OPENVSP, len(y_vector))
        if real_length < len(y_vector):
            warnings.warn("Defined maximum span mesh in constants.py exceeded!")

        if self.options["low_speed_aero"]:
            outputs['data:aerodynamics:aircraft:low_speed:CL0_clean'] = cl_0
            outputs['data:aerodynamics:aircraft:low_speed:CL_alpha'] = cl_alpha
            if real_length >= len(y_vector):
                outputs['data:aerodynamics:wing:low_speed:Y_vector'] = np.zeros(SPAN_MESH_POINT_OPENVSP)
                outputs['data:aerodynamics:wing:low_speed:CL_vector'] = np.zeros(SPAN_MESH_POINT_OPENVSP)
                outputs['data:aerodynamics:wing:low_speed:Y_vector'][0:real_length] = y_vector
                outputs['data:aerodynamics:wing:low_speed:CL_vector'][0:real_length] = cl_vector
            else:
                outputs['data:aerodynamics:aircraft:wing:Y_vector'] = np.linspace(y_vector[0], y_vector[1],
                                                                                  SPAN_MESH_POINT_OPENVSP)
                outputs['data:aerodynamics:aircraft:wing:CL_vector'] = \
                    np.interp(outputs['data:aerodynamics:aircraft:low_speed:Y_vector'], y_vector, cl_vector)
        else:
            outputs['data:aerodynamics:aircraft:cruise:CL0_clean'] = cl_0
            outputs['data:aerodynamics:aircraft:cruise:CL_alpha'] = cl_alpha
