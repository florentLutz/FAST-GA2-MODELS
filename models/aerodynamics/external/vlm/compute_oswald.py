"""
    Computation of Oswald coefficient using VLM
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

import logging

import numpy as np
import math
from fastoad.models.aerodynamics.constants import POLAR_POINT_COUNT
from fastoad.utils.physics import Atmosphere
from .vlm import VLM

_INPUT_AOAList = [14.0] # ???: why such angle choosen ?

_LOGGER = logging.getLogger(__name__)

class ComputeOSWALDvlm(VLM):
    """ Computes Oswald efficiency number """
    
    def initialize(self):
        self.options.declare("low_speed_aero", default=False, types=bool)
        
    def setup(self):
        
        super().setup()
        self.add_input("data:geometry:wing:area", val=np.nan, units='m**2')
        nans_array = np.full(POLAR_POINT_COUNT, np.nan)
        if self.options["low_speed_aero"]:
            self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
            self.add_input("data:aerodynamics:wing:low_speed:CL", val=nans_array)
            self.add_input("data:aerodynamics:wing:low_speed:CDp", val=nans_array)
            self.add_output("data:aerodynamics:aircraft:low_speed:induced_drag_coefficient")
        else:
            self.add_input("data:TLAR:v_cruise", val=np.nan, units='m/s')
            self.add_input("data:aerodynamics:cruise:mach", val=np.nan)
            self.add_input("data:aerodynamics:wing:cruise:CL", val=nans_array)
            self.add_input("data:aerodynamics:wing:cruise:CDp", val=nans_array)
            self.add_output("data:aerodynamics:aircraft:cruise:induced_drag_coefficient")
        
        
        self.declare_partials("*", "*", method="fd")        
    
    def compute(self, inputs, outputs):
        
        # Get inputs
        b_f = inputs['data:geometry:fuselage:maximum_width']
        span = inputs['data:geometry:wing:span']
        aspect_ratio = inputs['data:geometry:wing:aspect_ratio']
        wing_area = inputs['data:geometry:wing:area']
        if self.options["low_speed_aero"]:
            mach = inputs["data:aerodynamics:low_speed:mach"]
            V_inf = max(Atmosphere(0.0).speed_of_sound * mach, 0.01) # avoid V=0 m/s crashes
            CL_clean = inputs["data:aerodynamics:wing:low_speed:CL"]
            CDp_clean = inputs["data:aerodynamics:wing:low_speed:CDp"]
        else:
            mach = inputs["data:aerodynamics:cruise:mach"]
            V_inf = inputs["data:TLAR:v_cruise"]
            CL_clean = inputs["data:aerodynamics:wing:cruise:CL"]
            CDp_clean = inputs["data:aerodynamics:wing:cruise:CDp"]
        
        super()._run(inputs)
        Cl, Cdi, Oswald, _ = super().compute_wing(inputs, _INPUT_AOAList, V_inf, flaps_angle=0.0, use_airfoil=True)
        k_fus = 1 - 2*(b_f/span)**2
        oswald = Oswald[0] * k_fus #Fuselage correction
        if mach>0.4:
            oswald = oswald * (-0.001521 * ((mach - 0.05) / 0.3 - 1)**10.82 + 1) # Mach correction
        cdp_foil = self._interpolate_cdp(CL_clean, CDp_clean, Cl[0])
        cdi = (1.05*Cl[0])**2/(math.pi * aspect_ratio * oswald) + cdp_foil # Full aircraft correction: Wing lift is 105% of total lift.
        coef_e = Cl[0]**2/(math.pi * aspect_ratio * cdi)
        coef_k = 1. / (math.pi * span**2 / wing_area * coef_e)
        
        if self.options["low_speed_aero"]:
            outputs['data:aerodynamics:aircraft:low_speed:induced_drag_coefficient'] = coef_k
        else:
            outputs['data:aerodynamics:aircraft:cruise:induced_drag_coefficient'] = coef_k

    
    @staticmethod
    def _interpolate_cdp(lift_coeff: np.ndarray, drag_coeff: np.ndarray, ojective:float) -> float:
        """
        
        :param lift_coeff: CL array
        :param drag_coeff: CDp array
        :param ojective: CL_ref objective value
        :return: CD_ref if CL_ref encountered, or default value otherwise
        """
        # Reduce vectors for interpolation
        for idx in range(len(lift_coeff)):
            if np.sum(lift_coeff[idx:len(lift_coeff)]==0)==(len(lift_coeff)-idx):
                lift_coeff = lift_coeff[0:idx]
                drag_coeff = drag_coeff[0:idx]
                break

        # Interpolate value if within the interpolation range
        if ojective >= min(lift_coeff) and ojective <= max(lift_coeff):
            idx_max = np.where(lift_coeff == max(lift_coeff))
            return np.interp(ojective, lift_coeff[0:idx_max+1], drag_coeff[0:idx_max+1])
        elif ojective < lift_coeff[0]:
            cdp = drag_coeff[0] + (ojective - lift_coeff[0]) * (drag_coeff[1] - drag_coeff[0]) \
                  / (lift_coeff[1] - lift_coeff[0])
        elif ojective > lift_coeff[-1]:
            cdp = drag_coeff[-1] + (ojective - lift_coeff[-1]) * (drag_coeff[-1] - drag_coeff[-2]) \
                  / (lift_coeff[-1] - lift_coeff[-2])
        _LOGGER.warning("CL not in range. Linear extrapolation of CDp value (%s)", cdp)
        return cdp