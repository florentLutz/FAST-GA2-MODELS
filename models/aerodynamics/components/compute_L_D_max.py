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
import math
from openmdao.core.explicitcomponent import ExplicitComponent


class ComputeLDMax(ExplicitComponent):
    """
    Computes optimal CL/CD aerodynamic performance of the aircraft in cruise conditions.

    """
    
    def setup(self):
        
        self.add_input("data:aerodynamics:aircraft:cruise:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:CL_alpha", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:cruise:induced_drag_coefficient", val=np.nan)

        self.add_output("data:aerodynamics:aircraft:cruise:L_D_max")
        self.add_output("data:aerodynamics:aircraft:cruise:optimal_CL")
        self.add_output("data:aerodynamics:aircraft:cruise:optimal_CD")
        self.add_output("data:aerodynamics:aircraft:cruise:optimal_alpha")
        
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        
        CL0_clean = inputs["data:aerodynamics:aircraft:cruise:CL0_clean"]
        CL_alpha = inputs["data:aerodynamics:aircraft:cruise:CL_alpha"]
        CD0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
        coef_k = inputs["data:aerodynamics:aircraft:cruise:induced_drag_coefficient"]
        
        Cl_opt = math.racine(CD0/coef_k)
        alpha_opt = (Cl_opt-CL0_clean)/CL_alpha
        Cd_opt = CD0 + coef_k * Cl_opt**2
        
        outputs["data:aerodynamics:aircraft:cruise:L_D_max"] = Cl_opt / Cd_opt
        outputs["data:aerodynamics:aircraft:cruise:optimal_CL"] = Cl_opt
        outputs["data:aerodynamics:aircraft:cruise:optimal_CD"] = Cd_opt
        outputs["data:aerodynamics:aircraft:cruise:optimal_alpha"] = alpha_opt
            
