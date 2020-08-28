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
import openmdao.api as om
from openmdao.core.explicitcomponent import ExplicitComponent


        
class ComputeMaxCL(om.Group):
    """
    Computes maximum CL of the aircraft in landing/take-off conditions.

    3D CL is deduced from 2D CL using sweep angle.
    Contribution of high-lift devices is done appart and added.

    """
    def setup(self):
        self.add_subsystem("CL_2D_to_3D", Compute3DMaxCL(), promotes=["*"])
        self.add_subsystem("comp_cl_max", ComputeAircraftMaxCl(), promotes=["*"])
    
    
class Compute3DMaxCL(ExplicitComponent):
    """
    Computes wing 3D max CL from 2D CL (XFOIL-computed) and sweep angle
    """
    
    def setup(self):
        
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean_2D", val=np.nan)

        self.add_output("data:aerodynamics:wing:low_speed:CL_max_clean")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        sweep_25 = inputs["data:geometry:wing:sweep_25"]
        cl_max_2d = inputs["data:aerodynamics:wing:low_speed:CL_max_clean_2D"]
        
        CL_max_clean = cl_max_2d * 0.9 * np.cos(sweep_25)
        
        outputs["data:aerodynamics:wing:low_speed:CL_max_clean"] = CL_max_clean
        
        
class ComputeAircraftMaxCl(ExplicitComponent):
    """
    Add high-lift contribution (flaps)
    """
    
    def setup(self):
        
        self.add_input("data:aerodynamics:aircraft:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:flaps:takeoff:CL", val=np.nan)
        self.add_input("data:aerodynamics:flaps:landing:CL", val=np.nan)

        self.add_output("data:aerodynamics:aircraft:landing:CL_max")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        
        cl_max_clean = inputs["data:aerodynamics:aircraft:low_speed:CL_max_clean"]
        cl_max_takeoff = cl_max_clean + inputs["data:aerodynamics:flaps:takeoff:CL"]
        cl_max_landing = cl_max_clean + inputs["data:aerodynamics:flaps:landing:CL"]

        outputs["data:aerodynamics:aircraft:takeoff:CL_max"] = cl_max_takeoff
        outputs["data:aerodynamics:aircraft:landing:CL_max"] = cl_max_landing
