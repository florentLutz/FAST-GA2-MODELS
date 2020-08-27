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
from openmdao.core.explicitcomponent import ExplicitComponent
import openmdao.api as om

from high_lift_aero import ComputeDeltaHighLift
from ..external.xfoil import XfoilPolar
from ..external.xfoil.xfoil_polar import OPTION_XFOIL_EXE_PATH
from fastoad.models.options import OpenMdaoOptionDispatcherGroup
from fastoad.utils.physics import Atmosphere


class ComputeMaxCL(OpenMdaoOptionDispatcherGroup):
    """
    Computes maximum CL of the aircraft in landing/take-off conditions.

    Maximum 2D CL without high-lift is computed using XFoil (or provided as input if option
    use_xfoil is set to False). 3D CL is deduced using sweep angle.

    Contribution of high-lift devices is modelled according to their geometry (span and chord ratio)
    and their deflection angles.

    Options:
      - use_xfoil:
         - if True, maximum 2D CL without high-lift aerodynamics:aircraft:landing:CL_max_clean_2D
           is computed using XFOIL
         - if False, aerodynamics:aircraft:landing:CL_max_clean_2D must be provided as input (but
           process is faster)
      - alpha_min, alpha_max:
         - used if use_xfoil is True. Sets the alpha range that is explored to find maximum 2D CL
           without high-lift
      - xfoil_exe_path:
         - the path to the XFOIL executable. Needed for non-Windows OS.
    """

    def initialize(self):
        
        self.options.declare("use_xfoil", default=True, types=bool)
        self.options.declare("xfoil_alpha_min", default=0.0, types=float)
        self.options.declare("xfoil_alpha_max", default=30.0, types=float)
        self.options.declare("xfoil_iter_limit", default=500, types=int)
        self.options.declare(OPTION_XFOIL_EXE_PATH, default="", types=str, allow_none=True)

    def setup(self):
        
        self.add_subsystem("mach_reynolds", ComputeMachReynolds(), promotes=["*"])
        if self.options["use_xfoil"]:
            start = self.options["xfoil_alpha_min"]
            end = self.options["xfoil_alpha_max"]
            iter_limit = self.options["xfoil_iter_limit"]
            self.add_subsystem(
                "xfoil_run",
                XfoilPolar(alpha_start=start, alpha_end=end, iter_limit=iter_limit),
                promotes=["data:geometry:wing:thickness_ratio"],
            )
        self.add_subsystem("CL_2D_to_3D", Compute3DMaxCL(), promotes=["*"])
        self.add_subsystem(
            "delta_cl_landing", ComputeDeltaHighLift(landing_flag=True), promotes=["*"]
        )
        self.add_subsystem("compute_max_cl", ComputeAircraftMaxCl(), promotes=["*"])

        if self.options["use_xfoil"]:
            ivc = om.IndepVarComp()
            ivc.add_output("data:aerodynamics:wing:low_speed:alpha")
            ivc.add_output("data:aerodynamics:wing:low_speed:CL")
            ivc.add_output("data:aerodynamics:wing:low_speed:CD")
            ivc.add_output("data:aerodynamics:wing:low_speed:CDp")
            ivc.add_output("data:aerodynamics:wing:low_speed:CM")
            self.add_subsystem("low_speed_arrays", ivc, promotes=["*"])
            self.connect("data:aerodynamics:wing:low_speed:alpha", "xfoil_run.xfoil:alpha")
            self.connect("data:aerodynamics:wing:low_speed:CL", "xfoil_run.xfoil:CL")
            self.connect("data:aerodynamics:wing:low_speed:CD", "xfoil_run.xfoil:CD")
            self.connect("data:aerodynamics:wing:low_speed:CDp", "xfoil_run.xfoil:CDp")
            self.connect("data:aerodynamics:wing:low_speed:CM", "xfoil_run.xfoil:CM")
            self.connect("data:aerodynamics:aircraft:low_speed:mach", "xfoil_run.xfoil:mach")
            self.connect(
                "data:aerodynamics:wing:low_speed:reynolds", "xfoil_run.xfoil:reynolds")
            self.connect(
                "xfoil_run.xfoil:CL_max_2D", "data:aerodynamics:wing:low_speed:CL_max_clean_2D"
            )


class ComputeMachReynolds(om.ExplicitComponent):
    """
    Mach and Reynolds computation
    """

    def setup(self):
        
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")
        
        self.add_output("data:aerodynamics:aircraft:low_speed:mach")
        self.add_output("data:aerodynamics:wing:low_speed:reynolds")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        speed = inputs["data:TLAR:v_approach"]

        atm = Atmosphere(0.0, 15.0)
        mach = speed / atm.speed_of_sound
        reynolds = atm.get_unitary_reynolds(mach) * l0_wing

        outputs["data:aerodynamics:aircraft:low_speed:mach"] = mach
        outputs["data:aerodynamics:wing:low_speed:reynolds"] = reynolds


class Compute3DMaxCL(om.ExplicitComponent):
    """
    Computes 3D max CL from 2D CL (XFOIL-computed) and sweep angle
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
    def setup(self):
        
        self.add_input("data:aerodynamics:aircraft:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:high_lift_devices:takeoff:CL", val=np.nan)
        self.add_input("data:aerodynamics:high_lift_devices:landing:CL", val=np.nan)

        self.add_output("data:aerodynamics:aircraft:landing:CL_max")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        
        cl_max_clean = inputs["data:aerodynamics:aircraft:low_speed:CL_max_clean"]
        cl_max_takeoff = cl_max_clean + inputs["data:aerodynamics:high_lift_devices:takeoff:CL"]
        cl_max_landing = cl_max_clean + inputs["data:aerodynamics:high_lift_devices:landing:CL"]

        outputs["data:aerodynamics:aircraft:takeoff:CL_max"] = cl_max_takeoff
        outputs["data:aerodynamics:aircraft:landing:CL_max"] = cl_max_landing
