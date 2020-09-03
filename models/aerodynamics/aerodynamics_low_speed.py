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

from .components.cd0_fuselage import Cd0Fuselage
from .components.cd0_ht import Cd0HorizontalTail
from .components.cd0_lg import Cd0LandingGear
from .components.cd0_nacelle import Cd0Nacelle
from .components.cd0_total import Cd0Total
from .components.cd0_vt import Cd0VerticalTail
from .components.cd0_wing import Cd0Wing
from .components.cd0_other import Cd0Other
from .components.compute_cl_max import ComputeMaxCL
from .components.compute_reynolds import ComputeReynolds
from .components.high_lift_aero import ComputeDeltaHighLift

from .external.vlm import ComputeOSWALDvlm, ComputeWingCLALPHAvlm
from .external.openvsp import ComputeOSWALDopenvsp, ComputeWingCLALPHAopenvsp, ComputeHTPCLCMopenvsp, ComputeHTPCLALPHAopenvsp
from .external.xfoil import XfoilPolar

from openmdao.core.group import Group
from openmdao.core.indepvarcomp import IndepVarComp

_OSWALD_BY_VLM = True
_CLALPHA_BY_VLM = True

class AerodynamicsLowSpeed(Group):
    """
    Models for low speed aerodynamics
    """

    def setup(self):
        ivc = IndepVarComp("data:aerodynamics:low_speed:mach", val=0.2)
        self.add_subsystem("mach_low_speed", ivc, promotes=["*"])
        if _OSWALD_BY_VLM:
            self.add_subsystem("oswald", ComputeOSWALDvlm(low_speed_aero=True), promotes=["*"])
        else:
            self.add_subsystem("oswald", ComputeOSWALDopenvsp(low_speed_aero=True), promotes=["*"])
        if _CLALPHA_BY_VLM:
            self.add_subsystem("cl_alpha", ComputeWingCLALPHAvlm(low_speed_aero=True), promotes=["*"])
        else:
            self.add_subsystem("cl_alpha", ComputeWingCLALPHAopenvsp(low_speed_aero=True), promotes=["*"])
        self.add_subsystem("comp_re", ComputeReynolds(low_speed_aero=True), promotes=["*"])
        self.add_subsystem("cd0_wing", Cd0Wing(low_speed_aero=True), promotes=["*"])
        self.add_subsystem("cd0_fuselage", Cd0Fuselage(low_speed_aero=True), promotes=["*"])
        self.add_subsystem("cd0_ht", Cd0HorizontalTail(low_speed_aero=True), promotes=["*"])
        self.add_subsystem("cd0_vt", Cd0VerticalTail(low_speed_aero=True), promotes=["*"])
        self.add_subsystem("cd0_nacelle", Cd0Nacelle(low_speed_aero=True), promotes=["*"])
        self.add_subsystem("cd0_l_gear", Cd0LandingGear(low_speed_aero=True), promotes=["*"])
        self.add_subsystem("cd0_other", Cd0Other(low_speed_aero=True), promotes=["*"])
        self.add_subsystem("cd0_total", Cd0Total(low_speed_aero=True), promotes=["*"])
        self.add_subsystem("comp_polar", XfoilPolar(low_speed_aero=True), promotes=["*"])
        self.add_subsystem("cl_max", ComputeMaxCL(), promotes=["*"])
        self.add_subsystem("cl_cm_ht", ComputeHTPCLCMopenvsp(), promotes=["*"])
        self.add_subsystem("cl_alpha_ht", ComputeHTPCLALPHAopenvsp(low_speed_aero=True), promotes=["*"])
        self.add_subsystem("high_lift", ComputeDeltaHighLift(), promotes=["*"])
