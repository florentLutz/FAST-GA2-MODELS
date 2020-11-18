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

from .external.vlm import ComputeOSWALDvlm, ComputeWingCLALPHAvlm, ComputeHTPCLCMvlm, ComputeHTPCLALPHAvlm
from .external.openvsp import ComputeOSWALDopenvsp, ComputeWingCLALPHAopenvsp, ComputeHTPCLCMopenvsp, \
    ComputeHTPCLALPHAopenvsp
from .external.xfoil import XfoilPolar

from openmdao.core.group import Group
from openmdao.core.explicitcomponent import ExplicitComponent
import numpy as np


class AerodynamicsLowSpeed(Group):
    """
    Models for low speed aerodynamics
    """

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)
        self.options.declare("use_openvsp", default=False, types=bool)
        self.options.declare('wing_airfoil_file', default="naca23012.af", types=str, allow_none=True)
        self.options.declare('htp_airfoil_file', default="naca0012.af", types=str, allow_none=True)

    def setup(self):
        self.add_subsystem("comp_re", ComputeReynolds(low_speed_aero=True), promotes=["*"])
        self.add_subsystem(
            "xfoil_in",
            Connection(),
            promotes=[
                "data:geometry:wing:MAC:length",
                "data:geometry:wing:root:chord",
                "data:geometry:wing:tip:chord",
            ]
        )
        self.add_subsystem("comp_polar2", XfoilPolar(), promotes=["data:geometry:wing:thickness_ratio"])
        self.add_subsystem("comp_polar3", XfoilPolar(), promotes=["data:geometry:wing:thickness_ratio"])
        if not (self.options["use_openvsp"]):
            self.add_subsystem(
                "comp_polar1",
                XfoilPolar(wing_airfoil_file=self.options["wing_airfoil_file"]),
                promotes=["data:geometry:wing:thickness_ratio"])
            self.add_subsystem(
                "oswald",
                ComputeOSWALDvlm(low_speed_aero=True, wing_airfoil_file=self.options["wing_airfoil_file"]),
                promotes=["*"])
            self.add_subsystem(
                "cl_alpha",
                ComputeWingCLALPHAvlm(low_speed_aero=True, wing_airfoil_file=self.options["wing_airfoil_file"]),
                promotes=["*"])
        else:
            self.add_subsystem(
                "oswald",
                ComputeOSWALDopenvsp(low_speed_aero=True, wing_airfoil_file=self.options["wing_airfoil_file"]),
                promotes=["*"])
            self.add_subsystem(
                "cl_alpha",
                ComputeWingCLALPHAopenvsp(low_speed_aero=True, wing_airfoil_file=self.options["wing_airfoil_file"]),
                promotes=["*"])
        self.add_subsystem("cd0_wing", Cd0Wing(low_speed_aero=True), promotes=["*"])
        self.add_subsystem("cd0_fuselage", Cd0Fuselage(low_speed_aero=True), promotes=["*"])
        self.add_subsystem("cd0_ht", Cd0HorizontalTail(low_speed_aero=True), promotes=["*"])
        self.add_subsystem("cd0_vt", Cd0VerticalTail(low_speed_aero=True), promotes=["*"])
        self.add_subsystem("cd0_nacelle", Cd0Nacelle(propulsion_id=self.options["propulsion_id"],
                                                     low_speed_aero=True), promotes=["*"])
        self.add_subsystem("cd0_l_gear", Cd0LandingGear(low_speed_aero=True), promotes=["*"])
        self.add_subsystem("cd0_other", Cd0Other(low_speed_aero=True), promotes=["*"])
        self.add_subsystem("cd0_total", Cd0Total(low_speed_aero=True), promotes=["*"])
        self.add_subsystem("high_lift", ComputeDeltaHighLift(), promotes=["*"])
        if not (self.options["use_openvsp"]):
            self.add_subsystem(
                "cl_cm_ht",
                ComputeHTPCLCMvlm(
                    wing_airfoil_file=self.options["wing_airfoil_file"],
                    htp_airfoil_file=self.options["htp_airfoil_file"],
                ), promotes=["*"])
            self.add_subsystem(
                "cl_alpha_ht",
                ComputeHTPCLALPHAvlm(
                    low_speed_aero=True,
                    wing_airfoil_file=self.options["wing_airfoil_file"],
                    htp_airfoil_file=self.options["htp_airfoil_file"],
                ), promotes=["*"])
        else:
            self.add_subsystem(
                "cl_cm_ht",
                ComputeHTPCLCMopenvsp(
                    wing_airfoil_file=self.options["wing_airfoil_file"],
                    htp_airfoil_file=self.options["htp_airfoil_file"],
                ), promotes=["*"])
            self.add_subsystem(
                "cl_alpha_ht",
                ComputeHTPCLALPHAopenvsp(
                    low_speed_aero=True,
                    wing_airfoil_file=self.options["wing_airfoil_file"],
                    htp_airfoil_file=self.options["htp_airfoil_file"],
                ), promotes=["*"])
        self.add_subsystem("cl_max", ComputeMaxCL(), promotes=["*"])

        if not (self.options["use_openvsp"]):
            self.connect("data:aerodynamics:low_speed:mach", "comp_polar1.xfoil:mach")
            self.connect("data:aerodynamics:low_speed:unit_reynolds", "comp_polar1.xfoil:unit_reynolds")
            self.connect("xfoil_in.xfoil:length1", "comp_polar1.xfoil:length")
            self.connect("comp_polar1.xfoil:CL", "data:aerodynamics:wing:low_speed:CL")
            self.connect("comp_polar1.xfoil:CDp", "data:aerodynamics:wing:low_speed:CDp")
        self.connect("data:aerodynamics:low_speed:mach", "comp_polar2.xfoil:mach")
        self.connect("data:aerodynamics:low_speed:unit_reynolds", "comp_polar2.xfoil:unit_reynolds")
        self.connect("xfoil_in.xfoil:length2", "comp_polar2.xfoil:length")
        self.connect("comp_polar2.xfoil:CL_max_2D", "data:aerodynamics:wing:low_speed:root:CL_max_2D")
        self.connect("data:aerodynamics:low_speed:mach", "comp_polar3.xfoil:mach")
        self.connect("data:aerodynamics:low_speed:unit_reynolds", "comp_polar3.xfoil:unit_reynolds")
        self.connect("xfoil_in.xfoil:length3", "comp_polar3.xfoil:length")
        self.connect("comp_polar3.xfoil:CL_max_2D", "data:aerodynamics:wing:low_speed:tip:CL_max_2D")


class Connection(ExplicitComponent):
    def setup(self):
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_output("xfoil:length1", units="m")
        self.add_output("xfoil:length2", units="m")
        self.add_output("xfoil:length3", units="m")
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs["xfoil:length1"] = inputs["data:geometry:wing:MAC:length"]
        outputs["xfoil:length2"] = inputs["data:geometry:wing:root:chord"]
        outputs["xfoil:length3"] = inputs["data:geometry:wing:tip:chord"]
