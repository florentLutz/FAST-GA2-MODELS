"""
Main components for mass breakdown
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

import openmdao.api as om

from fastoad.models.options import PAYLOAD_FROM_NPAX
from .a_airframe import (
    ComputeWingWeight,
    ComputeFuselageWeight,
    ComputeTailWeight,
    ComputeFlightControlsWeight,
    ComputeLandingGearWeight,
)
from .b_propulsion import (
    ComputeEngineWeight,
    ComputeFuelLinesWeight,
)
from .c_systems import (
    ComputePowerSystemsWeight,
    ComputeLifeSupportSystemsWeight,
    ComputeNavigationSystemsWeight,
)
from .d_furniture import (
    ComputePassengerSeatsWeight,
)
from .payload import ComputePayload
from .update_mlw_and_mzfw import UpdateMLWandMZFW
from .update_mtow import UpdateMTOW


class MassBreakdown(om.Group):
    """
    Computes analytically the mass of each part of the aircraft, and the resulting sum,
    the Overall Weight Empty (OWE).

    Some models depend on MZFW (Max Zero Fuel Weight) and MTOW (Max TakeOff Weight),
    which depend on OWE.

    This model cycles for having consistent OWE, MZFW and MTOW based on MFW.

    Options:
    - payload_from_npax: If True (default), payload masses will be computed from NPAX, if False
                         design payload mass and maximum payload mass must be provided.
    """

    def initialize(self):
        self.options.declare(PAYLOAD_FROM_NPAX, types=bool, default=True)

    def setup(self):
        if self.options[PAYLOAD_FROM_NPAX]:
            self.add_subsystem("payload", ComputePayload(), promotes=["*"])
        self.add_subsystem("owe", ComputeOperatingWeightEmpty(), promotes=["*"])
        self.add_subsystem("update_mzfw_and_mlw", UpdateMLWandMZFW(), promotes=["*"])
        # self.add_subsystem("update_mtow", UpdateMTOW(), promotes=["*"])

        # Solvers setup
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options["debug_print"] = True
        self.nonlinear_solver.options["err_on_non_converge"] = True
        self.nonlinear_solver.options["iprint"] = 1
        self.nonlinear_solver.options["maxiter"] = 50
        self.nonlinear_solver.options["reraise_child_analysiserror"] = True
        self.nonlinear_solver.options["rtol"] = 1e-3
        # self.nonlinear_solver.options["stall_limit"] = 1
        # self.nonlinear_solver.options["stall_tol"] = 1e-5

        self.linear_solver = om.LinearBlockGS()
        self.linear_solver.options["err_on_non_converge"] = True
        self.linear_solver.options["iprint"] = 1
        self.linear_solver.options["maxiter"] = 10
        self.linear_solver.options["rtol"] = 1e-3


class ComputeOperatingWeightEmpty(om.Group):
    """ Operating Empty Weight (OEW) estimation

    This group aggregates weight from all components of the aircraft.
    """

    def setup(self):
        # Airframe
        self.add_subsystem("wing_weight", ComputeWingWeight(), promotes=["*"])
        self.add_subsystem("fuselage_weight", ComputeFuselageWeight(), promotes=["*"])
        self.add_subsystem("empennage_weight", ComputeTailWeight(), promotes=["*"])
        self.add_subsystem("flight_controls_weight", ComputeFlightControlsWeight(), promotes=["*"])
        self.add_subsystem("landing_gear_weight", ComputeLandingGearWeight(), promotes=["*"])
        self.add_subsystem("engine_weight", ComputeEngineWeight(), promotes=["*"])
        self.add_subsystem("fuel_lines_weight", ComputeFuelLinesWeight(), promotes=["*"])
        self.add_subsystem("navigation_systems_weight", ComputeNavigationSystemsWeight(), promotes=["*"])
        self.add_subsystem("power_systems_weight", ComputePowerSystemsWeight(), promotes=["*"])
        self.add_subsystem("life_support_systems_weight", ComputeLifeSupportSystemsWeight(), promotes=["*"])
        self.add_subsystem("passenger_seats_weight", ComputePassengerSeatsWeight(), promotes=["*"])


        airframe_sum = om.AddSubtractComp()
        airframe_sum.add_equation(
            "data:weight:airframe:mass",
            [
                "data:weight:airframe:wing:mass",
                "data:weight:airframe:fuselage:mass",
                "data:weight:airframe:horizontal_tail:mass",
                "data:weight:airframe:vertical_tail:mass",
                "data:weight:airframe:flight_controls:mass",
                "data:weight:airframe:landing_gear:main:mass",
                "data:weight:airframe:landing_gear:front:mass",
            ],
            units="kg",
            desc="Mass of crew",
        )

        # Make additions
        self.add_subsystem(
            "airframe_weight_sum",
            airframe_sum,
            promotes=["*"]
        )

        propulsion_sum = om.AddSubtractComp()
        propulsion_sum.add_equation(
            "data:weight:propulsion:mass",
            [
                "data:weight:propulsion:engine:mass",
                "data:weight:propulsion:fuel_lines:mass",
            ],
            units="kg",
            desc="Mass of the propulsion system",
        )

        self.add_subsystem(
            "propulsion_weight_sum",
            propulsion_sum,
            promotes=["*"]
        )

        systems_sum = om.AddSubtractComp()
        systems_sum.add_equation(
            "data:weight:systems:mass",
            [
                "data:weight:systems:power:electric_systems:mass",
                "data:weight:systems:power:hydraulic_systems:mass",
                "data:weight:systems:life_support:air_conditioning:mass",
                "data:weight:systems:navigation:mass",
            ],
            units="kg",
            desc="Mass of aircraft systems",
        )

        self.add_subsystem(
            "systems_weight_sum",
            systems_sum,
            promotes=["*"],
        )

        furniture_sum = om.AddSubtractComp()
        furniture_sum.add_equation(
            "data:weight:furniture:mass",
            [
                "data:weight:furniture:passenger_seats:mass",
                "data:weight:systems:navigation:mass",
            ],
            scaling_factors=[1.0, 0.0],
            units="kg",
            desc="Mass of aircraft furniture",
        )

        self.add_subsystem(
            "furniture_weight_sum",
            furniture_sum,
            promotes=["*"],
        )

        OWE_sum = om.AddSubtractComp()
        OWE_sum.add_equation(
            "data:weight:aircraft:OWE",
            [
                "data:weight:airframe:mass",
                "data:weight:propulsion:mass",
                "data:weight:systems:mass",
                "data:weight:furniture:mass",
            ],
            units="kg",
            desc="Mass of aircraft",  # !!!: initially "Mass of crew" changed description
        )

        self.add_subsystem(
            "OWE_sum",
            OWE_sum,
            promotes=["*"],
        )
