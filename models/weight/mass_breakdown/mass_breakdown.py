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
    WingWeight,
    FuselageWeight,
    EmpennageWeight,
    FlightControlsWeight,
    LandingGearWeight,
)
from .b_propulsion import (
    EngineWeight,
    FuelLinesWeight,
)
from .c_systems import (
    PowerSystemsWeight,
    LifeSupportSystemsWeight,
    NavigationSystemsWeight,
)
from .d_furniture import (
    PassengerSeatsWeight,
)
from .payload import ComputePayload
from .update_mlw_and_mzfw import UpdateMLWandMZFW


class MassBreakdown(om.Group):
    """
    Computes analytically the mass of each part of the aircraft, and the resulting sum,
    the Overall Weight Empty (OWE).

    Some models depend on MZFW (Max Zero Fuel Weight), MLW (Max Landing Weight) and
    MTOW (Max TakeOff Weight), which depend on OWE.

    This model cycles for having consistent OWE, MZFW and MLW.
    Consistency with MTOW can be achieved by cycling with a model that computes MTOW from OWE,
    which should come from a mission computation that will assess needed block fuel.

    Options:
    - payload_from_npax: If True (default), payload masses will be computed from NPAX, if False
                         design payload mass and maximum payload mass must be provided.
    """

    def initialize(self):
        self.options.declare(PAYLOAD_FROM_NPAX, types=bool, default=True)

    def setup(self):
        if self.options[PAYLOAD_FROM_NPAX]:
            self.add_subsystem("payload", ComputePayload(), promotes=["*"])
        self.add_subsystem("owe", OperatingWeightEmpty(), promotes=["*"])
        self.add_subsystem("update_mzfw_and_mlw", UpdateMLWandMZFW(), promotes=["*"])

        # Solvers setup
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 50

        self.linear_solver = om.LinearBlockGS()
        self.linear_solver.options["iprint"] = 0


class OperatingWeightEmpty(om.Group):
    """ Operating Empty Weight (OEW) estimation

    This group aggregates weight from all components of the aircraft.
    """

    def setup(self):
        # Airframe
        self.add_subsystem("wing_weight", WingWeight(), promotes=["*"])
        self.add_subsystem("fuselage_weight", FuselageWeight(), promotes=["*"])
        self.add_subsystem("empennage_weight", EmpennageWeight(), promotes=["*"])
        self.add_subsystem("flight_controls_weight", FlightControlsWeight(), promotes=["*"])
        self.add_subsystem("landing_gear_weight", LandingGearWeight(), promotes=["*"])
        self.add_subsystem("engine_weight", EngineWeight(), promotes=["*"])
        self.add_subsystem("fuel_lines_weight", FuelLinesWeight(), promotes=["*"])
        self.add_subsystem("power_systems_weight", PowerSystemsWeight(), promotes=["*"])
        self.add_subsystem("life_support_systems_weight", LifeSupportSystemsWeight(), promotes=["*"])
        self.add_subsystem("navigation_systems_weight", NavigationSystemsWeight(), promotes=["*"])
        self.add_subsystem("passenger_seats_weight", PassengerSeatsWeight(), promotes=["*"])

        # Make additions
        self.add_subsystem(
            "airframe_weight_sum",
            om.AddSubtractComp(
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
                desc="Mass of airframe",
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "propulsion_weight_sum",
            om.AddSubtractComp(
                "data:weight:propulsion:mass",
                [
                    "data:weight:propulsion:engine:mass",
                    "data:weight:propulsion:fuel_lines:mass",
                ],
                units="kg",
                desc="Mass of the propulsion system",
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "systems_weight_sum",
            om.AddSubtractComp(
                "data:weight:systems:mass",
                [
                    "data:weight:systems:power:electric_systems:mass",
                    "data:weight:systems:power:hydraulic_systems:mass",
                    "data:weight:systems:life_support:air_conditioning:mass",
                    "data:weight:systems:navigation:mass",
                ],
                units="kg",
                desc="Mass of aircraft systems",
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "furniture_weight_sum",
            om.AddSubtractComp(
                "data:weight:furniture:mass",
                [
                    "data:weight:furniture:passenger_seats:mass",
                ],
                units="kg",
                desc="Mass of aircraft furniture",
            ),
            promotes=["*"],
        )

        self.add_subsystem(
            "OWE_sum",
            om.AddSubtractComp(
                "data:weight:aircraft:OWE",
                [
                    "data:weight:airframe:mass",
                    "data:weight:propulsion:mass",
                    "data:weight:systems:mass",
                    "data:weight:furniture:mass",
                ],
                units="kg",
                desc="Mass of aicraft", # !!!: initially "Mass of crew" changed description
            ),
            promotes=["*"],
        )
