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

import openmdao.api as om
import numpy as np

from .takeoff import TakeOffPhase
from .mission import _compute_taxi, _compute_climb, _compute_cruise, _compute_descent


class Sizing(om.Group):
    """
    Computes analytically the fuel mass necessary for each part of the flight cycle.

    Loop on the distance crossed during descent and cruise distance/fuel mass.

    """

    def initialize(self):
        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)

    def setup(self):
        self.add_subsystem("taxi_out", _compute_taxi(
            propulsion_id=self.options["propulsion_id"],
            taxi_out=True,
        ), promotes=["*"])
        self.add_subsystem("takeoff", TakeOffPhase(propulsion_id=self.options["propulsion_id"]), promotes=["*"])
        self.add_subsystem("climb", _compute_climb(propulsion_id=self.options["propulsion_id"]), promotes=["*"])
        self.add_subsystem("cruise", _compute_cruise(propulsion_id=self.options["propulsion_id"]), promotes=["*"])
        self.add_subsystem("reserve", _compute_reserve(), promotes=["*"])
        self.add_subsystem("descent", _compute_descent(propulsion_id=self.options["propulsion_id"]), promotes=["*"])
        self.add_subsystem("taxi_in", _compute_taxi(
            propulsion_id=self.options["propulsion_id"],
            taxi_out=False,
        ), promotes=["*"])
        self.add_subsystem("update_fw", UpdateFW(), promotes=["*"])

        # Solvers setup
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.nonlinear_solver.options["debug_print"] = True
        self.nonlinear_solver.options["err_on_non_converge"] = True
        self.nonlinear_solver.options["iprint"] = 2
        self.nonlinear_solver.options["maxiter"] = 50
        self.nonlinear_solver.options["reraise_child_analysiserror"] = True
        self.nonlinear_solver.options["rtol"] = 1e-5
        # self.nonlinear_solver.options["stall_limit"] = 1
        # self.nonlinear_solver.options["stall_tol"] = 1e-5

        self.linear_solver = om.LinearBlockGS()
        self.linear_solver.options["err_on_non_converge"] = True
        self.linear_solver.options["iprint"] = 2
        self.linear_solver.options["maxiter"] = 10
        self.linear_solver.options["rtol"] = 1e-5


class _compute_reserve(om.ExplicitComponent):

    def setup(self):

        self.add_input("data:mission:sizing:main_route:cruise:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:cruise:duration", np.nan, units="s")
        self.add_input("data:mission:sizing:main_route:reserve:duration", np.nan, units="s")

        self.add_output("data:mission:sizing:main_route:reserve:fuel", units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        m_reserve = (
                inputs["data:mission:sizing:main_route:cruise:fuel"]
                * inputs["data:mission:sizing:main_route:reserve:duration"]
                / max(1e-6, inputs["data:mission:sizing:main_route:cruise:duration"])  # avoid 0 division
        )
        outputs["data:mission:sizing:main_route:reserve:fuel"] = m_reserve


class UpdateFW(om.ExplicitComponent):

    def setup(self):

        self.add_input("data:mission:sizing:taxi_out:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:takeoff:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:initial_climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:climb:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:cruise:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:reserve:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:descent:fuel", np.nan, units="kg")
        self.add_input("data:mission:sizing:taxi_in:fuel", np.nan, units="kg")

        self.add_output("data:mission:sizing:fuel", val=0.0, units="kg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        m_taxi_out = inputs["data:mission:sizing:taxi_out:fuel"]
        m_takeoff = inputs["data:mission:sizing:takeoff:fuel"]
        m_initial_climb = inputs["data:mission:sizing:initial_climb:fuel"]
        m_climb = inputs["data:mission:sizing:main_route:climb:fuel"]
        m_cruise = inputs["data:mission:sizing:main_route:cruise:fuel"]
        m_reserve = inputs["data:mission:sizing:main_route:reserve:fuel"]
        m_descent = inputs["data:mission:sizing:main_route:descent:fuel"]
        m_taxi_in = inputs["data:mission:sizing:taxi_in:fuel"]

        m_total = (
            m_taxi_out
            + m_takeoff
            + m_initial_climb
            + m_climb
            + m_cruise
            + m_reserve
            + m_descent
            + m_taxi_in
        )

        outputs["data:mission:sizing:fuel"] = m_total
