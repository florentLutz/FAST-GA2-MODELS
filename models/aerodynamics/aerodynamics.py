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

from .aerodynamics_high_speed import AerodynamicsHighSpeed
from .aerodynamics_low_speed import AerodynamicsLowSpeed
from openmdao.api import Group


class Aerodynamics(Group):

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)
        self.options.declare("use_openvsp", default=False, types=bool)
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare('wing_airfoil_file', default="naca23012.af", types=str, allow_none=True)
        self.options.declare('htp_airfoil_file', default="naca0012.af", types=str, allow_none=True)

    def setup(self):
        # Compute the low speed aero (landing/takeoff)
        self.add_subsystem("aero_low",
                           AerodynamicsLowSpeed(
                               propulsion_id=self.options["propulsion_id"],
                               use_openvsp=self.options["use_openvsp"],
                               result_folder_path=self.options["result_folder_path"],
                               wing_airfoil_file=self.options["wing_airfoil_file"],
                               htp_airfoil_file=self.options["htp_airfoil_file"],
                           ), promotes=["*"])

        # Compute cruise characteristics
        self.add_subsystem("aero_high",
                           AerodynamicsHighSpeed(
                               propulsion_id=self.options["propulsion_id"],
                               use_openvsp=self.options["use_openvsp"],
                               result_folder_path=self.options["result_folder_path"],
                               wing_airfoil_file=self.options["wing_airfoil_file"],
                               htp_airfoil_file=self.options["htp_airfoil_file"],
                           ), promotes=["*"])
