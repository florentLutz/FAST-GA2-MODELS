"""
    Estimation of global center of gravity
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

from ..cg_components.loadcase import ComputeCGLoadCase
from ..cg_components.loadcase_test import ComputeGroundCGCase, ComputeFlightCGCase
from ..cg_components.ratio_aft import ComputeCGRatioAft
from ..cg_components.max_cg_ratio import ComputeMaxCGratio, ComputeMaxMinCGratio

from openmdao.api import Group


class ComputeGlobalCG(Group):
    # TODO: Document equations. Cite sources
    """ Global center of gravity estimation """

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self.add_subsystem("cg_ratio_aft", ComputeCGRatioAft(), promotes=["*"])
        self.add_subsystem("cg_ratio_lc1", ComputeCGLoadCase(load_case=1), promotes=["*"])
        self.add_subsystem("cg_ratio_lc2", ComputeCGLoadCase(load_case=2), promotes=["*"])
        self.add_subsystem("cg_ratio_lc3", ComputeCGLoadCase(load_case=3), promotes=["*"])
        self.add_subsystem("cg_ratio_lc4", ComputeCGLoadCase(load_case=4), promotes=["*"])
        self.add_subsystem("cg_ratio_lc5", ComputeCGLoadCase(load_case=5), promotes=["*"])
        self.add_subsystem("cg_ratio_lc6", ComputeCGLoadCase(load_case=6), promotes=["*"])
        self.add_subsystem("cg_ratio_max", ComputeMaxCGratio(), promotes=["*"])


class ComputeGlobalCGnew(Group):
    # TODO: Document equations. Cite sources
    """ Global center of gravity estimation """

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self.add_subsystem("cg_ratio_aft", ComputeCGRatioAft(), promotes=["*"])
        self.add_subsystem("cg_ratio_lc_ground", ComputeGroundCGCase(), promotes=["*"])
        self.add_subsystem("cg_ratio_lc_flight",
                           ComputeFlightCGCase(propulsion_id=self.options["propulsion_id"]),
                           promotes=["*"])
        self.add_subsystem("cg_ratio_extrema", ComputeMaxMinCGratio(), promotes=["*"])
