"""
Estimation of static margin
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
from ..aerodynamics.aero_center import ComputeAeroCenter
from .compute_static_margin import _ComputeStaticMargin
from .tail_sizing.compute_to_rotation_limit import ComputeTORotationLimit, _ComputeAeroCoeffTO
from .tail_sizing.compute_to_rotation_limit import ComputeTORotationLimitGroup
from .tail_sizing.compute_balked_landing_limit import ComputeBalkedLandingLimit
from typing import Union, List, Optional, Tuple


class ComputeHandlingQualities(om.Group):
    """
    Calculate static margins and maneuver limits
    """

    def initialize(self):
        self.options.declare("propulsion_id", default="", types=str)

    def setup(self):
        self.add_subsystem("aero_center", ComputeAeroCenter(), promotes=["*"])
        self.add_subsystem("static_margin", _ComputeStaticMargin(), promotes=["*"])
        self.add_subsystem("to_rotation_limit",
                           ComputeTORotationLimitGroup(propulsion_id=self.options["propulsion_id"]), promotes=["*"])
        self.add_subsystem("balked_landing_limit",
                           ComputeBalkedLandingLimit(propulsion_id=self.options["propulsion_id"]), promotes=["*"])

    @staticmethod
    def get_io_names(
            component: om.ExplicitComponent,
            excludes: Optional[Union[str, List[str]]] = None,
            iotypes: Optional[Union[str, Tuple[str]]] = ('inputs', 'outputs')) -> List[str]:
        prob = om.Problem(model=component)
        prob.setup()
        data = []
        if type(iotypes) == tuple:
            data.extend(prob.model.list_inputs(out_stream=None))
            data.extend(prob.model.list_outputs(out_stream=None))
        else:
            if iotypes == 'inputs':
                data.extend(prob.model.list_inputs(out_stream=None))
            else:
                data.extend(prob.model.list_outputs(out_stream=None))
        list_names = []
        for idx in range(len(data)):
            variable_name = data[idx][0]
            if excludes is None:
                list_names.append(variable_name)
            else:
                if variable_name not in list(excludes):
                    list_names.append(variable_name)

        return list_names
