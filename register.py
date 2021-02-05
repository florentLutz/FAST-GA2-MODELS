"""
This module is for registering all internal OpenMDAO modules that we want
available through OpenMDAOSystemRegistry
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

from .models.aerodynamics.aerodynamics_high_speed import AerodynamicsHighSpeed
from .models.aerodynamics.aerodynamics_low_speed import AerodynamicsLowSpeed
from .models.geometry import Geometry
from .models.handling_qualities.compute_static_margin import ComputeStaticMargin
from .models.handling_qualities.tail_sizing.compute_tail_areas import ComputeTailAreas
from .models.loops.compute_wing_area import ComputeWingArea
from .models.loops.compute_wing_position import ComputeWingPosition
from .models.weight.mass_breakdown.update_mtow import UpdateMTOW
from .models.performances.sizing import Sizing
from .models.propulsion.fuel_propulsion.basicIC_engine import OMBasicICEngineComponent
from .models.weight.weight import Weight
from fastoad.module_management import OpenMDAOSystemRegistry
from fastoad.module_management.constants import ModelDomain


"""
The place where to register FAST-OAD internal models.

Warning: this function is effective only if called from a Python module that
is a started bundle for iPOPO
"""
# Aerodynamics ################################################################
OpenMDAOSystemRegistry.register_system(
    AerodynamicsLowSpeed,
    "fastga.aerodynamics.lowspeed.legacy",
    domain=ModelDomain.AERODYNAMICS
)
OpenMDAOSystemRegistry.register_system(
    AerodynamicsHighSpeed,
    "fastga.aerodynamics.highspeed.legacy",
    domain=ModelDomain.AERODYNAMICS,
)

# Geometry ####################################################################
OpenMDAOSystemRegistry.register_system(
    Geometry,
    "fastga.geometry.legacy",
    domain=ModelDomain.GEOMETRY
)

# handling qualities ##########################################################
OpenMDAOSystemRegistry.register_system(
    ComputeTailAreas,
    "fastga.handling_qualities.tail_sizing",
    domain=ModelDomain.HANDLING_QUALITIES,
)
OpenMDAOSystemRegistry.register_system(
    ComputeStaticMargin,
    "fastga.handling_qualities.static_margin",
    domain=ModelDomain.HANDLING_QUALITIES,
)

# Loops #######################################################################
OpenMDAOSystemRegistry.register_system(
    ComputeWingArea,
    "fastga.loop.wing_area",
    domain=ModelDomain.OTHER
)
OpenMDAOSystemRegistry.register_system(
    ComputeWingPosition,
    "fastoad.loop.wing_position",
    domain=ModelDomain.OTHER
)
OpenMDAOSystemRegistry.register_system(
    UpdateMTOW,
    "fastga.loop.mtow",
    domain=ModelDomain.OTHER
)


# Weight ######################################################################
OpenMDAOSystemRegistry.register_system(
    Weight,
    "fastga.weight.legacy",
    domain=ModelDomain.WEIGHT
)
# Performance #################################################################
OpenMDAOSystemRegistry.register_system(
    Sizing,
    "fastga.performances.sizing",
    domain=ModelDomain.PERFORMANCE
)
