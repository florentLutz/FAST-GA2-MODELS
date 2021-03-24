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
from .models.aerodynamics.aerodynamics import Aerodynamics
from .models.geometry import GeometryFixedFuselage, GeometryFixedTailDistance
from .models.handling_qualities.compute_static_margin import ComputeStaticMargin
from .models.handling_qualities.handling_qualities import ComputeHandlingQualities
from .models.handling_qualities.tail_sizing import UpdateTailAreas
from .models.loops import UpdateWingArea, UpdateWingPosition
from .models.weight.mass_breakdown.update_mtow import UpdateMTOW
from .models.performances.sizing import Sizing
from .models.weight.weight import Weight
from .models.load_analysis.loads import Loads
from .models.load_analysis.private.wing_mass_estimation import AerostructuralLoadsAlternate
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
    domain=ModelDomain.AERODYNAMICS
)
OpenMDAOSystemRegistry.register_system(
    Aerodynamics,
    "fastga.aerodynamics.legacy",
    domain=ModelDomain.AERODYNAMICS
)

# Geometry ####################################################################
OpenMDAOSystemRegistry.register_system(
    GeometryFixedTailDistance,
    "fastga.geometry.legacy",
    domain=ModelDomain.GEOMETRY
)

OpenMDAOSystemRegistry.register_system(
    GeometryFixedFuselage,
    "fastga.geometry.alternate",
    domain=ModelDomain.GEOMETRY
)

# handling qualities ##########################################################
OpenMDAOSystemRegistry.register_system(
    UpdateTailAreas,
    "fastga.handling_qualities.tail_sizing",
    domain=ModelDomain.HANDLING_QUALITIES,
)
OpenMDAOSystemRegistry.register_system(
    ComputeStaticMargin,
    "fastga.handling_qualities.static_margin",
    domain=ModelDomain.HANDLING_QUALITIES,
)
OpenMDAOSystemRegistry.register_system(
    ComputeHandlingQualities,
    "fastga.handling_qualities.all_handling_qualities",
    domain=ModelDomain.HANDLING_QUALITIES,
)
# Loops #######################################################################
OpenMDAOSystemRegistry.register_system(
    UpdateWingArea,
    "fastga.loop.wing_area",
    domain=ModelDomain.OTHER
)
OpenMDAOSystemRegistry.register_system(
    UpdateWingPosition,
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

# Aerostructural loads ########################################################
OpenMDAOSystemRegistry.register_system(
    Loads,
    "fastga.loads.legacy",
    domain=ModelDomain.OTHER
)

OpenMDAOSystemRegistry.register_system(
    AerostructuralLoadsAlternate,
    "fastga.loads.alternate",
    domain=ModelDomain.OTHER
)
