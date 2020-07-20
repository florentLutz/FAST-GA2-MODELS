"""
    Estimation of max fuel weight
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
from openmdao.core.explicitcomponent import ExplicitComponent

# TODO: This belongs more to mass breakdown than geometry
class ComputeMFW(ExplicitComponent):

    """ Max fuel weight estimation based o RAYMER table 10.5 p269"""

    def setup(self):
    
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:propulsion:engine:fuel_type", val=np.nan)

        self.add_output("data:weight:aircraft:MFW", units="kg")

        self.declare_partials("data:weight:aircraft:MFW", "data:geometry:wing:area", method="fd")

    def compute(self, inputs, outputs):
        
        fuel_type = inputs["data:propulsion:engine:fuel_type"]
        wing_area = inputs["data:geometry:wing:area"]
        
        if fuel_type == 1.0:
            m_vol_fuel = 0.73 # Cold because worst case
        elif fuel_type == 2.0:
            m_vol_fuel = 0.0 # FIXME: to be changed for real value
        else:
            raise IOError("Bad motor configuration: only fuel type 1/2 available.")

        # Tanks are between 1st (25% MAC) and 3rd (60% MAC) longeron: 35% of the wing
        mfv = 0.35*wing_area*0.12*1000 #in L                                                                                                              
        mfw = mfv*m_vol_fuel

        outputs["data:weight:aircraft:MFW"] = mfw
