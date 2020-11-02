"""
    Estimation of nacelle and pylon geometry
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


class ComputeNacelleGeometry(om.ExplicitComponent):
    # TODO: Document equations. Cite sources
    """ Nacelle and pylon geometry estimation """

    def setup(self):
        
        self.add_input("data:geometry:propulsion:layout", val=np.nan)
        self.add_input("data:geometry:propulsion:engine:height", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:engine:width", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:engine:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:engine:y_ratio", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        
        self.add_output("data:geometry:propulsion:nacelle:length", units="m")
        self.add_output("data:geometry:propulsion:nacelle:diameter", units="m")
        self.add_output("data:geometry:propulsion:nacelle:height", units="m")
        self.add_output("data:geometry:propulsion:nacelle:width", units="m")
        self.add_output("data:geometry:propulsion:nacelle:wet_area", units="m**2")
        self.add_output("data:geometry:landing_gear:height", units="m")
        self.add_output("data:geometry:propulsion:nacelle:y", units="m")
        
        self.declare_partials("data:geometry:propulsion:nacelle:length",
                              "data:geometry:propulsion:engine:length", method="fd")
        
        self.declare_partials(
                "data:geometry:propulsion:nacelle:diameter",
                [
                        "data:geometry:propulsion:engine:height",
                        "data:geometry:propulsion:engine:width",
                ],
                method="fd",
        )
        
        self.declare_partials("data:geometry:propulsion:nacelle:height",
                              "data:geometry:propulsion:engine:height", method="fd")
        
        self.declare_partials("data:geometry:propulsion:nacelle:width",
                              "data:geometry:propulsion:engine:width", method="fd")
                
        self.declare_partials(
                "data:geometry:propulsion:nacelle:wet_area",
                [
                        "data:geometry:propulsion:engine:length",
                        "data:geometry:propulsion:engine:height",
                        "data:geometry:propulsion:engine:width",
                ],
                method="fd",
        )
        
        self.declare_partials(
                "data:geometry:landing_gear:height",
                [
                        "data:geometry:propulsion:engine:height",
                        "data:geometry:propulsion:engine:width",
                ],
                method="fd",
        )
        
        self.declare_partials(
                "data:geometry:propulsion:nacelle:y",
                [
                        "data:geometry:propulsion:engine:height",
                        "data:geometry:propulsion:engine:width",
                        "data:geometry:propulsion:engine:y_ratio",
                        "data:geometry:wing:span",
                        "data:geometry:fuselage:maximum_width",
                ],
                method="fd",
        )
        

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        engine_loc = inputs["data:geometry:propulsion:layout"]
        engine_height = inputs["data:geometry:propulsion:engine:height"]
        engine_width = inputs["data:geometry:propulsion:engine:width"]
        engine_length = inputs["data:geometry:propulsion:engine:length"]
        span = inputs["data:geometry:wing:span"]
        y_ratio_engine = inputs["data:geometry:propulsion:engine:y_ratio"]
        b_f = inputs["data:geometry:fuselage:maximum_width"]

        nac_dia = 1.1*max(engine_height, engine_width)
        nac_height = engine_height*1.1
        nac_width = engine_width*1.1 
        nac_length = 1.5*engine_length
        nac_wet_area = 2 * (nac_height + nac_width) * nac_length
        if engine_loc == 1.0:
            y_nacelle = y_ratio_engine * span / 2
        elif engine_loc == 2.0:
            y_nacelle = b_f / 2 + 0.8 * nac_dia
        elif engine_loc == 3.0:
            y_nacelle = 0
        else:
            raise ValueError('compute_fuselage model only computes propulsion layout equal to 1, 2 or 3!')
        
        lg_height = 1.4 * nac_dia  # ???: always?
        
        outputs["data:geometry:propulsion:nacelle:length"] = nac_length
        outputs["data:geometry:propulsion:nacelle:diameter"] = nac_dia
        outputs["data:geometry:propulsion:nacelle:height"] = nac_height
        outputs["data:geometry:propulsion:nacelle:width"] = nac_width
        outputs["data:geometry:propulsion:nacelle:wet_area"] = nac_wet_area
        outputs["data:geometry:landing_gear:height"] = lg_height
        outputs["data:geometry:propulsion:nacelle:y"] = y_nacelle
