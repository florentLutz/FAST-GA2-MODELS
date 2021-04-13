"""
Computes the aerodynamic loads on the wing of the aircraft in the most stringent case according to aerostructural loads
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

from .aerostructural_loads_x57 import AerostructuralLoadX57, SPAN_MESH_POINT_LOADS
from ..aerodynamics.constants import SPAN_MESH_POINT
from fastoad.utils.physics.atmosphere import Atmosphere
from ..aerodynamics.lift_equilibrium import AircraftEquilibrium


class AerodynamicLoadsX57(AerostructuralLoadX57):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self):
        super().initialize()

    def setup(self):
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")

        nans_array_ov = np.full(SPAN_MESH_POINT, np.nan)
        self.add_input("data:loads:max_shear:load_factor", val=np.nan)
        self.add_input("data:loads:max_shear:cg_position", val=np.nan, units="m")
        self.add_input("data:loads:max_shear:mass", val=np.nan, units="kg")
        self.add_input("data:loads:max_rbm:load_factor", val=np.nan)
        self.add_input("data:loads:max_rbm:cg_position", val=np.nan, units="m")
        self.add_input("data:loads:max_rbm:mass", val=np.nan, units="kg")

        self.add_input("data:aerodynamics:wing:low_speed:Y_vector", val=nans_array_ov, shape=SPAN_MESH_POINT,
                       units="m")
        self.add_input("data:aerodynamics:wing:low_speed:chord_vector", val=nans_array_ov, shape=SPAN_MESH_POINT,
                       units="m")
        self.add_input("data:aerodynamics:wing:low_speed:CL_vector", val=nans_array_ov, shape=SPAN_MESH_POINT)
        self.add_input("data:aerodynamics:slipstream:wing:only_prop:CL_vector", val=nans_array_ov,
                       shape=SPAN_MESH_POINT)
        self.add_input("data:aerodynamics:slipstream:wing:prop_on:Y_vector", val=nans_array_ov,
                       shape=SPAN_MESH_POINT, units="m")
        self.add_input("data:aerodynamics:slipstream:wing:prop_on:velocity", val=np.nan, units="m/s")
        self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CM0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_min_clean", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:efficiency", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:cruise:CL_alpha", val=np.nan)

        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:virtual_chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:height", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:type", val=np.nan)
        self.add_input("data:geometry:propulsion:layout", val=np.nan)
        self.add_input("data:geometry:propulsion:count", val=np.nan)
        self.add_input("data:geometry:propulsion:y_ratio", val=np.nan)
        self.add_input("data:geometry:propulsion:nacelle:width", val=np.nan, units="m")

        self.add_input("data:weight:propulsion:engine:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:landing_gear:main:mass", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft_empty:CG:z", val=np.nan, units="m")

        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")

        self.add_output("data:loads:aerodynamic:ultimate:force_distribution", units="N/m", shape=SPAN_MESH_POINT_LOADS)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        y_vector = inputs["data:aerodynamics:wing:low_speed:Y_vector"]
        y_vector_slip = inputs["data:aerodynamics:slipstream:wing:prop_on:Y_vector"]
        cl_vector = inputs["data:aerodynamics:wing:low_speed:CL_vector"]
        cl_vector_slip = inputs["data:aerodynamics:slipstream:wing:only_prop:CL_vector"]
        chord_vector = inputs["data:aerodynamics:wing:low_speed:chord_vector"]
        cl_0 = inputs["data:aerodynamics:wing:cruise:CL0_clean"]
        v_ref = inputs["data:aerodynamics:slipstream:wing:prop_on:velocity"]

        semi_span = float(inputs["data:geometry:wing:span"])/2.0
        root_chord = float(inputs["data:geometry:wing:root:chord"])
        tip_chord = float(inputs["data:geometry:wing:tip:chord"])

        load_factor_shear = float(inputs["data:loads:max_shear:load_factor"])
        load_factor_rbm = float(inputs["data:loads:max_rbm:load_factor"])

        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]

        cruise_v_tas = inputs["data:TLAR:v_cruise"]

        # We delete the zeros we had to add to fit the size we set in the aerodynamics module and add the physic
        # extrema that are missing, the root and the full span,
        y_vector = AerostructuralLoadX57.delete_additional_zeros(y_vector)
        y_vector_slip = AerostructuralLoadX57.delete_additional_zeros(y_vector_slip)
        cl_vector = AerostructuralLoadX57.delete_additional_zeros(cl_vector)
        cl_vector_slip = AerostructuralLoadX57.delete_additional_zeros(cl_vector_slip)
        chord_vector = AerostructuralLoadX57.delete_additional_zeros(chord_vector)
        y_vector, _ = AerostructuralLoadX57.insert_in_sorted_array(y_vector, 0.)
        y_vector_slip, _ = AerostructuralLoadX57.insert_in_sorted_array(y_vector_slip, 0.)
        cl_vector = np.insert(cl_vector, 0, cl_vector[0])
        cl_vector_slip = np.insert(cl_vector_slip, 0, cl_vector_slip[0])
        chord_vector = np.insert(chord_vector, 0, root_chord)
        y_vector_orig, _ = AerostructuralLoadX57.insert_in_sorted_array(y_vector, semi_span)
        y_vector_slip_orig, _ = AerostructuralLoadX57.insert_in_sorted_array(y_vector_slip, semi_span)
        cl_vector = np.append(cl_vector, 0.)
        cl_vector_slip = np.append(cl_vector_slip, 0.)
        chord_vector = np.append(chord_vector, tip_chord)

        # To get the same y_vector array as in the aerostructural computation, the only important part here is the
        # location of the y samples so we don't need to register the structural mass array
        y_vector, _ = self.compute_relief_force_x57(inputs, y_vector_orig, chord_vector, 0., 0., False)

        # Now we identify the constraint that gives the highest lift distribution in amplitude which is linked with
        # the highest absolute load factor. From there we can recompute the equilibrium and get the lift distribution
        # in the most stringent case which will be what we will plot
        if load_factor_shear > load_factor_rbm:
            x_cg = inputs["data:loads:max_shear:cg_position"]
            mass = inputs["data:loads:max_shear:mass"]
            load_factor = load_factor_shear
        else:
            x_cg = inputs["data:loads:max_rbm:cg_position"]
            mass = inputs["data:loads:max_rbm:mass"]
            load_factor = load_factor_rbm

        atm = Atmosphere(cruise_alt)
        dynamic_pressure = 1. / 2. * atm.density * cruise_v_tas ** 2.0

        cl_wing, _, _, _ = AircraftEquilibrium.found_cl_repartition(inputs, load_factor, mass, dynamic_pressure,
                                                                    False, x_cg)
        cl_s = self.compute_Cl_S(y_vector_orig, y_vector, cl_vector, chord_vector)
        cl_s_slip = self.compute_Cl_S(y_vector_slip_orig, y_vector, cl_vector_slip, chord_vector)
        cl_s_actual = cl_s * cl_wing / cl_0
        cl_s_slip_actual = cl_s_slip * (v_ref / cruise_v_tas) ** 2.0
        lift_distribution = (cl_s_actual+cl_s_slip_actual) * dynamic_pressure

        additional_zeros = np.zeros(SPAN_MESH_POINT_LOADS - len(y_vector))
        lift_distribution_outputs = np.concatenate([lift_distribution, additional_zeros])

        outputs["data:loads:aerodynamic:ultimate:force_distribution"] = lift_distribution_outputs

