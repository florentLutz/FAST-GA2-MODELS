"""
Computes the aerostructural loads on the wing of the aircraft
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

import math

import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt

from ..aerodynamics.constants import SPAN_MESH_POINT, MACH_NB_PTS
from ..aerodynamics.external.openvsp.compute_vn import ComputeVNopenvsp
from fastoad.utils.physics.atmosphere import Atmosphere
from scipy.integrate import trapz, cumtrapz
from scipy.interpolate import interp1d
from ..aerodynamics.lift_equilibrium import AircraftEquilibrium

NB_POINTS_POINT_MASS = 5
# MUST BE AN EVEN NUMBER
POINT_MASS_SPAN_RATIO = 0.01
SPAN_MESH_POINT_LOADS = int(1.5 * SPAN_MESH_POINT)


class AerostructuralLoad(ComputeVNopenvsp):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self):
        super().initialize()

    def setup(self):

        self.add_input("data:TLAR:category", val=3.0)
        self.add_input("data:TLAR:level", val=1.0)
        self.add_input("data:TLAR:v_max_sl", val=np.nan, units="kn")
        self.add_input("data:TLAR:v_cruise", val=np.nan, units="m/s")
        self.add_input("data:TLAR:v_approach", val=np.nan, units="m/s")

        nans_array_OV = np.full(SPAN_MESH_POINT, np.nan)
        nans_array_M = np.full(MACH_NB_PTS+1, np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:Y_vector", val=nans_array_OV, shape=SPAN_MESH_POINT,
                       units="m")
        self.add_input("data:aerodynamics:wing:low_speed:chord_vector", val=nans_array_OV, shape=SPAN_MESH_POINT,
                       units="m")
        self.add_input("data:aerodynamics:wing:low_speed:CL_vector", val=nans_array_OV, shape=SPAN_MESH_POINT)
        self.add_input("data:aerodynamics:wing:low_speed:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CL0_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CM0_clean", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:efficiency", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:landing:CL_max", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_max_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_min_clean", val=np.nan)
        self.add_input("data:aerodynamics:wing:low_speed:CL_alpha", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:low_speed:CL_alpha", val=np.nan)
        self.add_input("data:aerodynamics:horizontal_tail:cruise:CL_alpha", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:mach_interpolation:CL_alpha_vector", val=nans_array_M, units="rad**-1",
                       shape=MACH_NB_PTS + 1)
        self.add_input("data:aerodynamics:aircraft:mach_interpolation:mach_vector", val=nans_array_M,
                       shape=MACH_NB_PTS + 1)

        self.add_input("data:weight:aircraft:MZFW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:weight:propulsion:engine:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:landing_gear:main:mass", val=np.nan, units="kg")
        self.add_input("data:weight:airframe:wing:mass", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft_empty:CG:z", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:CG:aft:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:CG:fwd:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:in_flight_variation:fixed_mass_comp:equivalent_moment", np.nan,
                       units="kg*m")
        self.add_input("data:weight:propulsion:tank:CG:x", np.nan, units="m")
        self.add_input("data:weight:aircraft:in_flight_variation:fixed_mass_comp:mass", np.nan, units="kg")

        self.add_input("data:geometry:wing:root:virtual_chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:landing_gear:height", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:layout", val=np.nan)
        self.add_input("data:geometry:propulsion:count", val=np.nan)
        self.add_input("data:geometry:propulsion:y_ratio", val=np.nan)

        self.add_input("data:mission:sizing:fuel", val=np.nan, units="kg")
        self.add_input("data:mission:sizing:main_route:cruise:altitude", val=np.nan, units="ft")

        self.add_output("data:loads:max_shear:mass", units="kg")
        self.add_output("data:loads:max_shear:load_factor")
        self.add_output("data:loads:max_shear:cg_position", units="m")
        self.add_output("data:loads:max_shear:lift_shear", units="N", shape=SPAN_MESH_POINT_LOADS)
        self.add_output("data:loads:max_shear:weight_shear", units="N", shape=SPAN_MESH_POINT_LOADS)

        self.add_output("data:loads:max_rbm:mass", units="kg")
        self.add_output("data:loads:max_rbm:load_factor")
        self.add_output("data:loads:max_rbm:cg_position", units="m")
        self.add_output("data:loads:max_rbm:lift_rbm", units="N*m", shape=SPAN_MESH_POINT_LOADS)
        self.add_output("data:loads:max_rbm:weight_rbm", units="N*m", shape=SPAN_MESH_POINT_LOADS)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        y_vector = inputs["data:aerodynamics:wing:low_speed:Y_vector"]
        cl_vector = inputs["data:aerodynamics:wing:low_speed:CL_vector"]
        cl_0 = inputs["data:aerodynamics:wing:low_speed:CL0_clean"]
        chord_vector = inputs["data:aerodynamics:wing:low_speed:chord_vector"]

        semi_span = inputs["data:geometry:wing:span"] / 2.0
        root_chord = inputs["data:geometry:wing:root:chord"]
        tip_chord = inputs["data:geometry:wing:tip:chord"]

        mtow = inputs["data:weight:aircraft:MTOW"]
        mzfw = inputs["data:weight:aircraft:MZFW"]
        wing_mass = inputs["data:weight:airframe:wing:mass"]
        cg_aft = inputs["data:weight:aircraft:CG:aft:x"]
        cg_fwd = inputs["data:weight:aircraft:CG:fwd:x"]

        fuel_mass = inputs["data:mission:sizing:fuel"]
        cruise_alt = inputs["data:mission:sizing:main_route:cruise:altitude"]

        cruise_v_tas = inputs["data:TLAR:v_cruise"]

        # We delete the zeros we had to add to fit the size we set in the aerodynamics module and add the physic
        # extrema that are missing, the root and the full span,
        y_vector = AerostructuralLoad.delete_additional_zeros(y_vector)
        cl_vector = AerostructuralLoad.delete_additional_zeros(cl_vector)
        chord_vector = AerostructuralLoad.delete_additional_zeros(chord_vector)
        y_vector, _ = AerostructuralLoad.insert_in_sorted_array(y_vector, 0.)
        cl_vector = np.insert(cl_vector, 0, cl_vector[0])
        chord_vector = np.insert(chord_vector, 0, root_chord)
        y_vector_orig, _ = AerostructuralLoad.insert_in_sorted_array(y_vector, semi_span)
        cl_vector = np.append(cl_vector, 0.)
        chord_vector = np.append(chord_vector, tip_chord)

        FoS = 1.5
        shear_max_conditions = []
        rbm_max_conditions = []

        y_vector, weight_array_orig = self.compute_relief_force(inputs, y_vector_orig, wing_mass, fuel_mass)
        cl_s = self.compute_Cl_S(y_vector_orig, y_vector, cl_vector, chord_vector)

        mass_array = np.array([mtow, 1.05 * mzfw])
        cg_array = np.array([cg_fwd, cg_aft])

        atm = Atmosphere(cruise_alt)

        shear_max = 0.0
        rbm_max = 0.0

        for mass in mass_array:

            cruise_v_keas = atm.get_equivalent_airspeed(cruise_v_tas)

            velocity_array, load_factor_array, _ = self.flight_domain(inputs, outputs, mass, cruise_alt, cruise_v_keas)

            Va = max(float(velocity_array[2]), float(velocity_array[4]))
            Vc = float(velocity_array[6])
            Vd = float(velocity_array[9])

            load_factor_list = np.array([max(load_factor_array), min(load_factor_array)])

            Vc_ktas = atm.get_true_airspeed(Vc)
            dynamic_pressure = 1. / 2. * atm.density * Vc_ktas ** 2.0

            for load_factor in load_factor_list:

                for x_cg in cg_array:
                    cl_wing, _, _, _ = AircraftEquilibrium.found_cl_repartition(inputs, load_factor, mass,
                                                                                   dynamic_pressure, False, x_cg)
                    cl_s_actual = cl_s * cl_wing / cl_0
                    lift_section = FoS * dynamic_pressure * cl_s_actual
                    weight_array = weight_array_orig * FoS * load_factor

                    tot_shear_diagram = AerostructuralLoad.compute_shear_diagram(y_vector, weight_array+lift_section)
                    tot_bending_moment_diagram = AerostructuralLoad.compute_bending_moment_diagram(
                        y_vector, weight_array+lift_section)
                    root_shear_force = tot_shear_diagram[0]
                    root_bending_moment = tot_bending_moment_diagram[0]

                    if abs(root_shear_force) > shear_max:
                        shear_max_conditions = [mass, load_factor, x_cg]
                        lift_shear_diagram = AerostructuralLoad.compute_shear_diagram(y_vector, lift_section)
                        weight_shear_diagram = AerostructuralLoad.compute_shear_diagram(y_vector, weight_array)
                        shear_max = abs(root_shear_force)

                    if abs(root_bending_moment) > rbm_max:
                        rbm_max_conditions = [mass, load_factor, x_cg]
                        lift_bending_diagram = AerostructuralLoad.compute_bending_moment_diagram(y_vector, lift_section)
                        weight_bending_diagram = AerostructuralLoad.compute_bending_moment_diagram(
                            y_vector, weight_array)
                        rbm_max = abs(root_bending_moment)

        additional_zeros = np.zeros(SPAN_MESH_POINT_LOADS - len(y_vector))
        lift_shear_diagram = np.concatenate([lift_shear_diagram, additional_zeros])
        weight_shear_diagram = np.concatenate([weight_shear_diagram, additional_zeros])

        lift_bending_diagram = np.concatenate([lift_bending_diagram, additional_zeros])
        weight_bending_diagram = np.concatenate([weight_bending_diagram, additional_zeros])

        outputs["data:loads:max_shear:mass"] = shear_max_conditions[0]
        outputs["data:loads:max_shear:load_factor"] = shear_max_conditions[1]
        outputs["data:loads:max_shear:cg_position"] = shear_max_conditions[2]
        outputs["data:loads:max_shear:lift_shear"] = lift_shear_diagram
        outputs["data:loads:max_shear:weight_shear"] = weight_shear_diagram

        outputs["data:loads:max_rbm:mass"] = rbm_max_conditions[0]
        outputs["data:loads:max_rbm:load_factor"] = rbm_max_conditions[1]
        outputs["data:loads:max_rbm:cg_position"] = rbm_max_conditions[2]
        outputs["data:loads:max_rbm:lift_rbm"] = lift_bending_diagram
        outputs["data:loads:max_rbm:weight_rbm"] = weight_bending_diagram

    @staticmethod
    def compute_shear_diagram(y_vector, force_array):

        shear_force_diagram = np.zeros(len(y_vector))

        for i in range(len(y_vector)):
            shear_force_diagram[i] = trapz(force_array[i:], y_vector[i:])

        return shear_force_diagram

    @staticmethod
    def compute_bending_moment_diagram(y_vector, force_array):

        bending_moment_diagram = np.zeros(len(y_vector))
        for i in range(len(y_vector)):
            lever_arm = y_vector - y_vector[i]
            test = lever_arm[i:]
            bending_moment_diagram[i] = trapz(force_array[i:] * lever_arm[i:], y_vector[i:])

        return bending_moment_diagram

    @staticmethod
    def compute_Cl_S(y_vector_orig, y_vector, cl_list, chord_list):

        cl_inter = interp1d(y_vector_orig, cl_list)
        chord_inter = interp1d(y_vector_orig, chord_list)
        cl_fin = cl_inter(y_vector)
        chord_fin = chord_inter(y_vector)
        lift_chord = np.multiply(cl_fin, chord_fin)

        return lift_chord

    @staticmethod
    def compute_relief_force(inputs, y_vector, wing_mass, fuel_mass):

        # Recuperating the data necessary for the computation

        tot_engine_mass = inputs["data:weight:propulsion:engine:mass"]
        tot_lg_mass = inputs["data:weight:airframe:landing_gear:main:mass"]
        z_cg = inputs["data:weight:aircraft_empty:CG:z"]

        lg_height = inputs["data:geometry:landing_gear:height"]
        engine_config = inputs["data:geometry:propulsion:layout"]
        engine_count = inputs["data:geometry:propulsion:count"]
        semi_span = inputs["data:geometry:wing:span"] / 2.0
        if engine_config != 1.0:
            y_ratio = 0.0
        else:
            y_ratio = inputs["data:geometry:propulsion:y_ratio"]

        g = 9.81

        # Computing the mass of the components for one wing
        single_engine_mass = tot_engine_mass / engine_count
        single_lg_mass = tot_lg_mass / 2.0  # We assume 2 MLG

        # Before computing the continued weight distribution we first take care of the point masses and modify the
        # y_vector accordingly

        # We create the array that will store the "point mass" which we chose to represent as distributed mass over a
        # small finite interval
        point_mass_array = np.zeros(len(y_vector))

        # Adding the motor weight
        if engine_config == 1.0:
            y_eng = y_ratio * semi_span
            y_vector, point_mass_array = AerostructuralLoad.add_point_mass(
                y_vector, point_mass_array, y_eng, single_engine_mass, inputs)

        # Computing and adding the lg weight
        # Overturn angle set as a fixed value, it is recommended to take over 25° and check that we can fit both LG in
        # the fuselage
        phi_ot = 35. * np.pi / 180.
        y_lg_1 = math.tan(phi_ot) * z_cg
        y_lg = max(y_lg_1, lg_height)

        y_vector, point_mass_array = AerostructuralLoad.add_point_mass(
            y_vector, point_mass_array, y_lg, single_lg_mass, inputs)

        # We can now choose what type of mass distribution we want for the mass and the fuel
        Y = y_vector / semi_span
        struct_weight_distribution = 4. / np.pi * np.sqrt(1. - Y ** 2.0)
        reajust_struct = trapz(struct_weight_distribution, y_vector)

        fuel_weight_distribution = 4. / np.pi * np.sqrt(1. - Y ** 2.0)
        reajust_fuel = trapz(fuel_weight_distribution, y_vector)

        wing_mass_array = wing_mass * struct_weight_distribution / (2. * reajust_struct)
        fuel_mass_array = fuel_mass * struct_weight_distribution / (2. * reajust_fuel)

        mass_array = wing_mass_array + fuel_mass_array + point_mass_array
        weight_array = - mass_array * g

        return y_vector, weight_array

    @staticmethod
    def insert_in_sorted_array(array, element):

        tmp_array = np.append(array, element)
        final_array = np.sort(tmp_array)
        index = np.where(final_array == element)

        return final_array, index

    @staticmethod
    def delete_additional_zeros(array):

        last_zero = np.amax(np.where(array != 0.)) + 1
        final_array = array[:int(last_zero)]

        return final_array

    @staticmethod
    def add_point_mass(y_vector, point_mass_array, y_point_mass, point_mass, inputs):

        semi_span = float(inputs["data:geometry:wing:span"]) / 2.0
        fake_point_mass_array = np.zeros(len(point_mass_array))
        present_mass_interp = interp1d(y_vector, point_mass_array)

        interval_len = POINT_MASS_SPAN_RATIO * semi_span / NB_POINTS_POINT_MASS
        nb_point_side = (NB_POINTS_POINT_MASS - 1.) / 2.
        y_added = []

        for i in range(NB_POINTS_POINT_MASS):
            y_current = y_point_mass + (i - nb_point_side) * interval_len
            if (y_current >= 0.0) and (y_current <= semi_span):
                y_added.append(y_current)
                y_vector, idx = AerostructuralLoad.insert_in_sorted_array(y_vector, y_current)
                index = int(float(idx[0]))
                point_mass_array = np.insert(point_mass_array, index, present_mass_interp(y_current))
                fake_point_mass_array = np.insert(fake_point_mass_array, index, 0.)

        y_min = min(y_added) - 1e-3
        y_vector, idx = AerostructuralLoad.insert_in_sorted_array(y_vector, y_min)
        index = int(float(idx[0]))
        point_mass_array = np.insert(point_mass_array, index, present_mass_interp(y_min))
        fake_point_mass_array = np.insert(fake_point_mass_array, index, 0.)

        y_max = max(y_added) + 1e-3
        y_vector, idx = AerostructuralLoad.insert_in_sorted_array(y_vector, y_max)
        index = int(float(idx[0]))
        point_mass_array = np.insert(point_mass_array, index, present_mass_interp(y_max))
        fake_point_mass_array = np.insert(fake_point_mass_array, index, 0.)

        where_add_mass_grt = np.greater_equal(y_vector, min(y_added))
        where_add_mass_lss = np.less_equal(y_vector, max(y_added))
        where_add_mass = np.logical_and(where_add_mass_grt, where_add_mass_lss)
        where_add_mass_index = np.where(where_add_mass)

        for idx in where_add_mass_index:
            fake_point_mass_array[idx] = 1.0

        reajust = trapz(fake_point_mass_array, y_vector)

        for idx in where_add_mass_index:
            point_mass_array[idx] += point_mass / reajust

        test = trapz(point_mass_array, y_vector)

        y_vector_new = y_vector
        point_mass_array_new = point_mass_array

        return y_vector_new, point_mass_array_new
