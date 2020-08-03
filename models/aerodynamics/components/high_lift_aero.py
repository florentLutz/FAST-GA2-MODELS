"""
Computation of lift and drag increment due to high-lift devices
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
import math
import openmdao.api as om
from importlib_resources import open_text
from scipy import interpolate

from . import resources

LIFT_EFFECTIVENESS_FILENAME = "interpolation of lift effectiveness.txt"


class ComputeDeltaHighLift(om.ExplicitComponent):
    """
    Provides lift and drag increments due to high-lift devices
    """

    def initialize(self):
        self.options.declare("landing_flag", default=False, types=bool)

    def setup(self):

        if self.options["landing_flag"]:
            self.add_input("data:mission:sizing:landing:flap_angle", val=np.nan, units="deg")
            self.add_input("data:mission:sizing:landing:slat_angle", val=np.nan, units="deg")
            self.add_input("data:aerodynamics:aircraft:landing:mach", val=np.nan)
            self.add_output("data:aerodynamics:high_lift_devices:landing:CL")
        else:
            self.add_input("data:mission:sizing:takeoff:flap_angle", val=np.nan, units="deg")
            self.add_input("data:mission:sizing:takeoff:slat_angle", val=np.nan, units="deg")
            self.add_input("data:aerodynamics:aircraft:takeoff:mach", val=np.nan)
            self.add_output("data:aerodynamics:high_lift_devices:takeoff:CL")
            self.add_output("data:aerodynamics:high_lift_devices:takeoff:CD")

        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:flap:chord_ratio", val=np.nan)
        self.add_input("data:geometry:flap:span_ratio", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:takeoff:CL_alpha", val=np.nan)
        self.add_input("configuration:flap_type", val=np.nan)

        self.declare_partials("*", "*", method="fd") # FIXME: define derivative without flap_type

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        if self.options["landing_flag"]:
            flap_angle = inputs["data:mission:sizing:landing:flap_angle"]
            slat_angle = inputs["data:mission:sizing:landing:slat_angle"]
            mach = inputs["data:aerodynamics:aircraft:landing:mach"]
        else:
            flap_angle = inputs["data:mission:sizing:takeoff:flap_angle"]
            slat_angle = inputs["data:mission:sizing:takeoff:slat_angle"]
            mach = inputs["data:aerodynamics:aircraft:takeoff:mach"]

        if self.options["landing_flag"]:
            outputs["data:aerodynamics:high_lift_devices:landing:CL"] = self._get_delta_cl(
                inputs,
                slat_angle,
                flap_angle,
                mach,
            )
        else:
            outputs["data:aerodynamics:high_lift_devices:takeoff:CL"] = self._get_delta_cl(
                inputs,
                slat_angle,
                flap_angle,
                mach,
            )
            outputs["data:aerodynamics:high_lift_devices:takeoff:CD"] = self._get_delta_cd(
                inputs,
                slat_angle, 
                flap_angle,
            )

    def _get_delta_cl(self, inputs, slat_angle, flap_angle, mach):
        """
        Method based on Roskam book and Raymer book

        :param slat_angle: in degrees
        :param flap_angle: in degrees
        :param mach:
        :return: increment of lift coefficient
        """

        flap_angle = np.radians(flap_angle)
        slat_angle = np.radians(slat_angle)

        # Compute flap lift coefficient
        if not(self.options["landing_flag"]) and (flap_angle != 30.0):
            delta_cl_flap = self._compute_delta_cl_flaps(inputs, flap_angle, mach)
        else:
            delta_cl_flap = self._compute_delta_clmax_flaps(inputs)
        # Compute slat lift coefficient
        delta_cl_slat = 0.0
        # Sum results
        delta_cl_total = delta_cl_flap + delta_cl_slat

        return delta_cl_total

    def _get_delta_cd(self, inputs, slat_angle, flap_angle):
        """

        :param slat_angle: in degrees
        :param flap_angle: in degrees
        :return: increment of drag coefficient
        """

        cd0_flap = self._compute_delta_cd_flaps(inputs, flap_angle)
        cd0_slat = 0.0 # !!!: no model for increased drag by slats
        total_cd0 = cd0_flap + cd0_slat

        return total_cd0
    
    def _compute_delta_cl_flaps(self, inputs, flap_angle, mach):
        """

        Method based on Roskam vol6 book and Raymer book
        """
        
        flap_span_ratio = inputs['ata:geometry:flap:span_ratio']
        flap_chord_ratio = inputs['data:geometry:flap:chord_ratio']
        flap_type = inputs['configuration:flap_type']
        cl_alpha_wing = inputs['data:aerodynamics:aircraft:takeoff:CL_alpha']
        
        if flap_type == 1: # slotted flap
            alpha_flap = self._compute_alpha_flap(flap_angle, flap_chord_ratio)
            delta_cl_airfoil = 2*math.pi / math.sqrt(1 - mach**2)* alpha_flap * (flap_angle / 180 * math.pi)
        else: # plain flap
            cldelta_theory = 4.1 # Fig 8.14, for t/c=0.12 and cf/c=0.25
            if flap_angle<=13: # Fig 8.13. Quick linear approximation of graph (not intended to be used)
                k = 1.0 
            elif flap_angle<=20:
                k = 0.83
            elif flap_angle<=30:
                k = 0.83 - 0.018* (flap_angle-20)
            else:
                k = 0.65 - 0.008 * (flap_angle - 30) 
            delta_cl_airfoil = cldelta_theory * k * (flap_angle / 180 * math.pi)
        # Roskam 3D flap parameters
        kb = 1.25*flap_span_ratio # !!!: PROVISIONAL (fig 8.52)
        effect  = 1.04 # Fig 8.53 (cf/c=0.25, small effect of AR)
        delta_cl_flap = kb * delta_cl_airfoil * (cl_alpha_wing/(2*math.pi)) * effect

        return delta_cl_flap
        
        
    def _compute_delta_clmax_flaps(self, inputs):
        """
        
        Method from Roskam vol.6.  Particularised for single slotted flaps in 
        airfoils with 12% thickness (which is the design case); with
        chord ratio of 0.25 and typical flap deflections (30deg landing, 10deg TO).
        Plain flap included (40 deg landing deflection here)
        """

        flap_type = inputs['configuration:flap_type']
        flap_area_ratio = self._compute_flap_area_ratio(inputs)
        
        if flap_type == 1.0: # simple slotted
            base_increment = 1.3 # Figure 8.31
            k1 = 1.0 # Figure 8.32 (chord ratio correction)
            if self.options["landing_flag"]: # Deflection correction
                k2 = 0.87 # Figure 8.33
                k3 = 0.77 # Figure 8.34
            else: # Takeoff position
                k2 = 0.47
                k3 = 0.3
        else: # plain flap
            base_increment = 0.9 # Figure 8.31
            k1 = 1.0 # Figure 8.32 (chord ratio correction)
            if self.options["landing_flag"]: # Deflection correction
                k2 = 0.87 # Figure 8.33
                k3 = 1.0 # Figure 8.34
            else: # Takeoff position
                k2 = 0.33
                k3 = 1.0
        k_planform = 0.92
        delta_clmax_flaps = base_increment*k1*k2*k3*k_planform * flap_area_ratio
        
        return delta_clmax_flaps
    
    def _compute_delta_cd_flaps(self, inputs, flap_angle):
        """
        
        Method from Young (in Gudmunsson book; page 725)
        """
        
        flap_type = inputs['configuration:flap_type']
        flap_chord_ratio = inputs['data:geometry:flap:chord_ratio']
        flap_area_ratio = self._compute_flap_area_ratio(inputs)
        
        if flap_type == 1.0: # slotted flap
            delta_cd_flaps = (-0.01523 + 0.05145 * flap_angle - 9.53201E-4 * flap_angle**2 \
                              + 7.5972E-5 * flap_angle**3) * flap_area_ratio / 100
        else: # plain flap
            k1 = - 21.09 * flap_chord_ratio**3 + 14.091 * flap_chord_ratio**2 \
                 + 3.165 * flap_chord_ratio - 0.00103
            k2 = -3.795E-7 * flap_angle**3 + 5.387E-5 * flap_angle**2 \
                 + 6.843E-4 * flap_angle - 1.4729E-3
            delta_cd_flaps = k1 * k2 * flap_area_ratio   
            
        return delta_cd_flaps
        
    def _compute_flap_area_ratio(self, inputs):
        
        wing_span = inputs["data:geometry:wing:span"]
        wing_area = inputs["data:geometry:wing:area"]
        wing_taper_ratio = inputs['data:geometry:wing:taper_ratio']
        y1_wing = inputs['data:geometry:fuselage:maximum_width']/2.0
        y2_wing = inputs['data:geometry:wing:root:y']
        wing_root_chord = inputs['data:geometry:wing:root:chord']
        flap_span_ratio = inputs['ata:geometry:flap:span_ratio']
        
        flap_area = (y2_wing-y1_wing)*wing_root_chord + \
                    flap_span_ratio*(wing_span/2.-y2_wing) * \
                    (wing_root_chord * (2 - (1-wing_taper_ratio)*flap_span_ratio))*0.5
                    
        flap_area_ratio = 2*flap_area / wing_area
        
        return flap_area_ratio
        

    def _compute_alpha_flap(self, flap_angle, ratio_cf_flap):
        temp_array = []
        with open_text(resources, LIFT_EFFECTIVENESS_FILENAME) as fichier:
            for line in fichier:
                temp_array.append([float(x) for x in line.split(",")])
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        x3 = []
        y3 = []
        x4 = []
        y4 = []
        x5 = []
        y5 = []

        for arr in temp_array:
            x1.append(arr[0])
            y1.append(arr[1])
            x2.append(arr[2])
            y2.append(arr[3])
            x3.append(arr[4])
            y3.append(arr[5])
            x4.append(arr[6])
            y4.append(arr[7])
            x5.append(arr[8])
            y5.append(arr[9])

        tck1 = interpolate.splrep(x1, y1, s=0)
        tck2 = interpolate.splrep(x2, y2, s=0)
        tck3 = interpolate.splrep(x3, y3, s=0)
        tck4 = interpolate.splrep(x4, y4, s=0)
        tck5 = interpolate.splrep(x5, y5, s=0)
        ynew1 = interpolate.splev(flap_angle, tck1, der=0)
        ynew2 = interpolate.splev(flap_angle, tck2, der=0)
        ynew3 = interpolate.splev(flap_angle, tck3, der=0)
        ynew4 = interpolate.splev(flap_angle, tck4, der=0)
        ynew5 = interpolate.splev(flap_angle, tck5, der=0)
        zs = [0.15, 0.20, 0.25, 0.30, 0.40]
        y_final = [ynew1, ynew2, ynew3, ynew4, ynew5]
        tck6 = interpolate.splrep(zs, y_final, s=0)
        return interpolate.splev(ratio_cf_flap, tck6, der=0)
