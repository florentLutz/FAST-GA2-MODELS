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
from pandas import read_csv
from importlib_resources import open_text
from typing import Union, Tuple, Optional
from scipy import interpolate
from ..constants import ELEV_POINT_COUNT

from . import resources

LIFT_EFFECTIVENESS_FILENAME = "interpolation of lift effectiveness.txt"
DELTA_CL_PLAIN_FLAP = "delta lift plain flap.csv"
K_PLAIN_FLAP = "k plain flap.csv"
KB_FLAPS = "kb flaps.csv"
ELEVATOR_ANGLE_LIST = np.linspace(-25.0, 25.0, num=ELEV_POINT_COUNT)

class ComputeDeltaHighLift(om.ExplicitComponent):
    """
    Provides lift and drag increments due to high-lift devices
    """

    def initialize(self):
        self.options.declare("landing_flag", default=False, types=bool)

    def setup(self):
        
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:flap:chord_ratio", val=np.nan)
        self.add_input("data:geometry:flap:span_ratio", val=np.nan)
        self.add_input("data:geometry:flap_type", val=np.nan)
        self.add_input("data:aerodynamics:aircraft:low_speed:CL_alpha", val=np.nan)
        self.add_input("data:aerodynamics:low_speed:mach", val=np.nan)
        self.add_input("data:mission:sizing:landing:flap_angle", val=np.nan, units="deg")
        self.add_input("data:mission:sizing:takeoff:flap_angle", val=np.nan, units="deg")
        
        self.add_output("data:aerodynamics:flaps:landing:CL")
        self.add_output("data:aerodynamics:flaps:landing:CM")
        self.add_output("data:aerodynamics:flaps:landing:CD")
        self.add_output("data:aerodynamics:flaps:takeoff:CL")
        self.add_output("data:aerodynamics:flaps:takeoff:CM")
        self.add_output("data:aerodynamics:flaps:takeoff:CD")
        self.add_output("data:aerodynamics:elevator:low_speed:angle", shape=len(ELEVATOR_ANGLE_LIST), units="deg")
        self.add_output("data:aerodynamics:elevator:low_speed:CL", shape=len(ELEVATOR_ANGLE_LIST))

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        
        mach = inputs["data:aerodynamics:low_speed:mach"]
        
        # Computes flaps contribution during low speed operations (take-off/landing)
        for self.phase in ['landing', 'takeoff']:
            if self.phase == 'landing':
                flap_angle = inputs["data:mission:sizing:landing:flap_angle"]
                outputs["data:aerodynamics:flaps:landing:CL"] = self._get_flaps_delta_cl(
                    inputs,
                    flap_angle,
                    mach,
                )
                outputs["data:aerodynamics:flaps:landing:CM"] = self._get_flaps_delta_cm(
                    inputs,
                    flap_angle,
                    mach,
                )
                outputs["data:aerodynamics:flaps:landing:CD"] = self._get_flaps_delta_cd(
                    inputs,
                    flap_angle,
                )
            else:
                flap_angle = inputs["data:mission:sizing:takeoff:flap_angle"]
                outputs["data:aerodynamics:flaps:takeoff:CL"] = self._get_flaps_delta_cl(
                    inputs,
                    flap_angle,
                    mach,
                )
                outputs["data:aerodynamics:flaps:takeoff:CM"] = self._get_flaps_delta_cm(
                    inputs,
                    flap_angle,
                    mach,
                )
                outputs["data:aerodynamics:flaps:takeoff:CD"] = self._get_flaps_delta_cd(
                    inputs,
                    flap_angle,
                )
        
        
        # Computes elevator contribution during low speed operations (for different deflection angle)
        outputs["data:aerodynamics:elevator:low_speed:angle"] = ELEVATOR_ANGLE_LIST
        outputs["data:aerodynamics:elevator:low_speed:CL"] = self._get_elevator_delta_cl(
                inputs,
                ELEVATOR_ANGLE_LIST,
            )
            
    def _get_elevator_delta_cl(self, inputs, elevator_angle: Union[float, np.array]) -> Union[float, np.array]:
        """
        Applies the plain flap lift variation function :meth:`_delta_lift_plainflap`.

        :param elevator_angle: elevator angle (in Degree)
        :return: increment of lift coefficient
        """
        
        ht_area = inputs["data:geometry:horizontal_tail:area"]
        wing_area = inputs["data:geometry:wing:area"]

        # Elevator (plain flap). Default: maximum deflection (25deg)
        cl_delta_theory, k = self._delta_lift_plainflap(abs(elevator_angle), 0.3, 0.1)
        delta_cl_elev = (cl_delta_theory * k * elevator_angle) * ht_area / wing_area
        delta_cl_elev *= 0.9  # Correction for the central fuselage part (no elevator there)
                        
        return delta_cl_elev

    def _get_flaps_delta_cl(self, inputs, flap_angle: float, mach: float) -> float:
        """
        Method based on...

        :param flap_angle: flap angle (in Degree)
        :param mach: air speed
        :return: increment of lift coefficient
        """

        cl_alpha_wing = inputs['data:aerodynamics:aircraft:low_speed:CL_alpha']
        span_wing = inputs['data:geometry:wing:span']
        y1_wing = inputs['data:geometry:fuselage:maximum_width'] / 2.0
        y2_wing = inputs['data:geometry:wing:root:y']
        flap_span_ratio = inputs['data:geometry:flap:span_ratio']

        if not(self.phase == 'landing') and (flap_angle != 30.0):
            # 2D flap lift coefficient
            delta_cl_airfoil = self._compute_delta_cl_airfoil_2D(inputs, flap_angle, mach)
            # Roskam 3D flap parameters
            eta_in = y1_wing / (span_wing/2.0)
            eta_out = ((y2_wing - y1_wing) + flap_span_ratio * (span_wing/2.0 - y2_wing)) / (span_wing/2.0 - y2_wing)
            kb = self._compute_kb_flaps(inputs, eta_in, eta_out)
            effect = 1.04  # fig 8.53 (cf/c=0.25, small effect of AR)
            delta_cl_flap = kb * delta_cl_airfoil * (cl_alpha_wing / (2 * math.pi)) * effect
        else:
            delta_cl_flap = self._compute_delta_clmax_flaps(inputs)

        return delta_cl_flap
    
    def _get_flaps_delta_cm(self, inputs, flap_angle: float, mach: float) -> float:
        """
        Method based on Roskam book

        :param flap_angle: flap angle (in Degree)
        :param mach: air speed
        :return: increment of moment coefficient
        """
        
        wing_taper_ratio = inputs['data:geometry:wing:taper_ratio']
        
        #Method from Roskam (sweep=0, flaps 60%, simple slotted and not extensible,
        #at 25% MAC, cf/c+0.25)
        k_p = interpolate.interp1d([0.,0.2,0.33,0.5,1.],[0.65,0.75,0.7,0.63,0.5])
        #k_p: Figure 8.105, interpolated function of taper ratio (span ratio fixed)
        delta_cl_flap = self._get_flaps_delta_cl(inputs, flap_angle, mach)
        delta_cm_flap = k_p(wing_taper_ratio) * (-0.27)*(delta_cl_flap) #-0.27: Figure 8.106

        return delta_cm_flap

    def _get_flaps_delta_cd(self, inputs, flap_angle: float) -> float:
        """
        Method from Young (in Gudmunsson book; page 725)
        
        :param flap_angle: flap angle (in Degree)
        :return: increment of drag coefficient
        """
        
        flap_type = inputs['data:geometry:flap_type']
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

    def _compute_delta_cl_airfoil_2D(self, inputs, angle: float, mach: float) -> float:
        """
        Compute airfoil 2D lift contribution.

        :param angle: airfoil angle (in Degree)
        :param mach: air speed
        :return: increment of lift coefficient
        """

        flap_type = inputs['data:geometry:flap_type']
        flap_chord_ratio = inputs['data:geometry:flap:chord_ratio']

        # 2D flap lift coefficient
        if flap_type == 1:  # Slotted flap
            alpha_flap = self._compute_alpha_flap(angle, flap_chord_ratio)
            delta_cl_airfoil = 2 * math.pi / math.sqrt(1 - mach ** 2) * alpha_flap * (angle * math.pi / 180)
        else:  # Plain flap
            cl_delta_theory, k = self._delta_lift_plainflap(angle, flap_chord_ratio)
            delta_cl_airfoil = cl_delta_theory * k * (angle * math.pi / 180)

        return delta_cl_airfoil

    def _compute_delta_clmax_flaps(self, inputs) -> float:
        """

        Method from Roskam vol.6.  Particularised for single slotted flaps in 
        airfoils with 12% thickness (which is the design case); with
        chord ratio of 0.25 and typical flap deflections (30deg landing, 10deg TO).
        Plain flap included (40 deg landing deflection here)
        """

        flap_type = inputs['data:geometry:flap_type']
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

    def _compute_flap_area_ratio(self, inputs) -> float:
        """
        Compute ratio of flap over wing (reference area).
        Takes into account the wing portion under the fuselage.
        """

        wing_span = inputs["data:geometry:wing:span"]
        wing_area = inputs["data:geometry:wing:area"]
        wing_taper_ratio = inputs['data:geometry:wing:taper_ratio']
        y1_wing = inputs['data:geometry:fuselage:maximum_width']/2.0
        y2_wing = inputs['data:geometry:wing:root:y']
        wing_root_chord = inputs['data:geometry:wing:root:chord']
        flap_span_ratio = inputs['data:geometry:flap:span_ratio']
        
        flap_area = (y2_wing-y1_wing)*wing_root_chord + \
                    flap_span_ratio*(wing_span/2.-y2_wing) * \
                    (wing_root_chord * (2 - (1-wing_taper_ratio)*flap_span_ratio))*0.5
                    
        flap_area_ratio = 2*flap_area / wing_area
        
        return flap_area_ratio

    def _compute_alpha_flap(self, flap_angle: float, chord_ratio: float) -> float:
        """
        Roskam data to calculate the effectiveness of a simple slotted flap.

        :param flap_angle: flap angle (in Degree)
        :param chord_ratio: position of flap on wing chord
        :return: effectiveness ratio
        """

        temp_array = []
        with open_text(resources, LIFT_EFFECTIVENESS_FILENAME) as file:
            for line in file:
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
        effectiveness = interpolate.splev(chord_ratio, tck6, der=0)

        return effectiveness

    def _delta_lift_plainflap(
            self,
            flap_angle: Union[float, np.array],
            chord_ratio : float,
            thickness: Optional[float]=0.12
    ) -> Tuple[np.array, np.array]:
        """
        Roskam data to estimate plain flap lift increment and correction factor K.

        :param flap_angle: flap angle (in Degree)
        :param chord_ratio: position of flap on wing chord
        :return: lift increment and correction factor
        """

        flap_angle = np.array(flap_angle)

        file = resources.__path__[0] + '\\' + DELTA_CL_PLAIN_FLAP
        db = read_csv(file)

        x_0 = db['X_0'].tolist()
        y_0 = db['Y_0'].tolist()
        x_04 = db['X_04'].tolist()
        y_04 = db['Y_04'].tolist()
        x_10 = db['X_10'].tolist()
        y_10 = db['Y_10'].tolist()
        x_15 = db['X_15'].tolist()
        y_15 = db['Y_15'].tolist()
        cld_thk0 = interpolate.interp1d(x_0, y_0)
        cld_thk04 = interpolate.interp1d(x_04, y_04)
        cld_thk10 = interpolate.interp1d(x_10, y_10)
        cld_thk15 = interpolate.interp1d(x_15, y_15)
        cld_t = [cld_thk0(chord_ratio), cld_thk04(chord_ratio), \
                 cld_thk10(chord_ratio), cld_thk15(chord_ratio)]
        cl_delta = interpolate.interp1d([0.0, 0.04, 0.1, 0.15], cld_t)(thickness)

        file = resources.__path__[0] + '\\' + K_PLAIN_FLAP
        db = read_csv(file)

        x_10 = db['X_10'].tolist()
        y_10 = db['Y_10'].tolist()
        x_15 = db['X_15'].tolist()
        y_15 = db['Y_15'].tolist()
        x_25 = db['X_25'].tolist()
        y_25 = db['Y_25'].tolist()
        x_30 = db['X_30'].tolist()
        y_30 = db['Y_30'].tolist()
        x_40 = db['X_40'].tolist()
        y_40 = db['Y_40'].tolist()
        x_50 = db['X_50'].tolist()
        y_50 = db['Y_50'].tolist()
        k_chord10 = interpolate.interp1d(x_10, y_10)
        k_chord15 = interpolate.interp1d(x_15, y_15)
        k_chord25 = interpolate.interp1d(x_25, y_25)
        k_chord30 = interpolate.interp1d(x_30, y_30)
        k_chord40 = interpolate.interp1d(x_40, y_40)
        k_chord50 = interpolate.interp1d(x_50, y_50)
        k = []
        for angle in list(flap_angle):
            k_chord = [k_chord10(angle), k_chord15(angle), \
                       k_chord25(angle), k_chord30(angle), \
                       k_chord40(angle), k_chord50(angle)]
            k.append(float(interpolate.interp1d([0.1, 0.15, 0.25, 0.3, 0.4, 0.5], k_chord)(chord_ratio)))
        k = np.array(k)

        return cl_delta, k

    def _compute_kb_flaps(self, inputs, eta_in: float, eta_out: float) -> float:
        """
        Use Roskam graph (Figure 8.52) to interpolate kb factor.
        This factor accounts for a finite flap contribution to the 3D lift increase, depending on its position and size
        and the taper ratio of the wing.

        :param eta_in: ????
        :param eta_out: ????
        :return: kb factor contribution to 3D lift
        """

        eta_in = float(eta_in)
        eta_out = float(eta_out)
        wing_taper_ratio = inputs['data:geometry:wing:taper_ratio']
        file = resources.__path__[0] + '\\' + KB_FLAPS
        db = read_csv(file)

        x_0 = db['X_0'].tolist()
        y_0 = db['Y_0'].tolist()
        x_05 = db['X_0.5'].tolist()
        y_05 = db['Y_0.5'].tolist()
        x_1 = db['X_1'].tolist()
        y_1 = db['Y_1'].tolist()
        k_taper0 = interpolate.interp1d(x_0, y_0)
        k_taper05 = interpolate.interp1d(x_05, y_05)
        k_taper1 = interpolate.interp1d(x_1, y_1)
        k_eta = [k_taper0(eta_in), k_taper05(eta_in), k_taper1(eta_in)]
        kb_in = interpolate.interp1d([0.0, 0.5, 1.0], k_eta)(wing_taper_ratio)
        k_eta = [k_taper0(eta_out), k_taper05(eta_out), k_taper1(eta_out)]
        kb_out = interpolate.interp1d([0.0, 0.5, 1.0], k_eta)(wing_taper_ratio)

        return float(kb_out - kb_in)