"""
    Vortex Lattice Method implementation
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
import numpy as np
import copy
import openmdao.api as om
from typing import Tuple

DEFAULT_NX = 19
DEFAULT_NY1 = 3
DEFAULT_NY2 = 14


class VLM(om.ExplicitComponent):

    def setup(self):

        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:kink:span_ratio", val=np.nan)
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:flap:span_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")

    def _run(self, inputs):

        wing_break = float(inputs["data:geometry:wing:kink:span_ratio"])

        # Define mesh size        
        self.nx = int(DEFAULT_NX)
        if wing_break > 0.0:
            self.ny1 = int(DEFAULT_NY1 + 5)  # n° of panels in the straight section of the wing
            self.ny2 = int((DEFAULT_NY2 - 5)/2)  # n° of panels in in the flapped portion of the wing
        else:
            self.ny1 = int(DEFAULT_NY1)  # n° of panels in the straight section of the wing
            self.ny2 = int(DEFAULT_NY2/2)  # n° of panels in in the flapped portion of the wing
        self.ny3 = self.ny2  # n° of panels in the un-flapped exterior portion of the wing

        self.ny = int(self.ny1 + self.ny2 + self.ny3)
        # Define elements
        self.WING = {'x_panel': np.zeros((self.nx + 1, 2 * self.ny + 1)),
                     'y_panel': np.zeros(2 * self.ny + 1),
                     'z': np.zeros(self.nx + 1),
                     'x_LE': np.zeros(2 * self.ny + 1),
                     'chord': np.zeros(2 * self.ny + 1),
                     'panel_span': np.zeros(2 * self.ny),
                     'panel_chord': np.zeros(self.nx * self.ny),
                     'panel_surf': np.zeros(self.nx * self.ny),
                     'xc': np.zeros(self.nx * 2 * self.ny),
                     'yc': np.zeros(self.nx * 2 * self.ny),
                     'x1': np.zeros(self.nx * 2 * self.ny),
                     'x2': np.zeros(self.nx * 2 * self.ny),
                     'y1': np.zeros(self.nx * 2 * self.ny),
                     'y2': np.zeros(self.nx * 2 * self.ny),
                     'panel_angle': np.zeros(self.nx),
                     'panel_angle_vect': np.zeros(self.nx * self.ny),
                     'AIC': np.zeros((self.nx * self.ny, self.nx * self.ny)),
                     'AIC_wake': np.zeros((self.nx * self.ny, self.nx * self.ny))}
        # Duplicate for HTP
        self.HTP = copy.deepcopy(self.WING)

        # Generate WING
        self._generate_wing(inputs)

        # Generate HTP
        self._generate_htp(inputs)

    def _generate_wing(self, inputs):
        """Generates the coordinates for VLM calculations and AIC matrix of the wing"""

        y2_wing = inputs['data:geometry:wing:root:y']
        semi_span = inputs['data:geometry:wing:span']/2.0
        root_chord = inputs['data:geometry:wing:root:chord']
        tip_chord = inputs['data:geometry:wing:tip:chord']
        flap_span_ratio = inputs['data:geometry:flap:span_ratio']

        # Initial data (zero matrix/array)
        y_panel = self.WING['y_panel']
        chord = self.WING['chord']
        x_LE = self.WING['x_LE']
        y_endflaps = y2_wing + flap_span_ratio * (semi_span - y2_wing)
        # Definition of x_panel, y_panel, x_LE and chord (Right side)
        for j in range(self.ny+1):
            if j < self.ny1:
                y_panel[j] = y2_wing * j/self.ny1
                chord[j] = root_chord
            elif (j >= self.ny1) and (j < (self.ny1+self.ny2)):
                y_panel[j] = y2_wing + (y_endflaps - y2_wing) * (j-self.ny1)/self.ny2
                y_tapered_section = (y_panel[j] - y2_wing)
                chord[j] = root_chord + (tip_chord-root_chord)*y_tapered_section/(semi_span-y2_wing)
                x_LE[j] = y_tapered_section * (root_chord-tip_chord)/(4*(semi_span-y2_wing))
            else:
                y_panel[j] = y_endflaps + (semi_span - y_endflaps) * (j-(self.ny1+self.ny2))/self.ny3
                y_tapered_section = (y_panel[j] - y2_wing)
                chord[j] = root_chord + (tip_chord-root_chord)*y_tapered_section/(semi_span-y2_wing)
                x_LE[j] = y_tapered_section * (root_chord-tip_chord)/(4*(semi_span-y2_wing))
        # Definition of Left side (symmetry)
        for j in range(1, self.ny+1):
            y_panel[self.ny+j] = -y_panel[j]
            chord[self.ny+j] = chord[j]
            x_LE[self.ny+j] = x_LE[j]
        # Save data
        self.WING['y_panel'] = y_panel
        self.WING['chord'] = chord
        self.WING['x_LE'] = x_LE
        # Launch common code
        self._generate_common(self.WING)

    def _generate_htp(self, inputs):
        """Generates the coordinates for VLM calculations and AIC matrix of the htp"""

        semi_span = inputs['data:geometry:horizontal_tail:span']/2.0
        root_chord = inputs['data:geometry:horizontal_tail:root:chord']
        tip_chord = inputs['data:geometry:horizontal_tail:tip:chord']

        # Initial data (zero matrix/array)
        y_panel = self.HTP['y_panel']
        chord = self.HTP['chord']
        x_LE = self.HTP['x_LE']
        # Definition of x_panel, y_panel, x_LE and chord (Right side)
        for j in range(self.ny+1):
            y_panel[j] = semi_span * j/self.ny
            chord[j] = root_chord + (tip_chord-root_chord)*y_panel[j]/semi_span
            x_LE[j] = y_panel[j] * (root_chord-tip_chord)/(4*semi_span)
        # Definition of Left side (symmetry)
        for j in range(1, self.ny+1):
            y_panel[self.ny+j] = -y_panel[j]
            chord[self.ny+j] = chord[j]
            x_LE[self.ny+j] = x_LE[j]
        # Save data
        self.HTP['y_panel'] = y_panel
        self.HTP['chord'] = chord
        self.HTP['x_LE'] = x_LE
        # Launch common code
        self._generate_common(self.HTP)

    def _generate_common(self, dictionary):
        """Common code shared between wing and htp to calculate geometry/aero parameters"""

        # Initial data (zero matrix/array)
        x_LE = dictionary['x_LE']
        chord = dictionary['chord']
        x_panel = dictionary['x_panel']
        y_panel = dictionary['y_panel']
        panelspan = dictionary['panel_span']
        panelchord = dictionary['panel_chord']
        panelsurf = dictionary['panel_surf']
        xc = dictionary['xc']
        yc = dictionary['yc']
        x1 = dictionary['x1']
        y1 = dictionary['y1']
        x2 = dictionary['x2']
        y2 = dictionary['y2']
        AIC = dictionary['AIC']
        AIC_wake = dictionary['AIC_wake']
        # Calculate panel corners x-coordinate
        for i in range(self.nx+1):
            for j in range(2*self.ny+1):
                x_panel[i, j] = x_LE[j] + chord[j] * i/self.nx
        # Calculate panel span with symmetry
        for j in range(self.ny):
            panelspan[j] = y_panel[j+1] - y_panel[j]
            panelspan[self.ny+j] = panelspan[j]
        # Calculate characteristic points (Right side)
        for i in range(self.nx):
            for j in range(self.ny):
                panelchord[i*self.ny+j] = 0.5 * (
                        (x_panel[i+1, j] - x_panel[i, j])
                        + (x_panel[i+1, j+1] - x_panel[i, j+1])
                )
                panelsurf[i*self.ny+j] = panelspan[j] * panelchord[i*self.ny+j]
                xc[i*self.ny+j] = (x_panel[i, j] + x_panel[i, j+1])*0.5 \
                                  + 0.75*panelchord[i*self.ny+j]
                yc[i*self.ny+j] = (y_panel[j] + y_panel[j+1])*0.5
                x1[i*self.ny+j] = x_panel[i, j] + 0.25*(x_panel[i+1, j] - x_panel[i, j])
                y1[i*self.ny+j] = y_panel[j]
                x2[i*self.ny+j] = x_panel[i, j+1] + 0.25*(x_panel[i+1, j+1] - x_panel[i, j+1])
                y2[i*self.ny+j] = y_panel[j+1]
        # Calculate characteristic points (Left side)
        for i in range(self.nx):
            for j in range(self.ny):
                xc[self.nx*self.ny+(i*self.ny+j)] = xc[i*self.ny+j]
                yc[self.nx*self.ny+(i*self.ny+j)] = -yc[i*self.ny+j]
                x1[self.nx*self.ny+(i*self.ny+j)] = (
                        x_panel[i, self.ny+j+1]
                        + 0.25*(x_panel[i+1, self.ny+j+1] - x_panel[i, self.ny+j+1])
                )
                y1[self.nx*self.ny+(i*self.ny+j)] = y_panel[self.ny+j+1]
                if j == 0:
                    y2[self.nx*self.ny+(i*self.ny+j)] = 0
                    x2[self.nx*self.ny+(i*self.ny+j)] = x_panel[i, 0] + 0.25 * (x_panel[i+1, 0] - x_panel[i, 0])
                else:
                    x2[self.nx*self.ny+(i*self.ny+j)] = x_panel[i, self.ny+j] \
                                                        + 0.25*(x_panel[i+1, self.ny+j] - x_panel[i, self.ny+j])
                    y2[self.nx*self.ny+(i*self.ny+j)] = y_panel[self.ny+j]
        # Aerodynamic coefficients computation (Right side)
        for i in range(self.nx*self.ny):
            for j in range(self.nx*self.ny):
                # Right wing
                a = xc[i] - x1[j]
                b = yc[i] - y1[j]
                c = xc[i] - x2[j]
                d = yc[i] - y2[j]
                e = math.sqrt(a**2 + b**2)
                f = math.sqrt(c**2 + d**2)
                g = x2[j] - x1[j]
                h = y2[j] - y1[j]
                k = (g*a + h*b)/e - (g*c + h*d)/f
                m = (1 + c/f)/d - (1 + a/e)/b
                if a*d-b*c != 0:
                    AIC[i, j] = (k/(a*d-b*c)) / (4*math.pi)
                AIC_wake[i, j] = m / (4*math.pi)
                AIC[i, j] = AIC[i, j] + m / (4*math.pi)
        # Aerodynamic coefficients computation (Left side)
        for i in range(self.nx*self.ny):
            for j in range(self.nx*self.ny):
                # Left wing
                a = xc[i] - x1[self.nx*self.ny+j]
                b = yc[i] - y1[self.nx*self.ny+j]
                c = xc[i] - x2[self.nx*self.ny+j]
                d = yc[i] - y2[self.nx*self.ny+j]
                e = math.sqrt(a**2 + b**2)
                f = math.sqrt(c**2 + d**2)
                g = x2[self.nx*self.ny+j] - x1[self.nx*self.ny+j]
                h = y2[self.nx*self.ny+j] - y1[self.nx*self.ny+j]
                k = (g*a + h*b)/e - (g*c + h*d)/f
                m = (1 + c/f)/d - (1 + a/e)/b
                if a*d-b*c != 0:
                    AIC[i, j] = AIC[i, j] + (k/(a*d-b*c)) / (4*math.pi)
                AIC_wake[i, j] = AIC_wake[i, j] + m / (4*math.pi)
                AIC[i, j] = AIC[i, j] + m / (4*math.pi)
        # Save data
        dictionary['x_panel'] = x_panel
        dictionary['panel_span'] = panelspan
        dictionary['panel_chord'] = panelchord
        dictionary['panel_surf'] = panelsurf
        dictionary['xc'] = xc
        dictionary['yc'] = yc
        dictionary['x1'] = x1
        dictionary['y1'] = y1
        dictionary['x2'] = x2
        dictionary['y2'] = y2
        dictionary['AIC'] = AIC
        dictionary['AIC_wake'] = AIC_wake

    def compute_wing(self, inputs, aoalist, vinf, flaps_angle=0.0,
                     use_airfoil=True) -> Tuple[list, list, list, list]:
        """
        VLM computations for the wing alone.

        :param inputs: inputs parameters for the explicit component
        :param aoalist: list of angle of attack to be computed (in Deg)
        :param vinf: air speed (in m/s)
        :param flaps_angle: (in Deg)
        :param use_airfoil: adds the camberline coordinates of the selected airfoil (NACA 230) (default=True)
        :return: [Cl, Cdi, Oswald, Cm] aerodynamic parameters
        """

        aspect_ratio = inputs['data:geometry:wing:aspect_ratio']
        meanchord = inputs['data:geometry:wing:MAC:length']

        # Initialization        
        Cl = []
        Cdi = []
        Oswald = []
        Cm = []
        xc = self.WING['xc']
        panelchord = self.WING['panel_chord']
        panelsurf = self.WING['panel_surf']
        if use_airfoil:
            self.read_af_file()
        panelangle_vect = self.WING['panel_angle_vect']
        AIC = self.WING['AIC']
        AIC_inv = np.linalg.inv(AIC)
        AIC_wake = self.WING['AIC_wake']
        self.flapped_airfoil(inputs, flaps_angle)

        # Calculate all the aerodynamic parameters 
        for AoA in aoalist:
            AoA = AoA*math.pi/180
            alpha = np.add(panelangle_vect, AoA)
            gamma = -np.dot(AIC_inv, alpha) * vinf
            cp = -2 / vinf * np.divide(gamma, panelchord)
            for i in range(self.nx):
                cp[i*self.ny] = cp[i*self.ny] * 1
            cl = -np.sum(cp*panelsurf)/np.sum(panelsurf)
            alphaind = np.dot(AIC_wake, gamma) / vinf
            cdind_panel = cp*alphaind
            cdi = np.sum(cdind_panel*panelsurf)/np.sum(panelsurf)
            oswald = cl**2/(math.pi*aspect_ratio*cdi) * 0.955  # !!!: manual correction?
            cmpanel = np.multiply(cp, (xc[:self.nx*self.ny]-meanchord/4))
            cm = np.sum(cmpanel*panelsurf)/np.sum(panelsurf)
            # Save data
            Cl.append(cl)
            Cdi.append(cdi)
            Oswald.append(oswald)
            Cm.append(cm)

        return Cl, Cdi, Oswald, Cm

    def compute_htp(self, aoalist: list, vinf: float) -> list:
        """VLM computation for the horizontal tail."""

        # Initialization 
        Cl = []
        panelchord = self.HTP['panel_chord']
        panelsurf = self.HTP['panelsurf']
        AIC = self.HTP['AIC']
        AIC_inv = np.linalg.inv(AIC)
        if 'w_farfield' in self.WING.keys():
            w_farfield = self.WING['w_farfield']
            # Calculate aerodynamic parameters
            for i in range(len(aoalist)):
                wi_ht_vect = self.interpolate_w_ht(w_farfield[i])
                alpha = np.ones(self.nx*self.ny) * aoalist[i] * math.pi / 180 + wi_ht_vect / vinf
                gamma = -np.dot(AIC_inv, alpha) * vinf
                cp = -2 / vinf * np.divide(gamma, panelchord)
                cl = -np.sum(cp*panelsurf)/np.sum(panelsurf)
                Cl.append(cl)

        return Cl

    def interpolate_w_ht(self, wi_wing: np.array) -> np.array:
        """Interpolates the down-wash velocity to the HT control points."""

        yc_ht = self.HTP['yc']
        yc_wing = self.WING['yc']

        wi_ht = np.zeros(self.ny)
        wi_ht_vect = np.zeros(self.nx*self.ny)
        for htpanel in range(self.ny):
            for wingpanel in range(self.ny):
                if yc_ht[htpanel] > yc_wing[wingpanel]:
                    wi_ht[htpanel] = wi_wing[wingpanel-1] \
                                     + (wi_wing[wingpanel] - wi_wing[wingpanel-1]) \
                                     / (yc_wing[wingpanel] - yc_wing[wingpanel-1]) \
                                     * (yc_ht[htpanel] - yc_wing[wingpanel-1])
                    break
        for i in range(self.nx):
            for j in range(self.ny):
                wi_ht_vect[i*self.nx+j] = wi_ht[j]

        return wi_ht_vect

    def get_cl_curve(self, aoa: float, vinf: float) -> Tuple[list, list]:
        """
        Get wing Cl at y position.

        :param aoa: angle of attack to be computed (in Deg)
        :param vinf: air speed (in m/s)
        :return: [y_position, cl_curve]
        """

        yc_wing = self.WING['yc']
        chord_wing = self.WING['chord']
        panelangle_vect = self.WING['panel_angle_vect']
        panelchord = self.WING['panel_chord']
        AIC = self.WING['AIC']
        AIC_inv = np.linalg.inv(AIC)
        aoa = aoa * math.pi / 180
        alpha = np.add(panelangle_vect, aoa)
        gamma = -np.dot(AIC_inv, alpha) * vinf
        cp = -2 / vinf * np.divide(gamma, panelchord)
        cl_curve = []
        y_position = []
        for j in range(self.ny):
            cl_span = 0.0
            y = yc_wing[j]
            chord = (chord_wing[j] + chord_wing[j+1]) / 2.0
            for i in range(self.nx):
                cl_span += -cp[i*self.ny + j] * panelchord[i*self.ny + j] / chord
            cl_curve.append(cl_span)
            y_position.append(y)

        return y_position, cl_curve


    def read_af_file(self):
        """Generates curvature corresponding to the airfoil contained in .af file"""

        x_panel = self.WING['x_panel']
        panelangle_vect = self.WING['panel_angle_vect']
        panelangle = self.WING['panel_angle']

        with open("D://a.reysset//Documents//Github//FAST-GA2-MODELS//models//aerodynamics//resources//naca23012.af", 'r') as lf:
            data = lf.readlines()
            # Extract data
            x_data = []
            z_data = []
            for i in range(len(data)):
                line = data[i].split()
                if len(line) == 2:
                    # noinspection PyBroadException
                    try:
                        float(line[0])
                        float(line[1])
                        x_data.append(float(line[0]))
                        z_data.append(float(line[1]))
                    except:
                        pass

        # Differentiate upper and lower curves
        x_data = np.array(x_data)
        z_data = np.array(z_data)
        idx = int(np.where(x_data[0:len(x_data)-2] > x_data[1:len(x_data)-1])[0])
        x_1 = x_data[0:idx+1]
        z_1 = z_data[0:idx+1]
        x_2 = x_data[idx+1:len(x_data)]
        z_2 = z_data[idx+1:len(x_data)]

        # Initialization
        z = np.zeros(self.nx + 1)
        rootchord = x_panel[self.nx, 0] - x_panel[0, 0]
        # Calculation of panelangle_vect
        for i in range(self.nx + 1):
            xred = (x_panel[i, 0] - x_panel[0, 0]) / rootchord
            z_1_interp = np.interp(xred, x_1, z_1)
            z_2_interp = np.interp(xred, x_2, z_2)
            z[i] = (z_1_interp + z_2_interp) / 2.0
        z = z * rootchord
        for i in range(self.nx):
            panelangle[i] = (z[i] - z[i+1]) / (x_panel[i+1, 0] - x_panel[i, 0])

        for i in range(self.nx):
            for j in range(self.ny):
                panelangle_vect[i*self.ny+j] = panelangle[i]

        # Save results
        self.WING['panel_angle_vect'] = panelangle_vect
        self.WING['panel_angle'] = panelangle
        self.WING['z'] = z


    def flapped_airfoil(self, inputs, deflection_angle):

        root_chord = inputs['data:geometry:wing:root:chord']
        x_start = (1.0 - inputs['data:geometry:flap:span_ratio'])*root_chord
        y1_wing = inputs['data:geometry:fuselage:maximum_width']/2.0

        deflection_angle *= math.pi/180  # converted to radian
        z_ = self.WING['z']
        x_panel = self.WING['x_panel']
        y_panel = self.WING['y_panel']
        panelangle = self.WING['panel_angle']
        panelangle_vect = self.WING['panel_angle_vect']

        z = np.zeros(self.nx+1)
        for i in range(self.nx+1):
            if x_panel[i, 0] > x_start:
                z[i] = z_[i] - math.sin(deflection_angle) * (x_panel[i, 0]-x_start)
        for i in range(self.nx):
            panelangle[i] = (z[i] - z[i+1]) / (x_panel[i+1, 0] - x_panel[i, 0])
        for j in range(self.ny1):
            if y_panel[j] > y1_wing:
                for i in range(self.nx):
                    panelangle_vect[i*self.ny+j] += panelangle[i]
        for j in range(self.ny1, self.ny1+self.ny2):
            for i in range(self.nx):
                panelangle_vect[i*self.ny+j] += panelangle[i]

        # Save results
        self.WING['panel_angle_vect'] = panelangle_vect
        self.WING['panel_angle'] = panelangle
        self.WING['z'] = z
