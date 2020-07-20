"""
This module launches OPENVSP computations
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

import multiprocessing
import os
import os.path as pth
import win32event
import win32process
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory

import math
import numpy as np
from fastoad.utils.physics import Atmosphere
from fastoad.models.aerodynamics.external.openvsp import openvsp3201
from fastoad.utils.resource_management.copy import copy_resource
from openmdao.components.external_code_comp import ExternalCodeComp

OPTION_VSPSCRIPT_EXE_PATH = "vspscript_exe_path"
OPTION_VSPAERO_EXE_PATH = "vspaero_exe_path"

_RESULT_FILE_NAME = "calc_result.txt"
_STDOUT_SCRIPT_FILE_NAME = "openvsp.vspscript"
_STDOUT_AERO_FILE_NAME = "openvsp.vspaero"
VSPSCRIPT_EXE_NAME = "vspscript.exe"
VSPAERO_EXE_NAME = "vspaero.exe"

class OpenVSP(ExternalCodeComp):

    def initialize(self):
        
        self.options.declare(OPTION_VSPSCRIPT_EXE_PATH, default="", types=str, allow_none=True)
        self.options.declare(OPTION_VSPAERO_EXE_PATH, default="", types=str, allow_none=True)
        self.options.declare("function_exe", types=str, allow_none=False)
        
    def setup(self):

        if self.options["function_exe"] == "stability":
            use_fuselage = True
            use_horizontal_tail = True
            use_definition_angle = True
            self.add_output("openvsp:cm_alpha")
        elif self.options["function_exe"] == "inducedDrag":
            use_fuselage = False
            use_horizontal_tail = False
            use_definition_angle = False
            self.add_output("openvsp:oswald")
        elif self.options["function_exe"] == "cl_alpha_wing":
            use_fuselage = False
            use_horizontal_tail = False
            use_definition_angle = False
            self.add_output("openvsp:cl_0")
            self.add_output("openvsp:cl_alpha_wing")
        elif self.options["function_exe"] == "cl_alpha_HTP":
            use_fuselage = False
            use_horizontal_tail = True
            use_definition_angle = False
            self.add_output("openvsp:cl_alpha_htp")
        elif self.options["function_exe"] == "HTP_sizing":
            use_fuselage = False
            use_horizontal_tail = True
            use_definition_angle = True
            self.add_output("openvsp:cl_htp")
            self.add_output("openvsp:cm_wing")
        else:
            raise IOError("Undefined function!")
        
        self.add_input("openvsp:altitude", val=np.nan, units="ft")
        self.add_input("openvsp:mach", val=np.nan)
        
        if use_definition_angle:
            self.add_input("openvsp:angle", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_0", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        if use_fuselage:
            self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
            self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")
            self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
        if use_horizontal_tail:
            self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
            self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
            self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
            self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
            self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
            self.add_input("ata:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
            self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:local", val=np.nan, units="m")
            self.add_input("data:geometry:horizontal_tail:height", val=np.nan, units="m")
        
        self.declare_partials("*", "*", method="fd")        
    
    def compute(self, inputs, outputs):
        
        # Pre-processing (populating temp directory) -----------------------------------------------
        tmp_directory = self._create_tmp_directory()
        copy_resource(openvsp3201, VSPSCRIPT_EXE_NAME, VSPAERO_EXE_NAME, tmp_directory.name)
        
        # Calculates the different options
        if self.options["function_exe"] == "stability":
            """Launches simulation of full aircraft to calculate Cm_alpha.
            Reference point (wing AC) to be defined in vspaero file
            """
            alpha = inputs["openvsp:alpha"]
            # Launch calculation steps
            self._write_vspscript_wing(tmp_directory, inputs)
            self._write_vspscript_fus(tmp_directory, inputs)
            self._write_vspscript_htp(tmp_directory, inputs)
            self._write_vspscript_end(tmp_directory)
            AOAList = [alpha-0.2, alpha+0.2]
            self._write_vspaero_file(tmp_directory, inputs, AOAList)
            self._run_vspaero_script(tmp_directory)
            # Open results
            vspaero_basename = os.path.splitext(_STDOUT_AERO_FILE_NAME)[0]
            lodfile = os.path.join(self.tmpdir,vspaero_basename + '.lod')
            with open(lodfile, 'r') as lf:
                data = lf.readlines()
                cm_wing = []
                cm_htp = []
                for i in range(len(data)):
                    line = data[i].split()
                    line.append('**')
                    if line[0] == 'Comp':
                        cm_wing.append(float(data[i+1].split()[12])\
                                      + float(data[i+2].split()[12]))
                        cm_htp.append(float(data[i+3].split()[12])\
                                      + float(data[i+4].split()[12]))
            # Calculate derivative
            cm_alpha_wing = (cm_wing[1]-cm_wing[0]) / (0.4*math.pi/180)
            cm_alpha_htp = (cm_htp[1]-cm_htp[0]) / (0.4*math.pi/180)
    
            outputs["openvsp:cm_alpha"] = cm_alpha_wing + cm_alpha_htp
        
        elif self.options["function_exe"] == "inducedDrag":
            """Returns the Oswald factor of the wing. Corrections:
                - Full aircraft
                - Fuselage
            """
            # Launch calculation steps
            self._write_vspscript_wing(tmp_directory, inputs)
            self._write_vspscript_end(tmp_directory)
            AOAList = [7.0] # ???: why such value chosen?
            self._write_vspaero_file(tmp_directory, inputs, AOAList)
            self._run_vspaero_script(tmp_directory)
            _, _, result_oswald, _ = self._read_polar_file(tmp_directory, AOAList)
            oswald = result_oswald[0]
            # Fuselage correction
            b_f = self.aircraft.vars_geometry['width_max']
            span = self.aircraft.vars_geometry['span']
            k_fus = 1 - 2*(b_f/span)**2
            # Full aircraft correction: Wing lift is 105% of total lift.
            # This means CDind = (CL*1.05)^2/(piAe) -> e' = e/1.05^2
            oswald = oswald * k_fus / 1.05**2
            
            outputs["openvsp:oswald"] = oswald
            
        elif self.options["function_exe"] == "cl_alpha_wing":
            """Wing simulation to get the CL_alpha of the wing"""
            # Launch calculation steps
            self._write_vspscript_wing(tmp_directory, inputs)
            self._write_vspscript_end(tmp_directory)
            AOAList = [0.0, 5.0] # !!!: arround 0° calculation
            self._write_vspaero_file(tmp_directory, inputs, AOAList)
            self._run_vspaero_script(tmp_directory)
            result_cl, _, _, _ = self._read_polar_file(tmp_directory, AOAList)
            # Fuselage correction
            b_f = self.aircraft.vars_geometry['width_max']
            span = self.aircraft.vars_geometry['span']
            k_fus = 1 + 0.025*b_f/span - 0.025*(b_f/span)**2
            cl_0 = result_cl[0] * k_fus
            cl_5 = result_cl[1] * k_fus
            # Calculate derivative
            cl_alpha_wing = (cl_5 - cl_0) / (5.*math.pi/180)
            
            outputs["openvsp:cl_0"] = cl_0
            outputs["openvsp:cl_alpha_wing"] = cl_alpha_wing

        elif self.options["function_exe"] == "cl_alpha_HTP":
            """Full aircraft simulation to get the CL_alpha of the tail (with downwash
            effects). Gets results from .lod file, in order to distinguish the tail.
            """
            # Launch calculation steps
            self._write_vspscript_wing(tmp_directory, inputs)
            self._write_vspscript_htp(tmp_directory, inputs)
            self._write_vspscript_end(tmp_directory)
            AOAList = [2.0, 4.0] # ???: why mid point is 3° ?
            self._write_vspaero_file(tmp_directory, inputs, AOAList)
            self._run_vspaero_script(tmp_directory)
             # Open results
            vspaero_basename = os.path.splitext(_STDOUT_AERO_FILE_NAME)[0]
            lodfile = os.path.join(self.tmpdir,vspaero_basename + '.lod')
            with open(lodfile, 'r') as lf:
                data = lf.readlines()
                cl_htp = []
                for i in range(len(data)):
                    line = data[i].split()
                    line.append('**')
                    if line[0] == 'Comp':
                        cl_htp.append(float(data[i+3].split()[5]))
                        cl_htp.append(float(data[i+4].split()[5]))
            cl_alpha_htp = (cl_htp[2] + cl_htp[3] - cl_htp[0] - cl_htp[1]) \
                            / (2 * math.pi/180)
            
            outputs["openvsp:cl_alpha_htp"] = cl_alpha_htp
        
        elif self.options["function_exe"] == "HTP_sizing":
            """Return necessary coefficients for the HT sizing (with a full AC run).
            """
            alpha = inputs["openvsp:alpha"]
            # Launch calculation steps
            self._write_vspscript_wing(tmp_directory, inputs)
            self._write_vspscript_htp(tmp_directory, inputs)
            self._write_vspscript_end(tmp_directory)
            AOAList = [alpha] 
            self._write_vspaero_file(tmp_directory, inputs, AOAList)
            self._run_vspaero_script(tmp_directory)
            # Open results
            vspaero_basename = os.path.splitext(_STDOUT_AERO_FILE_NAME)[0]
            lodfile = os.path.join(self.tmpdir,vspaero_basename + '.lod')
            with open(lodfile, 'r') as lf:
                data = lf.readlines()
                cl_htp = []
                cm_wing = []
                for i in range(len(data)):
                    line = data[i].split()
                    line.append('**')
                    if line[0] == 'Comp':
                        cl_htp.append(float(data[i+3].split()[5]))
                        cl_htp.append(float(data[i+4].split()[5]))
                        cm_wing.append(float(data[i+1].split()[12]))
                        cm_wing.append(float(data[i+2].split()[12]))      
            cl_htp = cl_htp[0] + cl_htp[1]
            cm_wing = cm_wing[0] + cm_wing[1] 
        
            outputs["openvsp:cl_htp"] = cl_htp
            outputs["openvsp:cm_wing"] = cm_wing
            
        # Delete temporary directory    
        tmp_directory.cleanup()                
    
    @staticmethod
    def _write_vspscript_wing(tmp_directory, inputs):
        
        """Wing definition in VSP (mandatory)."""
        
        x0_wing = inputs["data:geometry:wing:MAC:leading_edge:x:local"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        y1_wing = inputs["data:geometry:fuselage:maximum_width"]/2.0
        y2_wing = inputs["data:geometry:wing:root:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        sweep_0 = inputs["data:geometry:wing:sweep_0"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        
        # Calcultate input parameters
        x_wing = fa_length-x0_wing-0.25*l0_wing
        z_wing = -(height_max - 0.12*l2_wing)*0.5
        span_2 = y4_wing - y2_wing
        
        # Define wing section profile
        resourcesdir = pth.join(pth.dirname(__file__), 'resources')
        resourcesdir = pth.abspath(resourcesdir)
        airfoil_0 = pth.join(resourcesdir, "naca23012.af") # FIXME: only one definition of NACA, to be defined in xml?
        airfoil_0 = airfoil_0.replace('\\', '/')
        airfoil_2 = pth.join(resourcesdir, "naca23012.af") # FIXME: only one definition of NACA, to be defined in xml?
        airfoil_2 = airfoil_2.replace('\\', '/')
        airfoil_4 = pth.join(resourcesdir, "naca23012.af") # FIXME: only one definition of NACA, to be defined in xml?
        airfoil_4 = airfoil_4.replace('\\', '/')
        
        # Define temporary file
        file = open(pth.join(tmp_directory.name, _STDOUT_SCRIPT_FILE_NAME),"w")
        
        # Insert wing section
        file.write( '\n//==== Create A Multi Section Wing and Change Some Parameters ====//\n')
        file.write( 'void main()\n')
        file.write( '{\n')
        file.write( '   //==== Add Wing ====//\n')
        file.write( '   string wid = AddGeom( "WING", "");\n\n')
        file.write( '	string xsec_surf = GetXSecSurf( wid, 0 ); \n')
        file.write( '	string xsec = GetXSec( xsec_surf, 0 ); \n\n')
        # Insert several sections
        file.write( '   //===== Insert A Couple More Sections =====// \n\n')
        file.write( '   InsertXSec( wid, 1, XS_FILE_AIRFOIL ); \n')
        file.write( '   SetParmVal( GetParm(wid, "Tess_W", "Shape"), 41 ); \n')
        file.write( '   SetParmVal( GetParm(wid, "LECluster", "WingGeom"), 0.7 ); \n')
        file.write( '   SetParmVal( GetParm(wid, "TECluster", "WingGeom"), 0.9 ); \n')
        file.write( '   SetParmVal( GetParm(wid, "X_Rel_Location", "XForm"), {}); \n'.format(x_wing))
        file.write( '   SetParmVal( GetParm(wid, "Z_Rel_Location", "XForm"), {}); \n'.format(z_wing))
        file.write( '   Update(); \n\n')
        # Parameters section 1
        file.write( '   //===== Change Some Parameters 1st Section ====// \n\n')
        file.write( '   SetParmVal( GetParm(wid, "Span", "XSec_1"), {});\n'.format(y1_wing))
        file.write( '   SetParmVal( GetParm(wid, "Root_Chord", "XSec_1"), {}); \n'.format(l2_wing))
        file.write( '   SetParmVal( GetParm(wid, "Tip_Chord", "XSec_1"), {}); \n'.format(l2_wing))
        file.write( '   SetParmVal( GetParm(wid, "Sweep", "XSec_1"), 0.0); \n')
        file.write( '   SetParmVal( GetParm(wid, "Sec_Sweep_Location", "XSec_1"), 0.7); \n')
        file.write( '   SetParmVal( GetParm(wid, "Sweep_Location", "XSec_1"), 0); \n')
        file.write( '   SetParmVal( GetParm(wid, "SectTess_U", "XSec_1"), 8); \n')
        file.write( '   Update(); \n\n')
        # Parameters section 2 (case simple sweep)
        file.write( '   //===== Change Some Parameters 2nd Section ====// \n\n')
        file.write( '   SetParmVal( GetParm(wid, "Span", "XSec_2"), {});\n'.format(span_2))
        file.write( '   SetParmVal( GetParm(wid, "Root_Chord", "XSec_2"), {});\n'.format(l2_wing))
        file.write( '   SetParmVal( GetParm(wid, "Tip_Chord", "XSec_2"), {});\n'.format(l4_wing))
        file.write( '   SetParmVal( GetParm(wid, "Sweep", "XSec_2"), {});\n'.format(sweep_0))
        file.write( '   SetParmVal( GetParm(wid, "Sec_Sweep_Location", "XSec_2"), 0.7);\n')
        file.write( '   SetParmVal( GetParm(wid, "Sweep_Location", "XSec_2"), 0);\n')
        file.write( '   SetParmVal( GetParm(wid, "SectTess_U", "XSec_2"), 33);\n')
        file.write( '   SetParmVal( GetParm(wid, "OutCluster", "XSec_2"), 0.9);\n')
        file.write( '   Update(); \n\n')
        # Airfoil definition at y=0
        file.write( '	//==== Change Airfoil 0 ====// \n')
        file.write( '	ChangeXSecShape( xsec_surf, 0, XS_FILE_AIRFOIL ); \n')
        file.write( '	xsec = GetXSec( xsec_surf, 0 ); \n')
        file.write( '	ReadFileAirfoil( xsec, "' + airfoil_0 + '" ); \n')
        file.write( '	Update(); \n\n')
        # Airfoil definition at y2
        file.write( '	//==== Change Airfoil 1 (y=y2) ====// \n')
        file.write( '	ChangeXSecShape( xsec_surf, 1, XS_FILE_AIRFOIL ); \n')
        file.write( '	xsec = GetXSec( xsec_surf, 1 ); \n')
        file.write( '	ReadFileAirfoil( xsec, "' + airfoil_2 + '" ); \n')
        file.write( '	Update(); \n\n')
        # Airfoil definition at y4
        file.write( '	//==== Change Airfoil 2 (y=y4) ====// \n')
        file.write( '	ChangeXSecShape( xsec_surf, 2, XS_FILE_AIRFOIL ); \n')
        file.write( '	xsec = GetXSec( xsec_surf, 2 ); \n')
        file.write( '	ReadFileAirfoil( xsec, "' + airfoil_4 + '" ); \n')
        file.write( '	Update(); \n\n')
        file.close()
    
    @staticmethod    
    def _write_vspscript_fus(tmp_directory, inputs):
        
        """Fuselage definition in VSP (optional)."""

        fus_length = inputs["data:geometry:fuselage:length"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        lav = inputs["data:geometry:fuselage:front_length"]
        lar = inputs["data:geometry:fuselage:rear_length"]
        
        # Calculate input parameters of sections 1-4
        x1 = 0.7 * lav / fus_length
        z1 = -0.5 * 0.3 * height_max * 0.8 / fus_length
        h1 = 0.7 * height_max
        w1 = 0.95 * width_max
        x2 = lav / fus_length
        h2 = height_max
        w2 = width_max
        x3 = (fus_length - lar) / fus_length
        h3 = height_max * 0.9
        w3 = width_max * 0.9
        x4 = (fus_length - lar*0.25) / fus_length
        h4 = height_max * 0.5
        w4 = width_max * 0.25
        
        # Define temporary file
        file = open(pth.join(tmp_directory.name, _STDOUT_SCRIPT_FILE_NAME),"a")
        
        # Insert fuselage section
        file.write( '    //==== Add fuselage ====// \n' )
        file.write( '    string fid = AddGeom("FUSELAGE"); \n')
        file.write( '    SetParmVal(fid, "Length", "Design", {});\n'.format(fus_length))
        file.write( '    Update(); \n\n')
        file.write( '    //==== Insert sections ====//  By default there are 3 middle sections \n' )
        file.write( '    InsertXSec ( fid, 1, XS_ELLIPSE); \n')
        file.write( '    Update(); \n\n')
        # Parameters section 1
        file.write( '    //==== Change parameters 1st section ====//  \n' )
        file.write( '    SetParmVal( fid,"XLocPercent", "XSec_1", {});\n'.format(x1))
        file.write( '    SetParmVal( fid,"ZLocPercent", "XSec_1", {});\n'.format(z1))
        file.write( '    SetParmVal( fid,"Ellipse_Height", "XSecCurve_1", {});\n'.format(h1))
        file.write( '    SetParmVal( fid,"Ellipse_Width", "XSecCurve_1", {});\n'.format(w1))
        file.write( '    Update(); \n\n')
        # Parameters section 2
        file.write( '    //==== Change parameters 2nd section ====//  \n' )
        file.write( '    SetParmVal( fid,"XLocPercent", "XSec_2", {});\n'.format(x2))
        file.write( '    SetParmVal( fid,"Ellipse_Height", "XSecCurve_2", {});\n'.format(h2))
        file.write( '    SetParmVal( fid,"Ellipse_Width", "XSecCurve_2", {});\n'.format(w2))
        file.write( '    Update(); \n\n')
        # Parameters section 3
        file.write( '    //==== Change parameters 3rd section ====//  \n' )
        file.write( '    SetParmVal( fid,"XLocPercent", "XSec_3", {});\n'.format(x3))
        file.write( '    SetParmVal( fid,"Ellipse_Height", "XSecCurve_3", {});\n'.format(h3))
        file.write( '    SetParmVal( fid,"Ellipse_Width", "XSecCurve_3", {});\n'.format(w3))
        file.write( '    Update(); \n\n')
        # Parameters section 4
        file.write( '    //==== Change parameters 4th section ====//  \n' )
        file.write( '    SetParmVal( fid,"XLocPercent", "XSec_4", {});\n'.format(x4))
        file.write( '    SetParmVal( fid,"Ellipse_Height", "XSecCurve_4", {});\n'.format(h4))
        file.write( '    SetParmVal( fid,"Ellipse_Width", "XSecCurve_4", {});\n'.format(w4))
        file.write( '    Update(); \n\n')
        file.close()
    
    @staticmethod    
    def _write_vspscript_htp(tmp_directory, inputs):
            
        """Horizontal tail definition in VSP (optional)."""
        
        sweep_25 = inputs["data:geometry:horizontal_tail:sweep_25"]
        span = inputs["data:geometry:horizontal_tail:span"]/2.0
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        lp_ht = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"] 
        l0 = inputs["data:geometry:horizontal_tail:MAC:length"] 
        x0 = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:local"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        height = inputs["data:geometry:horizontal_tail:height"]
        
        # Calculate input parameters
        horizontal_distance = fa_length + lp_ht - 0.25 * l0 - x0
        
        # Define temporary file
        file = open(pth.join(tmp_directory.name, _STDOUT_SCRIPT_FILE_NAME),"a")
        
        # Insert horizontal tail section
        file.write( '    //==== Add wing (HTP) ====// \n' )
        file.write( '    string hid = AddGeom("WING", ""); \n\n')
        file.write( '	 xsec_surf = GetXSecSurf( hid, 0 ); \n\n')
        file.write( '    SetParmVal(hid, "Tess_W", "Shape", 31); \n')
        file.write( '    SetParmVal( GetParm(hid, "LECluster", "WingGeom"), 0.7 ); \n')
        file.write( '    SetParmVal( GetParm(hid, "TECluster", "WingGeom"), 0.9 ); \n')
        file.write( '    SetParmVal( GetParm(hid, "X_Rel_Location", "XForm"), {});\n'.format(horizontal_distance))
        file.write( '    SetParmVal( GetParm(hid, "Z_Rel_Location", "XForm"), {});\n'.format(height))
        file.write( '    Update(); \n\n')
        # Parameters section 1
        file.write( '    //===== Change Some Parameters 1st Section ====// \n\n')
        file.write( '    SetParmVal( GetParm(hid, "Span", "XSec_1"), {});\n'.format(span))
        file.write( '    SetParmVal( GetParm(hid, "Root_Chord", "XSec_1"), {});\n'.format(root_chord))
        file.write( '    SetParmVal( GetParm(hid, "Tip_Chord", "XSec_1"), {});\n'.format(tip_chord))
        file.write( '    SetParmVal( GetParm(hid, "Sweep", "XSec_1"), {});\n'.format(sweep_25))
        file.write( '    SetParmVal( GetParm(hid, "Sweep_Location", "XSec_1"), 0.25); \n')
        file.write( '    SetParmVal( GetParm(hid, "SectTess_U", "XSec_1"), 25); \n')
        file.write( '    Update(); \n\n')
        # Airfoils
        file.write( '	//==== Change Airfoils (symmetrical)  ====// \n\n')
        file.write( '	ChangeXSecShape( xsec_surf, 0, XS_FOUR_SERIES ); \n')
        file.write( '	ChangeXSecShape( xsec_surf, 1, XS_FOUR_SERIES ); \n')
        file.write( '   Update(); \n\n')
        file.close()
    
    @staticmethod    
    def _write_vspscript_end(tmp_directory):
        
        """Closing statements of the VSP script (mandatory)."""
        
        # Define temporary file
        file = open(pth.join(tmp_directory.name, _STDOUT_SCRIPT_FILE_NAME),"a")
        vspaero_basename = pth.splitext(_STDOUT_AERO_FILE_NAME)[0]
        csvfile = pth.join(tmp_directory.name, vspaero_basename + '.csv')
        csvfile = csvfile.replace('\\', '/')
        
        #Errors
        file.write( '	//==== Check For API Errors ====//\n')
        file.write( '   while ( GetNumTotalErrors() > 0 )\n')
        file.write( '   {\n')
        file.write( '       ErrorObj err = PopLastError();\n')
        file.write( '       Print( err.GetErrorString() );\n')
        file.write( '   }\n\n')
        #Compute DegenGeom
        file.write( '   //==== Set File Name ====//\n')
        file.write( '   SetComputationFileName( DEGEN_GEOM_CSV_TYPE, ' + csvfile + ' );\n\n')
        file.write( '   //==== Run Degen Geom ====//\n')
        file.write( '   ComputeDegenGeom( SET_ALL, DEGEN_GEOM_CSV_TYPE );\n}')
        file.close()
        
    @staticmethod
    def _run_vspscript_script(tmp_directory):
        # Run the vspscript using vsp
        handle = win32process.CreateProcess(
            None,
            pth.join(tmp_directory.name, VSPSCRIPT_EXE_NAME) + ' -script ' +
            pth.join(tmp_directory.name, _STDOUT_SCRIPT_FILE_NAME),
            None,
            None,
            0,
            win32process.CREATE_NO_WINDOW,
            None,
            None,
            win32process.STARTUPINFO())
        win32event.WaitForSingleObject(handle[0], -1) # to wait for the exit of the process
    
    @staticmethod  
    def _write_vspaero_file(tmp_directory, inputs, AOAList):
        
        """Defines the simulation parameters (AoA, mach, rho, reference point for moments)."""
        
        Sref = inputs['data:geometry:wing:area']
        Cref = inputs["data:geometry:wing:MAC:length"]
        Bref = inputs['data:geometry:wing:span']
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        altitude = inputs["openvsp:altitude"]
        mach = inputs["openvsp:mach"]
        AOAList = str(AOAList)
        AOAList = AOAList[1:len(AOAList)-1]
        
        # Define air parameters
        atm = Atmosphere(altitude)
        speed_of_sound = atm.speed_of_sound
        viscosity = atm.kinematic_viscosity
        rho = atm.density
        # Calculate Reynolds
        V_inf = min(speed_of_sound * mach, 0.1) # avoid V=0 m/s crashes
        reynolds = V_inf * Cref / viscosity
        
        # Define temporary file
        file = open(pth.join(tmp_directory.name, _STDOUT_AERO_FILE_NAME),"w")
        
        # Write parameters for simulation
        file.write("Sref = {}\n".format(Sref))
        file.write("Cref = {}\n".format(Cref))
        file.write("Bref = {}\n".format(Bref))
        file.write("X_cg = {}\n".format(fa_length))
        file.write("Y_cg = 0.000000\n")
        file.write("Z_cg = -0.150000\n")
        file.write("Mach = {}\n".format(mach))
        file.write("AoA = " + AOAList + "\n")
        file.write("Beta = 0.000000\n")
        file.write("Vinf = {}\n".format(V_inf))
        file.write("Rho = {}\n".format(rho))
        file.write("ReCref = {}\n".format(reynolds))
        file.write("ClMax = -1.000000\n")
        file.write("MaxTurningAngle = -1.000000 \n")
        file.write("Symmetry = No \n")
        file.write("FarDist = -1.000000\n")
        file.write("NumWakeNodes = " + str(multiprocessing.cpu_count()) + " \n")
        file.write("WakeIters = 5 \n")
        file.write("NumberOfRotors = 0")
        file.close()
    
    @staticmethod
    def _run_vspaero_script(tmp_directory):
        # Run DegenGeom using vspaero
        handle2 = win32process.CreateProcess(
            None,
            os.path.join(tmp_directory.name, VSPAERO_EXE_NAME) + " " +
            os.path.join(tmp_directory.name, _STDOUT_AERO_FILE_NAME),
            None,
            None,
            0,
            win32process.CREATE_NO_WINDOW,
            None,
            None,
            win32process.STARTUPINFO())
        win32event.WaitForSingleObject(handle2[0], -1) # to wait for the exit of the process
        
    @staticmethod
    def _read_polar_file(tmp_directory, AOAList):
        result_cl = []
        result_cdi = []
        result_oswald = []
        result_cm = []
        vspaero_basename = os.path.splitext(_STDOUT_AERO_FILE_NAME)[0]
        polar_file = pth.join(tmp_directory.name,vspaero_basename + '.polar')
        # Colect data from .polar file
        with open(polar_file, 'r') as hf:
            l1 = hf.readlines()
            for i in range(len(AOAList)):
                #Cl
                result = l1[i+1][40:50]
                result = result.replace(' ', '')
                result_cl.append(float(result))
                #Cdi
                result = l1[i+1][60:70]
                result = result.replace(' ', '')
                result_cdi.append(float(result))
                #Oswald
                result = l1[i+1][100:110]
                result = result.replace(' ', '')
                result_oswald.append(float(result))
                #Cm
                result = l1[i+1][150:160]
                result = result.replace(' ', '')
                result_cm.append(float(result))
        return result_cl, result_cdi, result_oswald, result_cm
    
    @staticmethod
    def _create_tmp_directory() -> TemporaryDirectory:
        
        """Provide temporary directory for calculation."""
        
        for tmp_base_path in [None, pth.join(str(Path.home()), ".fast")]:
            if tmp_base_path is not None:
                os.makedirs(tmp_base_path, exist_ok=True)
            tmp_directory = tempfile.TemporaryDirectory(prefix="x", dir=tmp_base_path)
            break
            
        return tmp_directory