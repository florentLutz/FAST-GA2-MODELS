"""
Test module for geometry functions of cg components
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

# pylint: disable=redefined-outer-name  # needed for pytest fixtures
import os.path as pth
import openmdao.api as om

import pytest
from fastoad.io import VariableIO

from tests.testing_utilities import run_system

from ..geom_components.fuselage import (
    ComputeFuselageGeometryBasic,
    ComputeFuselageGeometryCabinSizing,
)
from ..geom_components.wing.components import (
    ComputeMFW,
    ComputeWingB50,
    ComputeWingL1AndL4,
    ComputeWingL2AndL3,
    ComputeWingMAC,
    ComputeWingSweep,
    ComputeWingToc,
    ComputeWingWetArea,
    ComputeWingX,
    ComputeWingY,
)
from ..geom_components.ht.components import (
    ComputeHTChord,
    ComputeHTMAC,
    ComputeHTSweep,
    ComputeHTWetArea,
    ComputeHTDistance,
)
from ..geom_components.vt.components import (
    ComputeVTChords,
    ComputeVTMAC,
    ComputeVTSweep,
    ComputeVTWetArea,
)
from ..geom_components.nacelle.compute_nacelle import ComputeNacelleGeometry
from ..geom_components import ComputeTotalArea

XML_FILE = "beechcraft_76.xml"

def get_indep_var_comp(var_names):
    """ Reads required input data and returns an IndepVarcomp() instance"""
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", XML_FILE))
    reader.path_separator = ":"
    ivc = reader.read(only=var_names).to_ivc()
    return ivc


def list_inputs(group):
    """ Reads input variables from a group and return as a list (run model with 0 value can lead to raise configuration
    errors in models)"""
    prob = om.Problem(model=group)
    prob.setup()
    prob.run_model()
    data = prob.model.list_inputs(out_stream=None)
    list_names = []
    for idx in range(len(data)):
        variable_name = data[idx][0].split('.')
        list_names.append(variable_name[len(variable_name) - 1])
    return list(dict.fromkeys(list_names))


def test_compute_vt_chords():
    """ Tests computation of the vertical tail chords """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeVTChords(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTChords(), ivc)
    span = problem.get_val("data:geometry:vertical_tail:span", units="m")
    assert span == pytest.approx(1.734, abs=1e-3)
    root_chord = problem.get_val("data:geometry:vertical_tail:root:chord", units="m")
    assert root_chord == pytest.approx(1.785, abs=1e-3)
    tip_chord = problem.get_val("data:geometry:vertical_tail:tip:chord", units="m")
    assert tip_chord == pytest.approx(1.106, abs=1e-3)


def test_compute_vt_mac():
    """ Tests computation of the vertical tail mac """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeVTMAC(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:vertical_tail:span", 1.734, units="m")
    ivc.add_output("data:geometry:vertical_tail:root:chord", 1.785, units="m")
    ivc.add_output("data:geometry:vertical_tail:tip:chord", 1.106, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTMAC(), ivc)
    length = problem.get_val("data:geometry:vertical_tail:MAC:length", units="m")
    assert length == pytest.approx(1.472, abs=1e-3)
    vt_x0 = problem.get_val("data:geometry:vertical_tail:MAC:at25percent:x:local", units="m")
    assert vt_x0 == pytest.approx(0.219, abs=1e-3)
    vt_z0 = problem.get_val("data:geometry:vertical_tail:MAC:z", units="m")
    assert vt_z0 == pytest.approx(0.799, abs=1e-3)
    vt_lp = problem.get_val("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", units="m")
    assert vt_lp == pytest.approx(4.334, abs=1e-3)



def test_compute_vt_sweep():
    """ Tests computation of the vertical tail sweep """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeVTSweep(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:vertical_tail:span", 1.734, units="m")
    ivc.add_output("data:geometry:vertical_tail:root:chord", 1.785, units="m")
    ivc.add_output("data:geometry:vertical_tail:tip:chord", 1.106, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTSweep(), ivc)
    sweep_0 = problem.get_val("data:geometry:vertical_tail:sweep_0", units="deg")
    assert sweep_0 == pytest.approx(15.3, abs=1e-1)
    sweep_100 = problem.get_val("data:geometry:vertical_tail:sweep_100", units="deg")
    assert sweep_100 == pytest.approx(173.3, abs=1e-1)


def test_compute_vt_wet_area():
    """ Tests computation of the vertical wet area """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeVTWetArea(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTWetArea(), ivc)
    wet_area = problem.get_val("data:geometry:vertical_tail:wet_area", units="m**2")
    assert wet_area == pytest.approx(5.265, abs=1e-3)


def test_compute_ht_distance():
    """ Tests computation of the horizontal tail distance """

    # Input list from model (not generated because of assertion error on tail type)
    input_list = [
        "data:geometry:has_T_tail",
    ]

    # Add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:vertical_tail:span", 1.734, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTDistance(), ivc)
    lp_vt = problem.get_val("data:geometry:horizontal_tail:z:from_wingMAC25", units="m")
    assert lp_vt == pytest.approx(1.734, abs=1e-3)


def test_compute_ht_chord():
    """ Tests computation of the horizontal tail chords """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeHTChord(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTChord(), ivc)
    span = problem.get_val("data:geometry:horizontal_tail:span", units="m")
    assert span == pytest.approx(5.095, abs=1e-3)
    root_chord = problem.get_val("data:geometry:horizontal_tail:root:chord", units="m")
    assert root_chord == pytest.approx(0.868, abs=1e-3)
    tip_chord = problem.get_val("data:geometry:horizontal_tail:tip:chord", units="m")
    assert tip_chord == pytest.approx(0.868, abs=1e-3)
    aspect_ratio = problem.get_val("data:geometry:horizontal_tail:aspect_ratio")
    assert aspect_ratio == pytest.approx(5.871, abs=1e-3)


def test_compute_ht_mac():
    """ Tests computation of the horizontal tail mac """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeHTMAC(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:horizontal_tail:span", 5.095, units="m")
    ivc.add_output("data:geometry:horizontal_tail:root:chord", 0.868, units="m")
    ivc.add_output("data:geometry:horizontal_tail:tip:chord", 0.868, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTMAC(), ivc)
    length = problem.get_val("data:geometry:horizontal_tail:MAC:length", units="m")
    assert length == pytest.approx(0.868, abs=1e-3)
    ht_x0 = problem.get_val("data:geometry:horizontal_tail:MAC:at25percent:x:local", units="m")
    assert ht_x0 == pytest.approx(0.0890, abs=1e-3)
    ht_y0 = problem.get_val("data:geometry:horizontal_tail:MAC:y", units="m")
    assert ht_y0 == pytest.approx(1.274, abs=1e-3)


def test_compute_ht_sweep():
    """ Tests computation of the horizontal tail sweep """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeHTSweep(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:horizontal_tail:span", 5.095, units="m")
    ivc.add_output("data:geometry:horizontal_tail:root:chord", 0.868, units="m")
    ivc.add_output("data:geometry:horizontal_tail:tip:chord", 0.868, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTSweep(), ivc)
    sweep_0 = problem.get_val("data:geometry:horizontal_tail:sweep_0", units="deg")
    assert sweep_0 == pytest.approx(4.0, abs=1e-1)
    sweep_100 = problem.get_val("data:geometry:horizontal_tail:sweep_100", units="deg")
    assert sweep_100 == pytest.approx(4.0, abs=1e-1)


def test_compute_ht_wet_area():
    """ Tests computation of the horizontal tail wet area """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeHTWetArea(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTWetArea(), ivc)
    wet_area = problem.get_val("data:geometry:horizontal_tail:wet_area", units="m**2")
    assert wet_area == pytest.approx(7.428, abs=1e-2)


def test_compute_fuselage_cabin_sizing():
    """ Tests computation of the fuselage with cabin sizing """

    # Input list from model (not generated because of npax1 crash error for NaN values)
    input_list = [
        "data:TLAR:NPAX",
        "data:geometry:cabin:seats:pilot:length",
        "data:geometry:cabin:seats:pilot:width",
        "data:geometry:cabin:seats:passenger:length",
        "data:geometry:cabin:seats:passenger:width",
        "data:geometry:cabin:seats:passenger:count_by_row",
        "data:geometry:cabin:aisle_width",
        "data:geometry:propulsion:layout",
        "data:geometry:propulsion:length",
        "data:geometry:wing:MAC:at25percent:x",
        "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25",
        "data:geometry:propulsion:length",
    ]

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:horizontal_tail:MAC:length", 0.868, units="m")
    ivc.add_output("data:geometry:vertical_tail:MAC:length", 1.472, units="m")
    ivc.add_output("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", 4.334, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageGeometryCabinSizing(), ivc)
    npax_1 = problem.get_val("data:geometry:cabin:NPAX")
    assert npax_1 == pytest.approx(4.0, abs=1)
    fuselage_length = problem.get_val("data:geometry:fuselage:length", units="m")
    assert fuselage_length == pytest.approx(8.888, abs=1e-3)
    fuselage_width_max = problem.get_val("data:geometry:fuselage:maximum_width", units="m")
    assert fuselage_width_max == pytest.approx(1.198, abs=1e-3)
    fuselage_height_max = problem.get_val("data:geometry:fuselage:maximum_height", units="m")
    assert fuselage_height_max == pytest.approx(1.338, abs=1e-3)
    fuselage_lav = problem.get_val("data:geometry:fuselage:front_length", units="m")
    assert fuselage_lav == pytest.approx(2.274, abs=1e-3)
    fuselage_lar = problem.get_val("data:geometry:fuselage:rear_length", units="m")
    assert fuselage_lar == pytest.approx(2.852, abs=1e-3)
    fuselage_lpax = problem.get_val("data:geometry:fuselage:PAX_length", units="m")
    assert fuselage_lpax == pytest.approx(2.35, abs=1e-3)
    fuselage_lcabin = problem.get_val("data:geometry:cabin:length", units="m")
    assert fuselage_lcabin == pytest.approx(3.759, abs=1e-3)
    fuselage_wet_area = problem.get_val("data:geometry:fuselage:wet_area", units="m**2")
    assert fuselage_wet_area == pytest.approx(30.311, abs=1e-1) # difference comes from LAR=0.0 in old version
    luggage_length = problem.get_val("data:geometry:fuselage:luggage_length", units="m")
    assert luggage_length == pytest.approx(0.709, abs=1e-3)


def test_compute_fuselage_basic():
    """ Tests computation of the fuselage with no cabin sizing """

    # Define the independent input values that should be filled if basic function is choosen
    ivc = om.IndepVarComp()
    ivc.add_output("data:geometry:fuselage:length", 8.888, units="m")
    ivc.add_output("data:geometry:fuselage:maximum_width", 1.198, units="m")
    ivc.add_output("data:geometry:fuselage:maximum_height", 1.338, units="m")
    ivc.add_output("data:geometry:fuselage:front_length", 2.274, units="m")
    ivc.add_output("data:geometry:fuselage:rear_length", 2.852, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageGeometryBasic(), ivc)
    fuselage_lcabin = problem.get_val("data:geometry:cabin:length", units="m")
    assert fuselage_lcabin == pytest.approx(3.762, abs=1e-3)
    fuselage_wet_area = problem.get_val("data:geometry:fuselage:wet_area", units="m**2")
    assert fuselage_wet_area == pytest.approx(30.321, abs=1e-3)


def test_geometry_wing_mfw():
    """ Tests computation of the wing max fuel weight """

    # Input list from model (not generated because of the assertion error on bad fuel type configuration)
    input_list = [
        "data:geometry:wing:area",
        "data:propulsion:engine:fuel_type",
    ]

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeMFW(), ivc)
    mfw = problem.get_val("data:weight:aircraft:MFW", units="kg")
    assert mfw == pytest.approx(587.16, abs=1e-2)


def test_geometry_wing_toc():
    """ Tests computation of the wing ToC (Thickness of Chord) """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeWingToc(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file
    ivc = get_indep_var_comp(input_list)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingToc(), ivc)
    toc_root = problem["data:geometry:wing:root:thickness_ratio"]
    assert toc_root == pytest.approx(0.149, abs=1e-3)
    toc_kink = problem["data:geometry:wing:kink:thickness_ratio"]
    assert toc_kink == pytest.approx(0.113, abs=1e-3)
    toc_tip = problem["data:geometry:wing:tip:thickness_ratio"]
    assert toc_tip == pytest.approx(0.103, abs=1e-3)


def test_geometry_wing_y():
    """ Tests computation of the wing Ys """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeWingY(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:fuselage:maximum_width", 1.198, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingY(), ivc)
    span = problem.get_val("data:geometry:wing:span", units="m")
    assert span == pytest.approx(12.363, abs=1e-3)
    wing_y2 = problem.get_val("data:geometry:wing:root:y", units="m")
    assert wing_y2 == pytest.approx(0.599, abs=1e-3)
    wing_y3 = problem.get_val("data:geometry:wing:kink:y", units="m")
    assert wing_y3 == pytest.approx(0.599, abs=1e-3) # point 3 and 2 equal (previous version ignored)
    wing_y4 = problem.get_val("data:geometry:wing:tip:y", units="m")
    assert wing_y4 == pytest.approx(6.181, abs=1e-3)


def test_geometry_wing_l1_l4():
    """ Tests computation of the wing chords (l1 and l4) """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeWingL1AndL4(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:wing:root:y", 0.6, units="m")
    ivc.add_output("data:geometry:wing:tip:y", 6.181, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingL1AndL4(), ivc)
    wing_l1 = problem.get_val("data:geometry:wing:root:virtual_chord", units="m")
    assert wing_l1 == pytest.approx(1.549, abs=1e-3)
    wing_l4 = problem.get_val("data:geometry:wing:tip:chord", units="m")
    assert wing_l4 == pytest.approx(1.549, abs=1e-3)


def test_geometry_wing_l2_l3():
    """ Tests computation of the wing chords (l2 and l3) """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeWingL2AndL3(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:wing:root:y", 0.6, units="m")
    ivc.add_output("data:geometry:wing:tip:y", 6.181, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingL2AndL3(), ivc)
    wing_l2 = problem.get_val("data:geometry:wing:root:chord", units="m")
    assert wing_l2 == pytest.approx(1.549, abs=1e-2)
    wing_l3 = problem.get_val("data:geometry:wing:kink:chord", units="m")
    assert wing_l3 == pytest.approx(1.549, abs=1e-2) #point 3 and 2 equal (previous version ignored)


def test_geometry_wing_x():
    """ Tests computation of the wing Xs """

    # Define input values calculated from other modules
    ivc = om.IndepVarComp()
    ivc.add_output("data:geometry:wing:root:chord", 1.549, units="m")
    ivc.add_output("data:geometry:wing:tip:chord", 1.549, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingX(), ivc)
    wing_x4 = problem.get_val("data:geometry:wing:tip:leading_edge:x:local", units="m")
    assert wing_x4 == pytest.approx(0.0, abs=1e-3)


def test_geometry_wing_b50():
    """ Tests computation of the wing B50 """

    # Define input values calculated from other modules
    ivc = om.IndepVarComp()
    ivc.add_output("data:geometry:wing:span", 12.363, units="m")
    ivc.add_output("data:geometry:wing:root:y", 0.6, units="m")
    ivc.add_output("data:geometry:wing:tip:y", 6.181, units="m")
    ivc.add_output("data:geometry:wing:root:virtual_chord", 1.549, units="m")
    ivc.add_output("data:geometry:wing:tip:chord", 1.549, units="m")
    ivc.add_output("data:geometry:wing:tip:leading_edge:x:local", 0.0, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingB50(), ivc)
    wing_b_50 = problem.get_val("data:geometry:wing:b_50", units="m")
    assert wing_b_50 == pytest.approx(12.363, abs=1e-3)


def test_geometry_wing_mac():
    """ Tests computation of the wing mean aerodynamic chord """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeWingMAC(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:wing:root:y", 0.6, units="m")
    ivc.add_output("data:geometry:wing:tip:y", 6.181, units="m")
    ivc.add_output("data:geometry:wing:root:chord", 1.549, units="m")
    ivc.add_output("data:geometry:wing:tip:chord", 1.549, units="m")
    ivc.add_output("data:geometry:wing:tip:leading_edge:x:local", 0.0, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingMAC(), ivc)
    wing_l0 = problem.get_val("data:geometry:wing:MAC:length", units="m")
    assert wing_l0 == pytest.approx(1.549, abs=1e-3)
    wing_x0 = problem.get_val("data:geometry:wing:MAC:leading_edge:x:local", units="m")
    assert wing_x0 == pytest.approx(0.0, abs=1e-3)
    wing_y0 = problem.get_val("data:geometry:wing:MAC:y", units="m")
    assert wing_y0 == pytest.approx(3.091, abs=1e-3)


def test_geometry_wing_sweep():
    """ Tests computation of the wing sweeps """

    # Define input values calculated from other modules
    ivc = om.IndepVarComp()
    ivc.add_output("data:geometry:wing:root:y", 0.6, units="m")
    ivc.add_output("data:geometry:wing:tip:y", 6.181, units="m")
    ivc.add_output("data:geometry:wing:root:chord", 1.549, units="m")
    ivc.add_output("data:geometry:wing:tip:chord", 1.549, units="m")
    ivc.add_output("data:geometry:wing:tip:leading_edge:x:local", 0.0, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingSweep(), ivc)
    sweep_0 = problem.get_val("data:geometry:wing:sweep_0", units="deg")
    assert sweep_0 == pytest.approx(0.0, abs=1e-1)
    sweep_100_inner = problem.get_val("data:geometry:wing:sweep_100_inner", units="deg")
    assert sweep_100_inner == pytest.approx(0.0, abs=1e-1)
    sweep_100_outer = problem.get_val("data:geometry:wing:sweep_100_outer", units="deg")
    assert sweep_100_outer == pytest.approx(0.0, abs=1e-1)


def test_geometry_wing_wet_area():
    """ Tests computation of the wing wet area """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeWingWetArea(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:wing:root:virtual_chord", 1.549, units="m")
    ivc.add_output("data:geometry:fuselage:maximum_width", 1.198, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingWetArea(), ivc)
    area_pf = problem.get_val("data:geometry:wing:outer_area", units="m**2")
    assert area_pf == pytest.approx(17.295, abs=1e-1)
    wet_area = problem.get_val("data:geometry:wing:wet_area", units="m**2")
    assert wet_area == pytest.approx(37.011, abs=1e-3)


def test_geometry_nacelle():
    """ Tests computation of the nacelle and pylons component """

    # Input list from model (not generated because of the assertion error on bad propulsion layout configuration)
    input_list = [
        "data:geometry:propulsion:layout",
        "data:geometry:propulsion:engine:height",
        "data:geometry:propulsion:engine:width",
        "data:geometry:propulsion:engine:length",
        "data:geometry:wing:span",
        "data:geometry:propulsion:engine:y_ratio",
        "data:geometry:fuselage:maximum_width",
    ]

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:fuselage:maximum_width", 1.198, units="m")
    ivc.add_output("data:geometry:wing:span", 12.363, units="m")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeNacelleGeometry(), ivc)
    nacelle_length = problem.get_val("data:geometry:propulsion:nacelle:length", units="m")
    assert nacelle_length == pytest.approx(1.44, abs=1e-3)
    nacelle_diameter = problem.get_val("data:geometry:propulsion:nacelle:diameter", units="m")
    assert nacelle_diameter == pytest.approx(0.871, abs=1e-3)
    nacelle_height = problem.get_val("data:geometry:propulsion:nacelle:height", units="m")
    assert nacelle_height == pytest.approx(0.691, abs=1e-3)
    nacelle_width = problem.get_val("data:geometry:propulsion:nacelle:width", units="m")
    assert nacelle_width == pytest.approx(0.871, abs=1e-3)
    nacelle_wet_area = problem.get_val("data:geometry:propulsion:nacelle:wet_area", units="m**2")
    assert nacelle_wet_area == pytest.approx(4.499, abs=1e-3)
    lg_height = problem.get_val("data:geometry:landing_gear:height", units="m")
    assert lg_height == pytest.approx(1.22, abs=1e-3)
    y_nacelle = problem.get_val("data:geometry:propulsion:nacelle:y", units="m")
    assert y_nacelle == pytest.approx(2.102, abs=1e-3)


def test_geometry_total_area():
    """ Tests computation of the total area """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeTotalArea(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:vertical_tail:wet_area", 5.265, units="m**2")
    ivc.add_output("data:geometry:horizontal_tail:wet_area", 7.428, units="m**2")
    ivc.add_output("data:geometry:fuselage:wet_area", 33.354, units="m**2")
    ivc.add_output("data:geometry:wing:wet_area", 37.011, units="m**2")

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeTotalArea(), ivc)
    total_surface = problem.get_val("data:geometry:aircraft:wet_area", units="m**2")
    assert total_surface == pytest.approx(92.056, abs=1e-3)