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
    ComputeHTDistance,
    ComputeHTMAC,
    ComputeHTSweep,
    ComputeHTWetArea,
)
from ..geom_components.vt.components import (
    ComputeVTChords,
    ComputeVTDistance,
    ComputeVTMAC,
    ComputeVTSweep,
    ComputeVTWetArea,
)
from ..geom_components.nacelle.compute_nacelle import ComputeNacelleGeometry
from ..geom_components import ComputeTotalArea


def get_indep_var_comp(var_names):
    """ Reads required input data and returns an IndepVarcomp() instance"""
    reader = VariableIO(pth.join(pth.dirname(__file__), "data", "geometry_inputs.xml"))
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
        "data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25",
        "data:geometry:horizontal_tail:span",
        "data:geometry:vertical_tail:span",
    ]

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:propulsion:length", 1.0)
    ivc.add_output("data:geometry:horizontal_tail:span", 12.28)
    ivc.add_output("data:geometry:vertical_tail:span", 6.62)
    ivc.add_output("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", 21.5)
    ivc.add_output("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", 21.5)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageGeometryCabinSizing(), ivc)
    npax_1 = problem["data:geometry:cabin:NPAX"]
    assert npax_1 == pytest.approx(4.0, abs=1)
    fuselage_length = problem["data:geometry:fuselage:length"]
    assert fuselage_length == pytest.approx(47.16, abs=1e-2)
    fuselage_width_max = problem["data:geometry:fuselage:maximum_width"]
    assert fuselage_width_max == pytest.approx(1.99, abs=1e-2)
    fuselage_height_max = problem["data:geometry:fuselage:maximum_height"]
    assert fuselage_height_max == pytest.approx(2.13, abs=1e-2)
    fuselage_lav = problem["data:geometry:fuselage:front_length"]
    assert fuselage_lav == pytest.approx(3.62, abs=1e-2)
    fuselage_lar = problem["data:geometry:fuselage:rear_length"]
    assert fuselage_lar == pytest.approx(40.2, abs=1e-1)
    fuselage_lpax = problem["data:geometry:fuselage:PAX_length"]
    assert fuselage_lpax == pytest.approx(2.35, abs=1e-2)
    fuselage_lcabin = problem["data:geometry:cabin:length"]
    assert fuselage_lcabin == pytest.approx(3.30, abs=1e-2)
    fuselage_wet_area = problem["data:geometry:fuselage:wet_area"]
    assert fuselage_wet_area == pytest.approx(230.5, abs=1e-1)
    luggage_length = problem["data:geometry:fuselage:luggage_length"]
    assert luggage_length == pytest.approx(0.25, abs=1e-1)


def test_compute_fuselage_basic():
    """ Tests computation of the fuselage with no cabin sizing """

    # Define the independent input values that should be filled if basic function is choosen
    ivc = om.IndepVarComp()
    ivc.add_output("data:geometry:fuselage:length", 43.34)
    ivc.add_output("data:geometry:fuselage:maximum_width", 1.20)
    ivc.add_output("data:geometry:fuselage:maximum_height", 1.34)
    ivc.add_output("data:geometry:fuselage:front_length", 2.27)
    ivc.add_output("data:geometry:fuselage:rear_length", 33.8)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeFuselageGeometryBasic(), ivc)
    fuselage_lcabin = problem["data:geometry:cabin:length"]
    assert fuselage_lcabin == pytest.approx(35.1, abs=1e-1)
    fuselage_wet_area = problem["data:geometry:fuselage:wet_area"]
    assert fuselage_wet_area == pytest.approx(134.6, abs=1e-1)


def test_geometry_wing_y():
    """ Tests computation of the wing Ys """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeWingY(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:fuselage:maximum_width", 1.20)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingY(), ivc)
    span = problem["data:geometry:wing:span"]
    assert span == pytest.approx(34.4, abs=1e-1)
    wing_y2 = problem["data:geometry:wing:root:y"]
    assert wing_y2 == pytest.approx(0.6, abs=1e-1)
    wing_y3 = problem["data:geometry:wing:kink:y"]
    assert wing_y3 == pytest.approx(0.6, abs=1e-1)
    wing_y4 = problem["data:geometry:wing:tip:y"]
    assert wing_y4 == pytest.approx(17.2, abs=1e-1)


def test_geometry_wing_l1_l4():
    """ Tests computation of the wing chords (l1 and l4) """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeWingL1AndL4(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:wing:root:y", 0.6)
    ivc.add_output("data:geometry:wing:tip:y", 17.2)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingL1AndL4(), ivc)
    wing_l1 = problem["data:geometry:wing:root:virtual_chord"]
    assert wing_l1 == pytest.approx(5.17, abs=1e-2)
    wing_l4 = problem["data:geometry:wing:tip:chord"]
    assert wing_l4 == pytest.approx(1.96, abs=1e-2)


def test_geometry_wing_l2_l3():
    """ Tests computation of the wing chords (l2 and l3) """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeWingL2AndL3(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:wing:root:y", 0.6)
    ivc.add_output("data:geometry:wing:tip:y", 17.2)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingL2AndL3(), ivc)
    wing_l2 = problem["data:geometry:wing:root:chord"]
    assert wing_l2 == pytest.approx(5.17, abs=1e-2)
    wing_l3 = problem["data:geometry:wing:kink:chord"]
    assert wing_l3 == pytest.approx(5.17, abs=1e-2)


def test_geometry_wing_x():
    """ Tests computation of the wing Xs """

    # Define input values calculated from other modules
    ivc = om.IndepVarComp()
    ivc.add_output("data:geometry:wing:root:chord", 5.17)
    ivc.add_output("data:geometry:wing:tip:chord", 1.96)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingX(), ivc)
    wing_x4 = problem["data:geometry:wing:tip:leading_edge:x:local"]
    assert wing_x4 == pytest.approx(0.80, abs=1e-2)


def test_geometry_wing_b50():
    """ Tests computation of the wing B50 """

    # Define input values calculated from other modules
    ivc = om.IndepVarComp()
    ivc.add_output("data:geometry:wing:span", 34.4)
    ivc.add_output("data:geometry:wing:root:y", 0.6)
    ivc.add_output("data:geometry:wing:tip:y", 17.2)
    ivc.add_output("data:geometry:wing:root:virtual_chord", 5.17)
    ivc.add_output("data:geometry:wing:tip:chord", 1.96)
    ivc.add_output("data:geometry:wing:tip:leading_edge:x:local", 0.8)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingB50(), ivc)
    wing_b_50 = problem["data:geometry:wing:b_50"]
    assert wing_b_50 == pytest.approx(34.44, abs=1e-2)


def test_geometry_wing_mac():
    """ Tests computation of the wing mean aerodynamic chord """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeWingMAC(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:wing:root:y", 0.6)
    ivc.add_output("data:geometry:wing:tip:y", 17.2)
    ivc.add_output("data:geometry:wing:root:chord", 5.17)
    ivc.add_output("data:geometry:wing:tip:chord", 1.96)
    ivc.add_output("data:geometry:wing:tip:leading_edge:x:local", 0.8)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingMAC(), ivc)
    wing_l0 = problem["data:geometry:wing:MAC:length"]
    assert wing_l0 == pytest.approx(3.86, abs=1e-2)
    wing_x0 = problem["data:geometry:wing:MAC:leading_edge:x:local"]
    assert wing_x0 == pytest.approx(0.32, abs=1e-2)
    wing_y0 = problem["data:geometry:wing:MAC:y"]
    assert wing_y0 == pytest.approx(7.27, abs=1e-2)


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
    mfw = problem["data:weight:aircraft:MFW"]
    assert mfw == pytest.approx(3827, abs=1)


def test_geometry_wing_sweep():
    """ Tests computation of the wing sweeps """

    # Define input values calculated from other modules
    ivc = om.IndepVarComp()
    ivc.add_output("data:geometry:wing:root:y", 0.6)
    ivc.add_output("data:geometry:wing:tip:y", 17.2)
    ivc.add_output("data:geometry:wing:root:chord", 5.17)
    ivc.add_output("data:geometry:wing:tip:chord", 1.96)
    ivc.add_output("data:geometry:wing:tip:leading_edge:x:local", 0.8)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingSweep(), ivc)
    sweep_0 = problem["data:geometry:wing:sweep_0"]
    assert sweep_0 == pytest.approx(2.75, abs=1e-2)
    sweep_100_inner = problem["data:geometry:wing:sweep_100_inner"]
    assert sweep_100_inner == pytest.approx(-8.26, abs=1e-1)
    sweep_100_outer = problem["data:geometry:wing:sweep_100_outer"]
    assert sweep_100_outer == pytest.approx(-8.26, abs=1e-1)


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
    assert toc_root == pytest.approx(0.159, abs=1e-3)
    toc_kink = problem["data:geometry:wing:kink:thickness_ratio"]
    assert toc_kink == pytest.approx(0.121, abs=1e-3)
    toc_tip = problem["data:geometry:wing:tip:thickness_ratio"]
    assert toc_tip == pytest.approx(0.11, abs=1e-2)


def test_geometry_wing_wet_area():
    """ Tests computation of the wing wet area """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeWingWetArea(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:wing:root:virtual_chord", 5.17)
    ivc.add_output("data:geometry:fuselage:maximum_width", 1.20)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeWingWetArea(), ivc)
    area_pf = problem["data:geometry:wing:outer_area"]
    assert area_pf == pytest.approx(118.6, abs=1e-1)
    wet_area = problem["data:geometry:wing:wet_area"]
    assert wet_area == pytest.approx(253.8, abs=1e-1)


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
    span = problem["data:geometry:horizontal_tail:span"]
    assert span == pytest.approx(12.28, abs=1e-2)
    root_chord = problem["data:geometry:horizontal_tail:root:chord"]
    assert root_chord == pytest.approx(4.406, abs=1e-3)
    tip_chord = problem["data:geometry:horizontal_tail:tip:chord"]
    assert tip_chord == pytest.approx(1.322, abs=1e-3)


def test_compute_ht_mac():
    """ Tests computation of the horizontal tail mac """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeHTMAC(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:horizontal_tail:span", 12.28)
    ivc.add_output("data:geometry:horizontal_tail:root:chord", 4.406)
    ivc.add_output("data:geometry:horizontal_tail:tip:chord", 1.322)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTMAC(), ivc)
    length = problem["data:geometry:horizontal_tail:MAC:length"]
    assert length == pytest.approx(3.141, abs=1e-3)
    ht_x0 = problem["data:geometry:horizontal_tail:MAC:at25percent:x:local"]
    assert ht_x0 == pytest.approx(1.656, abs=1e-3)
    ht_y0 = problem["data:geometry:horizontal_tail:MAC:y"]
    assert ht_y0 == pytest.approx(2.519, abs=1e-3)


def test_compute_ht_distance():
    """ Tests computation of the horizontal tail distance """

    # Input list from model (not generated because of the assertion error on bad propulsion layout configuration)
    input_list = [
        "data:geometry:fuselage:length",
        "data:geometry:wing:MAC:length",
        "data:geometry:horizontal_tail:span",
        "data:geometry:propulsion:layout",
        "data:geometry:has_T_tail",
    ]

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:fuselage:length", 43)
    ivc.add_output("data:geometry:wing:MAC:length", 3.141)
    ivc.add_output("data:geometry:horizontal_tail:span", 12.28)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTDistance(), ivc)
    ht_x1 = problem["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
    assert ht_x1 == pytest.approx(21.5, abs=1e-1)
    height = problem["data:geometry:horizontal_tail:z:from_wingMAC25"]
    assert height == pytest.approx(0, abs=1e-1)


def test_compute_ht_sweep():
    """ Tests computation of the horizontal tail sweep """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeHTSweep(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:horizontal_tail:span", 12.28)
    ivc.add_output("data:geometry:horizontal_tail:root:chord", 4.406)
    ivc.add_output("data:geometry:horizontal_tail:tip:chord", 1.322)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeHTSweep(), ivc)
    sweep_0 = problem["data:geometry:horizontal_tail:sweep_0"]
    assert sweep_0 == pytest.approx(33.317, abs=1e-3)
    sweep_100 = problem["data:geometry:horizontal_tail:sweep_100"]
    assert sweep_100 == pytest.approx(8.81, abs=1e-2)
    aspect_ratio = problem["data:geometry:horizontal_tail:aspect_ratio"]
    assert aspect_ratio == pytest.approx(4.28, abs=1e-2)


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
    wet_area = problem["data:geometry:horizontal_tail:wet_area"]
    assert wet_area == pytest.approx(59.08, abs=1e-2)


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
    span = problem["data:geometry:vertical_tail:span"]
    assert span == pytest.approx(6.62, abs=1e-2)
    root_chord = problem["data:geometry:vertical_tail:root:chord"]
    assert root_chord == pytest.approx(5.837, abs=1e-3)
    tip_chord = problem["data:geometry:vertical_tail:tip:chord"]
    assert tip_chord == pytest.approx(1.751, abs=1e-3)


def test_compute_vt_mac():
    """ Tests computation of the vertical tail mac """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeVTMAC(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:vertical_tail:span", 6.62)
    ivc.add_output("data:geometry:vertical_tail:root:chord", 5.837)
    ivc.add_output("data:geometry:vertical_tail:tip:chord", 1.751)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTMAC(), ivc)
    length = problem["data:geometry:vertical_tail:MAC:length"]
    assert length == pytest.approx(4.16, abs=1e-2)
    vt_x0 = problem["data:geometry:vertical_tail:MAC:at25percent:x:local"]
    assert vt_x0 == pytest.approx(2.32, abs=1e-2)
    vt_z0 = problem["data:geometry:vertical_tail:MAC:z"]
    assert vt_z0 == pytest.approx(2.71, abs=1e-2)


def test_compute_vt_distance():
    """ Tests computation of the vertical tail distance """

    # Input list from model (not generated because of the assertion error on bad propulsion layout configuration)
    input_list = [
        "data:geometry:fuselage:length",
        "data:geometry:wing:MAC:at25percent:x",
        "data:geometry:horizontal_tail:span",
        "data:geometry:propulsion:layout",
        "data:geometry:has_T_tail",
    ]

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:fuselage:length", 43)
    ivc.add_output("data:geometry:wing:MAC:length", 3.141)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTDistance(), ivc)
    lp_vt = problem["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
    assert lp_vt == pytest.approx(21.5, abs=1e-1)


def test_compute_vt_sweep():
    """ Tests computation of the vertical tail sweep """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeVTSweep(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:vertical_tail:span", 6.62)
    ivc.add_output("data:geometry:vertical_tail:root:chord", 5.837)
    ivc.add_output("data:geometry:vertical_tail:tip:chord", 1.751)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeVTSweep(), ivc)
    sweep_0 = problem["data:geometry:vertical_tail:sweep_0"]
    assert sweep_0 == pytest.approx(40.5, abs=1e-1)
    sweep_100 = problem["data:geometry:vertical_tail:sweep_100"]
    assert sweep_100 == pytest.approx(13.3, abs=1e-1)


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
    wet_area = problem["data:geometry:vertical_tail:wet_area"]
    assert wet_area == pytest.approx(52.74, abs=1e-2)


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
    ivc.add_output("data:geometry:fuselage:maximum_width", 1.20)
    ivc.add_output("data:geometry:wing:span", 34.4)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeNacelleGeometry(), ivc)
    nacelle_length = problem["data:geometry:propulsion:nacelle:length"]
    assert nacelle_length == pytest.approx(4.09, abs=1e-2)
    nacelle_dia = problem["data:geometry:propulsion:nacelle:diameter"]
    assert nacelle_dia == pytest.approx(2.2, abs=1e-2)
    nacelle_hei = problem["data:geometry:propulsion:nacelle:height"]
    assert nacelle_hei == pytest.approx(2.2, abs=1e-2)
    nacelle_wid = problem["data:geometry:propulsion:nacelle:width"]
    assert nacelle_wid == pytest.approx(2.2, abs=1e-2)
    nacelle_wet_area = problem["data:geometry:propulsion:nacelle:wet_area"]
    assert nacelle_wet_area == pytest.approx(36.07, abs=1e-2)
    lg_height = problem["data:geometry:landing_gear:height"]
    assert lg_height == pytest.approx(3.08, abs=1e-2)
    y_nacell = problem["data:geometry:propulsion:nacelle:y"]
    assert y_nacell == pytest.approx(5.84, abs=1e-2)


def test_geometry_total_area():
    """ Tests computation of the total area """

    # Generate input list from model
    group = om.Group()
    group.add_subsystem("my_model", ComputeTotalArea(), promotes=["*"])
    input_list = list_inputs(group)

    # Research independent input value in .xml file and add values calculated from other modules
    ivc = get_indep_var_comp(input_list)
    ivc.add_output("data:geometry:fuselage:wet_area", 134.3)
    ivc.add_output("data:geometry:wing:wet_area", 253.8)
    ivc.add_output("data:geometry:horizontal_tail:wet_area", 59.08)
    ivc.add_output("data:geometry:vertical_tail:wet_area", 52.74)

    # Run problem and check obtained value(s) is/(are) correct
    problem = run_system(ComputeTotalArea(), ivc)
    total_surface = problem["data:geometry:aircraft:wet_area"]
    assert total_surface == pytest.approx(543.1, abs=1e-1)