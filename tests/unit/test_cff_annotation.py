import unittest

from allensdk.core.swc import Compartment, Morphology

from aind_morphology_utils.ccf_annotation import CCFMorphologyMapper
from tests.utils import get_test_resources_dir


def _create_test_morphology():
    # Created using the 100 micron resolution ccf_2017 annotation volume
    compartments = [
        {
            "id": 1,
            "type": 2,
            "x": 4500.0,
            "y": 6400.0,
            "z": 5100.0,
            "radius": 1.0,
            "parent": -1,
            "tree_id": -1,
            "children": [],
        },
        {
            "id": 2,
            "type": 2,
            "x": 3500.0,
            "y": 3100.0,
            "z": 4300.0,
            "radius": 1.0,
            "parent": 1,
            "tree_id": -1,
            "children": [],
        },
        {
            "id": 3,
            "type": 2,
            "x": 10400.0,
            "y": 6300.0,
            "z": 5600.0,
            "radius": 1.0,
            "parent": 2,
            "tree_id": -1,
            "children": [],
        },
        {
            "id": 4,
            "type": 2,
            "x": 2100.0,
            "y": 2400.0,
            "z": 4200.0,
            "radius": 1.0,
            "parent": 3,
            "tree_id": -1,
            "children": [],
        },
    ]
    compartments = [Compartment(**c) for c in compartments]
    return Morphology(compartments)


def _create_expected_morphology():
    compartments = [
        {
            "id": 0,
            "type": 2,
            "x": 4500.0,
            "y": 6400.0,
            "z": 5100.0,
            "radius": 1.0,
            "parent": -1,
            "tree_id": 0,
            "children": [1],
            "allenInformation": {
                "acronym": "HY",
                "graph_id": 1,
                "graph_order": 715,
                "id": 1097,
                "name": "Hypothalamus",
                "structure_id_path": [997, 8, 343, 1129, 1097],
                "structure_set_ids": [
                    2,
                    112905828,
                    691663206,
                    12,
                    184527634,
                    112905813,
                    687527670,
                    114512891,
                    114512892,
                ],
                "rgb_triplet": [230, 68, 56],
            },
        },
        {
            "id": 1,
            "type": 2,
            "x": 3500.0,
            "y": 3100.0,
            "z": 4300.0,
            "radius": 1.0,
            "parent": 0,
            "tree_id": 0,
            "children": [2],
            "allenInformation": {
                "acronym": "MOs6b",
                "graph_id": 1,
                "graph_order": 29,
                "id": 1085,
                "name": "Secondary motor area, layer 6b",
                "structure_id_path": [
                    997,
                    8,
                    567,
                    688,
                    695,
                    315,
                    500,
                    993,
                    1085,
                ],
                "structure_set_ids": [184527634, 12, 667481450, 691663206],
                "rgb_triplet": [31, 157, 90],
            },
        },
        {
            "id": 2,
            "type": 2,
            "x": 10400.0,
            "y": 6300.0,
            "z": 5600.0,
            "radius": 1.0,
            "parent": 1,
            "tree_id": 0,
            "children": [3],
            "allenInformation": {
                "acronym": "GRN",
                "graph_id": 1,
                "graph_order": 975,
                "id": 1048,
                "name": "Gigantocellular reticular nucleus",
                "structure_id_path": [997, 8, 343, 1065, 354, 370, 1048],
                "structure_set_ids": [
                    112905828,
                    691663206,
                    687527945,
                    12,
                    688152367,
                    184527634,
                    112905813,
                    167587189,
                    114512891,
                    114512892,
                ],
                "rgb_triplet": [255, 179, 217],
            },
        },
        {
            "id": 3,
            "type": 2,
            "x": 2100.0,
            "y": 2400.0,
            "z": 4200.0,
            "radius": 1.0,
            "parent": 2,
            "tree_id": 0,
            "children": [],
            "allenInformation": {
                "acronym": "MOs2/3",
                "graph_id": 1,
                "graph_order": 26,
                "id": 962,
                "name": "Secondary motor area, layer 2/3",
                "structure_id_path": [
                    997,
                    8,
                    567,
                    688,
                    695,
                    315,
                    500,
                    993,
                    962,
                ],
                "structure_set_ids": [667481441, 184527634, 12, 691663206],
                "rgb_triplet": [31, 157, 90],
            },
        },
    ]
    compartments = [Compartment(**c) for c in compartments]
    return Morphology(compartments)


class TestCCFMorphologyMapper(unittest.TestCase):
    def test_annotate_morphology(self):
        morph = _create_test_morphology()
        # This will use the test data in tests/resources
        mapper = CCFMorphologyMapper(
            resolution=100, cache_dir=str(get_test_resources_dir())
        )
        mapper.annotate_morphology(morph)
        expected_morph = _create_expected_morphology()
        for i in range(expected_morph.num_nodes):
            self.assertEqual(expected_morph.node(i), morph.node(i))
