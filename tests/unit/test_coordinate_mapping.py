import unittest

import numpy as np
import zarr

from aind_morphology_utils.coordinate_mapping import OMEZarrTransform
from allensdk.core.swc import Morphology, Compartment


class TestOMEZarrTransform(unittest.TestCase):
    def setUp(self):
        # Define a known transformation matrix and vector field
        self.transformation_matrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        self.vector_field = np.zeros((3, 10, 10, 10))
        self.vector_field[:, 1, 2, 3] = np.array(
            [1, -2, 3]
        )  # Known displacement at position (1, 2, 3)

        # Create the mock HDF5 file with the known transformation matrix and vector field
        self.z = zarr.open("mock.zarr", "w")
        self.z["DisplacementField/0"] = self.vector_field
        self.z["TransformationMatrix"] = self.transformation_matrix

    def test_transform(self):
        morph_data = [
            Compartment(
                {
                    "id": 1,
                    "type": 1,
                    "x": 1.0,
                    "y": 2.0,
                    "z": 3.0,
                    "radius": 1.0,
                    "parent": -1,
                }
            )
        ]
        original_morph = Morphology(morph_data)

        transformer = OMEZarrTransform("mock.zarr")

        transformed_morph = transformer.transform(original_morph)

        # Check if the vectors were added correctly
        expected_x = original_morph.compartment_list[0]["x"] + 1
        expected_y = original_morph.compartment_list[0]["y"] - 2
        expected_z = original_morph.compartment_list[0]["z"] + 3
        self.assertEqual(
            transformed_morph.compartment_list[0]["x"], expected_x
        )
        self.assertEqual(
            transformed_morph.compartment_list[0]["y"], expected_y
        )
        self.assertEqual(
            transformed_morph.compartment_list[0]["z"], expected_z
        )


if __name__ == "__main__":
    unittest.main()
