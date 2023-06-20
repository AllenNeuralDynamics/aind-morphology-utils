import json
import os
import tempfile
import unittest

from allensdk.core.swc import read_swc

from aind_morphology_utils.writers import MouseLightJsonWriter
from tests.utils import get_test_swc_path


class TestMouseLightJsonWriter(unittest.TestCase):
    def setUp(self):
        self.morphology = read_swc(str(get_test_swc_path()))
        self.writer = MouseLightJsonWriter(
            self.morphology,
            id_string="AA1543",
            sample={"sample": "sample"},
            label={"label": "label"},
            comment="comment",
        )
        self.output_path = tempfile.mktemp()

    def tearDown(self) -> None:
        if os.path.exists(self.output_path):
            os.remove(self.output_path)

    def test_write(self):
        # Call the method to be tested
        self.writer.write(self.output_path)

        # Check if the file was created
        self.assertTrue(
            os.path.exists(self.output_path), "Output file was not created."
        )

        # Open the file and load the data
        with open(self.output_path, "r") as f:
            data = json.load(f)

        # Perform assertions about the structure of the data
        self.assertEqual("comment", data["comment"])
        self.assertEqual(1, len(data["neurons"]))
        self.assertEqual("AA1543", data["neurons"][0]["idString"])
        self.assertEqual({"sample": "sample"}, data["neurons"][0]["sample"])
        self.assertEqual({"label": "label"}, data["neurons"][0]["label"])
        self.assertEquals(
            {
                "version": 3,
                "description": "Annotation Space: CCFv3.0 Axes> X: Anterior-Posterior; Y: Inferior-Superior; "
                "Z:Left-Right",
            },
            data["neurons"][0]["annotationSpace"],
        )
        self.assertEqual(8755, len(data["neurons"][0]["axon"]))
        self._check_samples(data["neurons"][0]["axon"])

        self.assertEqual(2321, len(data["neurons"][0]["dendrite"]))
        self._check_samples(data["neurons"][0]["dendrite"])

        self.assertEqual(4, len(data["neurons"][0]["soma"]))
        self._check_soma(data["neurons"][0]["soma"])

        # This will be empty unless we map annotations first
        self.assertEquals(0, len(data["neurons"][0]["allenInformation"]))

    def test_write_indent(self):
        self.writer.write(self.output_path, indent=2)

        # Open the file and check the first few characters to infer indentation
        with open(self.output_path, "r") as f:
            start = f.read(10)
        self.assertTrue(
            start.startswith('{\n  "'),
            "Incorrect indentation in written file.",
        )

    def test_write_invalid_path(self):
        with self.assertRaises(Exception):
            self.writer.write("/invalid/path/output.json")

    def _check_soma(self, soma):
        self.assertIsInstance(soma["x"], float)
        self.assertIsInstance(soma["y"], float)
        self.assertIsInstance(soma["z"], float)
        self.assertIn("allenId", soma)

    def _check_samples(self, sample_list):
        for s in sample_list:
            self.assertIsInstance(s["sampleNumber"], int)
            self.assertIsInstance(s["structureIdentifier"], int)
            self.assertIsInstance(s["x"], float)
            self.assertIsInstance(s["y"], float)
            self.assertIsInstance(s["z"], float)
            self.assertIsInstance(s["radius"], float)
            self.assertIsInstance(s["parentNumber"], int)
            self.assertIn("allenId", s)


if __name__ == "__main__":
    unittest.main()
