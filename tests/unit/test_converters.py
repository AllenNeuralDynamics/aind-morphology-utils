import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import h5py

from aind_morphology_utils.converters import NRRDToHDF5


class TestNRRDToHDF5(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)

        # Create mock data for the NRRD content
        self.mock_nrrd_data = np.random.rand(
            5, 5, 5, 3
        )  # Random 3D vector field
        self.mock_nrrd_header = {
            "space": "left-posterior-superior",
            "space directions": np.array(
                [[np.nan, np.nan, np.nan], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
            ),
            "space origin": np.array([0, 0, 0]),
        }

    @patch("nrrd.read")
    def test_save_hdf5(self, mock_nrrd_read):
        # Mock nrrd.read to return the mock data and header
        mock_nrrd_read.return_value = (
            self.mock_nrrd_data,
            self.mock_nrrd_header,
        )

        # Use a temporary file for the HDF5 output
        with tempfile.NamedTemporaryFile(
            dir=self.test_dir.name, delete=False
        ) as tmp_file:
            converter = NRRDToHDF5("dummy_path")
            converter.save_hdf5(tmp_file.name)

            # Open the file outside the context to avoid file lock issues on some systems
            output_file = tmp_file.name

        # Check if HDF5 file is created with correct content
        with h5py.File(output_file, "r") as h5file:
            expected_tmat = np.eye(4)
            expected_tmat[:, 0:3] /= 25
            np.testing.assert_array_equal(
                h5file["/DisplacementField"][...], self.mock_nrrd_data
            )
            np.testing.assert_array_equal(
                h5file["/DisplacementField"].attrs["Transformation_Matrix"],
                expected_tmat,
            )


if __name__ == "__main__":
    unittest.main()
