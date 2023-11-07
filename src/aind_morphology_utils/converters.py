import logging
from pathlib import Path

import numpy as np
import h5py
import nrrd


class NRRDToHDF5:
    """
    A class to handle the conversion of displacement fields created in 3D Slicer from NRRD to HDF5 format.

    Attributes
    ----------
    disp : ndarray
        The displacement field data.
    tmat : ndarray
        The transformation matrix associated with the displacement field.
    """

    def __init__(self, nrrd_file: str):
        """
        Parameters
        ----------
        nrrd_file : str
            Path to the NRRD file containing the displacement field.
        """
        self.disp, header = nrrd.read(nrrd_file)
        self._process_header(header)

    def _process_header(self, header: dict, voxel_size: int = 25) -> None:
        """
        Process the header information from the NRRD file to extract the transformation matrix.

        Parameters
        ----------
        header : dict
            Header information from the NRRD file.
        """
        if header["space"] in ["left-posterior-superior", "LPS"]:
            # Change orientation from LPS to RAS
            self.disp[0:2, :, :, :] = -self.disp[0:2, :, :, :]

        space_directions = header["space directions"]
        # First row is nan for some reason??
        space_directions = space_directions[
            ~np.all(np.isnan(space_directions), axis=1)
        ]
        assert space_directions.shape == (3, 3)
        # make sure it's diagonal
        assert (
            np.count_nonzero(
                space_directions - np.diag(np.diagonal(space_directions))
            )
            == 0
        )
        # if all the diagonal elements are 1, multiply by the voxel size
        if np.all(np.diagonal(space_directions) == 1):
            logging.warning(
                "Space directions are all 1, so multiplying by voxel size."
            )
            space_directions = space_directions * voxel_size
        logging.info(f"Space directions: {space_directions}")

        origin = header["space origin"]
        assert origin.shape == (3,)
        logging.info(f"Space origin: {origin}")

        tmat = np.vstack((space_directions, origin))
        # Make homogeneous
        tmat = np.hstack((tmat, [[0], [0], [0], [1]]))

        self.tmat = np.linalg.inv(tmat.T)

    def save_hdf5(self, output_file: str) -> None:
        """
        Save the displacement field and transformation matrix to an HDF5 file.

        Parameters
        ----------
        output_folder : str
            Path to the folder where the HDF5 file will be saved.
        sample_name : str
            Name of the sample to include in the HDF5 file name.
        """
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_file, "w") as h5file:
            h5file.create_dataset("/DisplacementField", data=self.disp)
            h5file["/DisplacementField"].attrs[
                "Transformation_Matrix"
            ] = self.tmat
