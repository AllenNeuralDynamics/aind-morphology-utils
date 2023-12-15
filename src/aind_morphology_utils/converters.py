import logging
from pathlib import Path

import numpy as np
import nrrd
import zarr
import ome_zarr.writer


class NRRDToOMEZarr:
    def __init__(self, nrrd_file: str):
        """
        Parameters
        ----------
        nrrd_file : str
            Path to the NRRD file containing the displacement field.
        """
        self.disp, header = nrrd.read(nrrd_file)
        self._process_header(header)

    def _process_header(self, header: dict, voxel_size: int = 25):
        """
        Process the header information from the NRRD file to extract the transformation matrix.

        Parameters
        ----------
        header : dict
            Header information from the NRRD file.
        voxel_size : int
            The size of the voxels in microns.
        """
        if header["space"] in ["left-posterior-superior", "LPS"]:
            # Change orientation from LPS to RAS
            self.disp[0:2, :, :, :] = -self.disp[0:2, :, :, :]

        space_directions = header["space directions"]
        space_directions = space_directions[~np.all(np.isnan(space_directions), axis=1)]
        assert space_directions.shape == (3, 3)
        assert np.count_nonzero(space_directions - np.diag(np.diagonal(space_directions))) == 0

        if np.all(np.diagonal(space_directions) == 1):
            logging.warning("Space directions are all 1, so multiplying by voxel size.")
            space_directions *= voxel_size

        logging.info(f"Space directions: {space_directions}")

        origin = header["space origin"]
        assert origin.shape == (3,)
        logging.info(f"Space origin: {origin}")

        tmat = np.vstack((space_directions, origin))
        tmat = np.hstack((tmat, [[0], [0], [0], [1]]))

        self.tmat = np.linalg.inv(tmat.T)

    def save(self, out_zarr: str) -> None:
        Path(out_zarr).mkdir(parents=True, exist_ok=True)

        zarr_file = zarr.open(out_zarr, mode="w")
        group = zarr_file.create_group("DisplacementField")

        ome_zarr.writer.write_image(
            self.disp,
            group,
            scaler=None,
            storage_options={"chunks": (3, 64, 64, 64)},
            axes=["c", "x", "y", "z"],
        )

        zarr_file.create_dataset("TransformationMatrix", data=self.tmat)
