import os
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

    def __init__(self, nrrd_file):
        """
        Parameters
        ----------
        nrrd_file : str
            Path to the NRRD file containing the displacement field.
        """
        self.disp, header = nrrd.read(nrrd_file)
        # Change orientation from LPS to RAS
        self.disp[0:2, :, :, :] = -self.disp[0:2, :, :, :]
        self._process_header(header)

    def _process_header(self, header):
        """
        Process the header information from the NRRD file to extract the transformation matrix.

        Parameters
        ----------
        header : dict
            Header information from the NRRD file.
        """
        space_directions = header['space directions']
        # First row is nan for some reason??
        space_directions = space_directions[~np.all(np.isnan(space_directions), axis=1)] * 25
        assert space_directions.shape == (3, 3)

        origin = header['space origin']

        tmat = np.vstack((space_directions, origin))
        # Make homogeneous
        tmat = np.hstack((tmat, [[0], [0], [0], [1]]))

        self.tmat = np.linalg.inv(tmat.T)

    def save_to_hdf5(self, output_folder, sample_name):
        """
        Save the displacement field and transformation matrix to an HDF5 file.

        Parameters
        ----------
        output_folder : str
            Path to the folder where the HDF5 file will be saved.
        sample_name : str
            Name of the sample to include in the HDF5 file name.
        """
        output_file = os.path.join(output_folder, f'Transform.{sample_name}.h5')
        if os.path.exists(output_file):
            os.remove(output_file)

        with h5py.File(output_file, 'w') as h5file:
            h5file.create_dataset('/DisplacementField', data=self.disp)
            h5file['/DisplacementField'].attrs['Transformation_Matrix'] = self.tmat


if __name__ == '__main__':
    nrrd_file = r"C:\Users\cameron.arshadi\Downloads\Displacement Field_653980.nrrd"
    sample_name = '653158'
    output_folder = r'C:\Users\cameron.arshadi\Downloads'

    converter = NRRDToHDF5(nrrd_file)
    converter.save_to_hdf5(output_folder, sample_name)
    print('\nDone!\n')
