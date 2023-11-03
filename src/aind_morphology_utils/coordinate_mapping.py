import logging
import os
from typing import List

import h5py
import numpy as np
from allensdk.core.swc import Morphology

from aind_morphology_utils.utils import (
    read_registration_transform,
    get_voxel_size_image,
    flip_pt, read_swc
)

_LOGGER = logging.getLogger(__name__)


class AntsTransform:
    """
    Class for Transforming Morphologies from imaging space to CCF space.

    Parameters
    ----------
    registration_folder : str
        Path to the folder containing the registration transforms.
    image_path : str
        Path to the image file.
    transform_res : List[float]
        The resolution of the transform.
    input_res : List[float]
        The resolution of the input.
    swc_scale : List[float]
        The scale factors for the SWC file.
    flip_axes : List[bool]
        The axes to flip.
    input_scale : int
        The scale factors for the input.
    """

    def __init__(
            self,
            registration_folder: str,
            image_path: str,
            transform_res: List[float],
            input_res: List[float],
            swc_scale: List[float],
            flip_axes: List[bool],
            input_scale: int
    ):
        self.affinetx, self.warptx = read_registration_transform(registration_folder)
        self.sx, self.sy, self.sz = get_voxel_size_image(image_path, input_scale)
        self.transform_res = transform_res
        self.input_res = input_res
        self.swc_scale = swc_scale
        self.flip_axes = flip_axes
        self.input_scale = input_scale

    def transform(self, morph: Morphology) -> Morphology:
        """
        Transform the given Morphology from imaging space to CCF space.

        Parameters
        ----------
        morph: Morphology
            the morphology object.

        Returns
        -------
        allensdk.core.swc.Morphology
            The transformed Morphology object.
        """

        _LOGGER.info("# points: " + str(len(morph.compartment_list)))

        scale = [raw / trans for trans, raw in zip(self.transform_res, self.input_res)]

        for node in morph.compartment_list:
            pt = np.array([node['x'], node['y'], node['z']]) / np.array(self.swc_scale)
            pt = flip_pt(pt, [self.sx, self.sy, self.sz], self.flip_axes)
            scaled_pt = [dim * scale for dim, scale in zip(pt, scale)]
            affine_pt = self.affinetx.apply_to_point(scaled_pt)
            warp_pt = self.warptx.apply_to_point(affine_pt)
            scaled_warp_pt = [dim * scale for dim, scale in zip(warp_pt, self.transform_res)]
            node['x'] = scaled_warp_pt[0]
            node['y'] = scaled_warp_pt[1]
            node['z'] = scaled_warp_pt[2]

        return morph


class HDF5Transform:
    """
    A class to handle transformations using a displacement vector field.

    Attributes
    ----------
    transform_matrix : ndarray
        The transformation matrix to apply.
    vector_field : ndarray
        The vector field providing displacement vectors.
    """

    def __init__(self, transform_file: str):
        """
        Initialize the Transformer object by loading the transformation matrix and vector field.

        Parameters
        ----------
        transform_file : str
            Path to the HDF5 transformation file.
        """
        with h5py.File(transform_file, 'r') as f:
            self.transform_matrix = f['/DisplacementField'].attrs['Transformation_Matrix']
            self.vector_field = f['/DisplacementField'][...]

    def transform(self, morph: Morphology) -> Morphology:
        """
        Apply the transformation to a set of points.

        Parameters
        ----------
        morph : Morphology
            The morphology to transform.

        Returns
        -------
        Morphology
            Transformed points.
        """
        points = np.array([[c['x'], c['y'], c['z']] for c in morph.compartment_list])
        points = np.hstack((points, np.ones((points.shape[0], 1))))
        pix_pos = np.ceil(np.dot(self.transform_matrix, points.T)).astype(int).T
        vec = np.zeros((points.shape[0], 3))
        for i, pos in enumerate(pix_pos):
            try:
                vec[i] = self.vector_field[:, pos[0], pos[1], pos[2]]
            except IndexError:
                print(f"Index out of bounds at point {i}: {pos}")
        for i, node in enumerate(morph.compartment_list):
            node['x'] += vec[i][0]
            node['y'] += vec[i][1]
            node['z'] += vec[i][2]
        return morph

    def transform_swc_files(self, input_folder: str, output_folder: str):
        """
        Transform all SWC files in the input folder and save the transformed files in the output folder.

        Parameters
        ----------
        input_folder : str
            Path to the folder containing SWC files.
        output_folder : str
            Path to the output folder where transformed SWC files will be saved.
        """
        swc_files = [f for f in os.listdir(input_folder) if f.endswith('.swc')]

        if not swc_files:
            raise ValueError(f'No SWC files found in folder: {input_folder}')

        os.makedirs(output_folder, exist_ok=True)
        for swc_file in swc_files:
            swc_path = os.path.join(input_folder, swc_file)
            morph = read_swc(swc_path)
            morph = self.transform(morph)
            output_path = os.path.join(output_folder, swc_file)
            morph.save(output_path)
