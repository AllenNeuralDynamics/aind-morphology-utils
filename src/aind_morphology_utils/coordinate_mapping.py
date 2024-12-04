import logging
import os
from copy import deepcopy
from typing import List

import numpy as np
import zarr
from allensdk.core.swc import Morphology

from aind_morphology_utils.utils import (
    read_registration_transform,
    get_voxel_size_image,
    flip_pt,
    read_swc,
)

_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)

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
        affine_only: bool = False,
    ):
        self.affinetx, self.warptx = read_registration_transform(
            registration_folder, affine_only
        )
        self.sx, self.sy, self.sz = get_voxel_size_image(
            image_path
        )
        self.transform_res = transform_res
        self.input_res = input_res
        self.swc_scale = swc_scale
        self.flip_axes = flip_axes

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

        scale = [
            raw / trans
            for trans, raw in zip(self.transform_res, self.input_res)
        ]

        morph_copy = deepcopy(
            morph
        )  # TODO: build new Morphology from scratch instead of deep copying
        for node in morph_copy.compartment_list:
            pt = np.array([node["x"], node["y"], node["z"]]) / np.array(
                self.swc_scale
            )
            pt = flip_pt(pt, [self.sx, self.sy, self.sz], self.flip_axes)
            scaled_pt = [dim * scale for dim, scale in zip(pt, scale)]
            affine_pt = self.affinetx.apply_to_point(scaled_pt)
            if self.warptx is None:
                # _LOGGER.info("No warp transform found, using affine only")
                warp_pt = affine_pt
            else:
                warp_pt = self.warptx.apply_to_point(affine_pt)
            scaled_warp_pt = [
                dim * scale for dim, scale in zip(warp_pt, self.transform_res)
            ]
            node["x"] = scaled_warp_pt[0]
            node["y"] = scaled_warp_pt[1]
            node["z"] = scaled_warp_pt[2]

        return morph_copy
    
    def inverse_transform(self, morph: Morphology) -> Morphology:
        """
        Inverse Transform the given Morphology from CCF space to imaging space.

        Parameters
        ----------
        morph: Morphology
            the morphology object.

        Returns
        -------
        allensdk.core.swc.Morphology
            The transformed Morphology object.
        """

        #Invert the entire procedure.
        _LOGGER.info("# points: " + str(len(morph.compartment_list)))

        scale = [
            raw / trans
            for trans, raw in zip(self.transform_res, self.input_res)
        ]

        morph_copy = deepcopy(
            morph
        )  # TODO: build new Morphology from scratch instead of deep copying
        for node in morph_copy.compartment_list:
            
            scaled_warp_pt = [ node["x"] , node["y"], node["z"] ]
            warp_pt = [dim / scale for dim, scale in zip(scaled_warp_pt, self.transform_res)]
            
            if self.warptx is None:
                affine_pt = warp_pt
            else:
                #invert warp field
                affine_pt = self.warptx.apply_to_point(warp_pt) #assume warptx is inverted

            scaled_pt = self.affinetx.apply_to_point(affine_pt)#assume affinetx is inverted
            pt = [dim / scale for dim, scale in zip(scaled_pt, scale)]
            pt = flip_pt(pt, [self.sx, self.sy, self.sz], self.flip_axes)
            pt = pt* np.array(self.swc_scale)

            node["x"] = pt[0]
            node["y"] = pt[1]
            node["z"] = pt[2]

        return morph_copy


class OMEZarrTransform:
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
            Path to the Zarr file containing the transformation matrix and vector field.
        """
        z = zarr.open(transform_file, mode="r")
        self.transform_matrix = z["TransformationMatrix"][:]
        self.vector_field = z["DisplacementField"]['0'][:]

    def transform(self, morph) -> Morphology:
        """
        Apply the transformation to a set of points.

        Parameters
        ----------
        morph : Morphology
            The morphology to transform.

        Returns
        -------
        Morphology
            The transformed morphology.
        """
        vec = self._get_disp_vec(morph)

        morph_copy = deepcopy(
            morph
        )  # TODO: build new Morphology from scratch instead of deep copying
        # Apply displacement vectors to points
        for i, node in enumerate(morph_copy.compartment_list):
            node["x"], node["y"], node["z"] = np.add(
                [node["x"], node["y"], node["z"]], vec[i]
            )

        return morph_copy

    def _get_disp_vec(self, morph: Morphology) -> np.ndarray:
        """
        Get the displacement vectors for the given morphology.
        Parameters
        ----------
        morph : Morphology
            The morphology to transform.

        Returns
        -------
        np.ndarray
            The displacement vectors.
        """
        # Extract coordinates and convert them to homogeneous coordinates
        points = np.array(
            [[c["x"], c["y"], c["z"], 1] for c in morph.compartment_list]
        )

        # Convert micron points to voxel coordinates
        pix_pos = (
            np.ceil(np.dot(self.transform_matrix, points.T)).astype(int).T
        )

        vector_field_shape = np.array(self.vector_field.shape[1:])

        # Clip the first three columns of pix_pos (the spatial dimensions)
        pix_pos[:, :3] = np.clip(pix_pos[:, :3], 0, vector_field_shape - 1)

        # Extract displacement vectors
        vec = self.vector_field[0, pix_pos[:, 0], pix_pos[:, 1], pix_pos[:, 2]]
        vec = np.vstack(
            (
                vec,
                self.vector_field[
                    1, pix_pos[:, 0], pix_pos[:, 1], pix_pos[:, 2]
                ],
            )
        )
        vec = np.vstack(
            (
                vec,
                self.vector_field[
                    2, pix_pos[:, 0], pix_pos[:, 1], pix_pos[:, 2]
                ],
            )
        )

        return vec.T

    def transform_swc_files(
        self, input_folder: str, output_folder: str
    ) -> None:
        """
        Transform all SWC files in the input folder and save the transformed files in the output folder.

        Parameters
        ----------
        input_folder : str
            Path to the folder containing SWC files.
        output_folder : str
            Path to the output folder where transformed SWC files will be saved.
        """
        swc_files = [f for f in os.listdir(input_folder) if f.endswith(".swc")]

        if not swc_files:
            raise ValueError(f"No SWC files found in folder: {input_folder}")

        os.makedirs(output_folder, exist_ok=True)
        for swc_file in swc_files:
            swc_path = os.path.join(input_folder, swc_file)
            morph = read_swc(swc_path)
            morph = self.transform(morph)
            output_path = os.path.join(output_folder, swc_file)
            morph.save(output_path)
