import logging
from typing import List

import numpy as np
from allensdk.core.swc import Morphology

from aind_morphology_utils.utils import (
    read_registration_transform,
    get_voxel_size_image,
    flip_pt
)

_LOGGER = logging.getLogger(__name__)


class MorphologyTransformer:
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
