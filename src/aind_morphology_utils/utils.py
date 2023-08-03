import json
import os
from glob import glob
from typing import Union, Any, Tuple, Set, List

import numpy
import numpy as np
import zarr
from allensdk.core import swc
from allensdk.core.swc import Compartment, Morphology
from ants import read_transform, transform_from_displacement_field, image_read


def get_ccf_id(compartment: Compartment) -> Union[int, None]:
    """
    Get the CCF region ID of the given compartment.

    Parameters
    ----------
    compartment : Compartment
        The compartment to analyze.

    Returns
    -------
    ccf_region_id : int
        The CCF region ID, or None if not found.
    """
    try:
        return compartment["allenInformation"]["id"]
    except KeyError:
        return None


def get_structure_types(morphology: Morphology) -> Set[int]:
    """
    Get the set of structure types in the given Morphology object.

    Parameters
    ----------
    morphology : Morphology
        The Morphology object to analyze.

    Returns
    -------
    type_set : set
        The set of structure types.
    """
    type_set = set()
    for c in morphology.compartment_list:
        type_set.add(c[swc.NODE_TYPE])
    return type_set


def rgb_to_hex(rgb_tuple: Tuple[int, int, int]) -> str:
    """
    Convert an RGB tuple into a hexadecimal color string.
    Does not check if RGB values are outside the range (0,255).

    Parameters
    ----------
    rgb_tuple : tuple
        The RGB tuple to convert.

    Returns
    -------
    hex_color : str
        The hexadecimal color string.
    """
    return "#%02x%02x%02x" % rgb_tuple


def fix_swc_whitespace(
        input_swc_path: Union[str, os.PathLike],
        output_swc_path: Union[str, os.PathLike]
) -> None:
    """
    Process an SWC file, replacing any sequence of whitespace characters with a single space.

    Parameters
    ----------
    input_swc_path : str
        The path to the input SWC file.
    output_swc_path : str
        The path to the output SWC file.
    """
    with open(input_swc_path, "r") as input_file, open(
            output_swc_path, "w"
    ) as output_file:
        for line in input_file:
            parts = line.split()
            new_line = " ".join(parts)
            output_file.write(new_line + "\n")


def read_json(file_path: Union[str, os.PathLike]) -> Any:
    """
    Read a JSON file and return its content as a Python object.

    Parameters
    ----------
    file_path : str or pathlib.Path
        The path to the JSON file to be read.

    Returns
    -------
    Any
        The Python object represented by the JSON file.
    """
    with open(file_path, "r") as file:
        return json.load(file)


def read_swc_offset(swc_path: Union[str, os.PathLike]) -> Union[None, numpy.ndarray]:
    """
    Read the offset from an SWC file.

    Parameters
    ----------
    swc_path : str or pathlib.Path
        The path to the SWC file to be read.

    Returns
    -------
    numpy.ndarray
        The offset from the SWC file as a NumPy array with shape (3,), or None if not found.
    """
    with open(swc_path, "r") as file:
        for line in file:
            if line.startswith("# OFFSET"):
                return np.array([float(x) for x in line.split()[2:5]])
    return None


def read_swc(swc_path: Union[str, os.PathLike], add_offset: bool = True) -> Morphology:
    """
    Read an SWC file and return its content as a Morphology object.
    Optionally add the offset from the file header to the node coordinates,
    if it exists.

    Parameters
    ----------
    swc_path : str or pathlib.Path
        The path to the SWC file to be read.
    add_offset : bool, optional
        If True, the offset from the SWC file header will be added to the
        node coordinates. Default is True.

    Returns
    -------
    Morphology
        The Morphology object represented by the SWC file.
    """
    morph = swc.read_swc(swc_path)

    if add_offset:
        offset = read_swc_offset(swc_path)
        if offset is not None:
            for node in morph.compartment_list:
                node['x'] += offset[0]
                node['y'] += offset[1]
                node['z'] += offset[2]

    return morph


def read_registration_transform(reg_path: Union[str, os.PathLike]) -> Tuple[Any, Any]:
    """
    Imports ants transformation from registration output

    Parameters
    -------------
    reg_path: str or pathlib.Path
        Path to .gz file from registration

    Returns
    -------------
    Tuple[Any, Any]
        affine transform and nonlinear warp field from ants.registration()
    """
    affine_file = glob(os.path.join(reg_path, '*.mat'))[0]
    affine = read_transform(affine_file)
    affinetx = affine.invert()
    warp_file = glob(os.path.join(reg_path, '*.gz'))[0]
    warp = image_read(warp_file)
    warptx = transform_from_displacement_field(warp)

    return affinetx, warptx


def get_voxel_size_image(image_path: Union[str, os.PathLike], input_scale: int) -> Tuple[float, float, float]:
    """
    Get the size of the scaled image at the specified path and scale.

    Parameters
    ----------
    image_path : str or pathlib.Path
        The path to the image file.
    input_scale : int
        The input scale.

    Returns
    -------
    Tuple[float, float, float]
        The sizes of the scaled image in x, y, and z.
    """
    ds = zarr.open(image_path, mode="r")[str(input_scale)]
    sx = ds.shape[-1] * 2 ** input_scale
    sy = ds.shape[-2] * 2 ** input_scale
    sz = ds.shape[-3] * 2 ** input_scale
    return sx, sy, sz


def flip_pt(pt: List[float], s: List[float], flip_axes: List[int]) -> List[float]:
    """
    Flip the specified axes of a point.

    Parameters
    ----------
    pt : List[float]
        The point to flip.
    s : List[float]
        The sizes of the dimensions.
    flip_axes : List[bool]
        A list of integers indicating which axes to flip.

    Returns
    -------
    List[float]
        The point with the specified axes flipped.
    """
    for fa in flip_axes:
        pt[fa] = s[fa] - pt[fa]
    return pt
