from allensdk.core import swc
from allensdk.core.swc import Compartment, Morphology


def get_ccf_id(compartment: Compartment) -> int:
    """
    Get the CCF region ID of the given compartment.

    Parameters
    ----------
    compartment : Compartment
        The compartment to analyze.

    Returns
    -------
    ccf_region_id : int
        The CCF region ID.

    """
    try:
        return compartment["allenInformation"]["id"]
    except KeyError:
        return None


def get_structure_types(morphology: Morphology) -> set:
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


def rgb_to_hex(rgb_tuple: tuple) -> str:
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
