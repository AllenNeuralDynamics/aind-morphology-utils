import json
from pathlib import Path
from typing import Union, Any

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


def fix_swc_whitespace(input_swc_path: str, output_swc_path: str) -> None:
    """
    Process an SWC file, replacing any sequence of whitespace characters with a single space.

    This function reads each line in the input SWC file, splits the line into parts at any
    sequence of whitespace characters, then joins the parts back together with a single space
    between each. The processed lines are written to the output SWC file.

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
            # split the line into a list based on any type of whitespace
            parts = line.split()
            # join the elements into a string with a single-space separator
            new_line = " ".join(parts)
            # write the processed line to the output file, adding a newline character
            output_file.write(new_line + "\n")


def read_json(file_path: Union[str, Path]) -> Any:
    """
    Read a JSON file and return its content as a Python object.

    Parameters
    ----------
    file_path : str or pathlib.Path
        The path to the JSON file to be read.

    Returns
    -------
    Any
        The Python object represented by the JSON file. The exact type (e.g., dict, list)
        depends on the structure of the JSON file.
    """
    with open(file_path, "r") as file:
        return json.load(file)
