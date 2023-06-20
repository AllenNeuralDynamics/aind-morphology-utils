import os
import math
from pathlib import Path
from typing import Any


def get_test_swc_path() -> Path:
    """
    Get the path to the test SWC file.

    Returns
    -------
    pathlib.Path
        The path to the test SWC file.
    """
    return get_test_resources_dir() / "AA0983-cleaned-whitespace.swc"


def get_test_json_path() -> Path:
    """
    Get the path to the test JSON file.

    Returns
    -------
    pathlib.Path
        The path to the test JSON file.
    """
    return get_test_resources_dir() / "AA0983.json"


def get_test_resources_dir() -> Path:
    """
    Get the path to the test resource directory.

    Returns
    -------
    pathlib.Path
        The path to the test resource directory.
    """
    return Path(os.path.dirname(os.path.abspath(__file__))) / "resources"


def is_close(a: Any, b: Any, rel_tol: float = 1e-9) -> bool:
    """
    Check whether two items are close to each other. For floats, this means they are nearly equal
    with a certain tolerance. For strings, lists and dicts, specific rules are applied.

    Parameters
    ----------
    a : Any
        The first item to compare.
    b : Any
        The second item to compare.
    rel_tol : float, optional
        The relative tolerance for float comparison (default is 1e-9).

    Returns
    -------
    bool
        True if items are close, False otherwise.
    """
    if isinstance(a, float) and isinstance(b, float):
        isclose = math.isclose(a, b, rel_tol=rel_tol)
        return isclose
    if isinstance(a, dict) and isinstance(b, dict):
        return dict_equal(a, b, rel_tol=rel_tol)
    if isinstance(a, list) and isinstance(b, list):
        return list_equal(a, b, rel_tol=rel_tol)
    if isinstance(a, str) and isinstance(b, str):
        # Json files downloaded from the MouseLight Neuron Browser
        # sometimes have commas removed from the structure names.
        return a.replace(",", "") == b.replace(",", "")
    return a == b


def list_equal(l1: list, l2: list, rel_tol: float = 1e-9) -> bool:
    """
    Check whether two lists are close to each other. Lists are considered close if they have the same length
    and all corresponding elements are close.

    Parameters
    ----------
    l1 : list
        The first list to compare.
    l2 : list
        The second list to compare.
    rel_tol : float, optional
        The relative tolerance for float comparison (default is 1e-9).

    Returns
    -------
    bool
        True if lists are close, False otherwise.
    """
    if len(l1) != len(l2):
        return False
    return all(is_close(a, b, rel_tol) for a, b in zip(l1, l2))


def dict_equal(d1: dict, d2: dict, rel_tol: float = 1e-9) -> bool:
    """
    Recursively check whether two dictionaries are equivalent, allowing a tolerance for floating point comparisons.

    Parameters
    ----------
    d1 : dict
        The first dictionary to compare.
    d2 : dict
        The second dictionary to compare.
    rel_tol : float, optional
        The relative tolerance for float comparison (default is 1e-9).

    Returns
    -------
    bool
        True if dictionaries are close, False otherwise.
    """
    if d1.keys() != d2.keys():
        return False
    return all(is_close(v, d2[k], rel_tol) for k, v in d1.items())
