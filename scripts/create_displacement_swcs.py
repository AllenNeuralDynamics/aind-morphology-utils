import logging
import os
import numpy as np
from typing import List, Tuple, Dict
from allensdk.core.swc import Morphology, Compartment
from aind_morphology_utils.utils import read_swc
import matplotlib.pyplot as plt


def calculate_displacements(
    orig_morph: Morphology, trans_morph: Morphology
) -> List[Tuple[Compartment, Compartment, float]]:
    """
    Calculate the displacements between the original and transformed compartments.

    Parameters
    ----------
    orig_morph : Morphology
        The original morphology before transformation.
    trans_morph : Morphology
        The transformed morphology.

    Returns
    -------
    list of tuples
        Each tuple contains the original compartment, transformed compartment, and the displacement magnitude.
    """
    displacements = []
    for oc, tc in zip(
        orig_morph.compartment_list, trans_morph.compartment_list
    ):
        dist = np.linalg.norm(
            np.array([oc["x"], oc["y"], oc["z"]])
            - np.array([tc["x"], tc["y"], tc["z"]])
        )
        displacements.append((Compartment(**oc), Compartment(**tc), dist))
    return displacements


def get_color_map(
    displacements: Dict[str, List[Tuple[Compartment, Compartment, float]]],
    colormap: str = "coolwarm",
) -> Tuple[plt.cm.ScalarMappable, float, float]:
    """
    Generate a colormap based on the displacements.

    Parameters
    ----------
    displacements : dict
        A dictionary containing displacements for each SWC file.
    colormap : str
        The name of the colormap to use. Default is "coolwarm".

    Returns
    -------
    tuple
        A tuple containing the colormap, minimum displacement, and maximum displacement.
    """
    min_disp = min(
        displacement[2]
        for displacements in displacements.values()
        for displacement in displacements
    )
    max_disp = max(
        displacement[2]
        for displacements in displacements.values()
        for displacement in displacements
    )
    if min_disp == max_disp:
        logging.warning(
            "Minimum and maximum displacements are equal, so colormap will be all one color."
        )
    cmap = plt.get_cmap(colormap)
    return cmap, min_disp, max_disp


def save_displacement_swcs(
    displacements: Dict[str, List[Tuple[Compartment, Compartment, float]]],
    output_folder: str,
    cmap: plt.cm.ScalarMappable,
    min_disp: float,
    max_disp: float,
) -> None:
    """
    Save SWC files that represent the displacement of compartments with color mapping.

    Parameters
    ----------
    displacements : dict
        A dictionary containing displacements for each SWC file.
    output_folder : str
        The directory where the displacement SWCs will be saved.
    cmap : matplotlib.colors.Colormap
        The colormap used for coloring the displacements.
    min_disp : float
        The minimum displacement value found across all SWCs.
    max_disp : float
        The maximum displacement value found across all SWCs.
    """
    os.makedirs(output_folder, exist_ok=True)
    for swc_file, displacements in displacements.items():
        for disp_idx, (oc, tc, dist) in enumerate(displacements):
            color = (
                cmap((dist - min_disp) / (max_disp - min_disp))[:3]
                if min_disp != max_disp
                else (1, 0, 0)
            )
            disp_swc_path = os.path.join(
                output_folder, f"{swc_file[:-4]}_disp_{disp_idx}.swc"
            )
            with open(disp_swc_path, "w") as f:
                f.write(f"# COLOR {color[0]} {color[1]} {color[2]}\n")
                f.write(f"1 2 {oc['x']} {oc['y']} {oc['z']} 1.0 -1\n")
                f.write(f"2 2 {tc['x']} {tc['y']} {tc['z']} 1.0 1\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create displacement SWCs with color mapping."
    )
    parser.add_argument(
        "--original_folder",
        help="Path to the folder containing the original SWC files.",
    )
    parser.add_argument(
        "--transformed_folder",
        help="Path to the folder containing the transformed SWC files.",
    )
    parser.add_argument(
        "--output_folder",
        help="Path to the folder where the displacement SWCs will be saved.",
    )
    args = parser.parse_args()

    displacement_info: Dict[
        str, List[Tuple[Compartment, Compartment, float]]
    ] = {}
    for swc_file in os.listdir(args.original_folder):
        if swc_file.endswith(".swc"):
            original_swc_path = os.path.join(args.original_folder, swc_file)
            transformed_swc_path = os.path.join(
                args.transformed_folder, swc_file
            )
            original_morph = read_swc(original_swc_path)
            transformed_morph = read_swc(transformed_swc_path)
            displacement_info[swc_file] = calculate_displacements(
                original_morph, transformed_morph
            )

    cmap, min_disp, max_disp = get_color_map(displacement_info)
    save_displacement_swcs(
        displacement_info, args.output_folder, cmap, min_disp, max_disp
    )
