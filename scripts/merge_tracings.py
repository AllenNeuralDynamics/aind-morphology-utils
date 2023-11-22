import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple
import networkx as nx
from aind_morphology_utils.swc import NeuronGraph


def group_neuron_files(
    file_names: List[str],
) -> Dict[Tuple[str, str], List[str]]:
    """
    Group neuron files by neuron ID and tracer initials.

    Parameters
    ----------
    file_names : List[str]
        List of absolute file paths.

    Returns
    -------
    Dict[Tuple[str, sample, str], List[str]]
        A dictionary where the key is a tuple (neuron ID, sample, tracer initials) and the value is a list of file paths.
    """
    grouped_files = defaultdict(list)
    for file_path in file_names:
        file_name = os.path.basename(file_path)
        parts = file_name.split("-")
        neuron_id, sample, tracer_initials = (
            parts[0],
            parts[1],
            parts[-1].split(".")[0],
        )
        group_key = (neuron_id, sample, tracer_initials)
        grouped_files[group_key].append(file_path)
    return grouped_files


def collect_swcs(directory: str, ignore_list: List[str]) -> List[str]:
    """
    Walk through a directory and collect SWC files, ignoring files with certain substrings.

    Parameters
    ----------
    directory : str
        The directory to search for SWC files.
    ignore_list : List[str]
        List of substrings to ignore in file names.

    Returns
    -------
    List[str]
        A list of absolute file paths for SWC files.
    """
    swc_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".swc") and not any(
                ign in file for ign in ignore_list
            ):
                swc_files.append(os.path.join(root, file))
    return swc_files


def merge_swcs(files: List[str]) -> NeuronGraph:
    """
    Merge the given graphs based on whether they represent axon or dendrite.

    Parameters
    ----------
    graphs : List[nx.DiGraph]
        List of graphs to be merged.
    files : List[str]
        List of file names corresponding to the graphs.

    Returns
    -------
    nx.DiGraph
        The merged graph.

    Raises
    ------
    ValueError
        If an axon or dendrite file is not found, or if the expected number of files is not provided.
    """
    if len(files) != 2:
        raise ValueError(f"Expected 2 graphs to merge, got {len(files)}")

    dendrite_file = next(
        (f for f in files if "dendrite" in Path(f).name), None
    )
    axon_file = next((f for f in files if "axon" in Path(f).name), None)

    if not dendrite_file or not axon_file:
        raise ValueError("Could not find both axon and dendrite files")

    dendrite = NeuronGraph.from_swc(dendrite_file)
    axon = NeuronGraph.from_swc(axon_file)

    # Set structure types for dendrite and axon
    dendrite.set_constant_structure_type(3)
    axon.set_constant_structure_type(2)

    # Merge the graphs
    dendrite_root = min(
        dendrite.nodes
    )  # Assumes the root is the node with the lowest ID
    axon = nx.convert_node_labels_to_integers(
        axon, first_label=max(dendrite.nodes) + 1
    )
    axon_root = min(axon.nodes)

    # Create a union of the two graphs
    merged_graph = nx.union(dendrite, axon)

    # Connect the axon to the dendrite
    merged_graph.add_edge(dendrite_root, axon_root)

    # Set the structure type of the root node to 1
    merged_root = min(merged_graph.nodes)
    merged_graph.nodes[merged_root]["struct_type"] = 1

    return merged_graph


def main(swc_dir: str, out_dir: str, ignore_list: List[str]) -> None:
    """
    Process SWC files in the given directory by merging axon and dendrite files.

    Parameters
    ----------
    swc_dir : str
        Directory containing SWC files.
    out_dir : str
        Directory to save the merged SWC files.
    ignore_list : List[str]
        Filenames to ignore.
    """
    swc_files = collect_swcs(swc_dir, ignore_list)
    grouped_files = group_neuron_files(swc_files)

    for group_key, files in grouped_files.items():
        if len(files) > 2:
            raise ValueError(f"Expected 2 graphs to merge, got {len(files)}")
        elif len(files) == 2:
            merged_graph = merge_swcs(files)
        elif len(files) == 1:
            merged_graph = NeuronGraph.from_swc(files[0])
        else:
            raise ValueError(
                f"Expected 1 or more graphs to merge, got {len(files)}"
            )

        merged_graph.save_swc(
            os.path.join(
                out_dir, f"{group_key[0]}-{group_key[1]}-{group_key[2]}.swc"
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge neuron SWC files.")
    parser.add_argument(
        "--swc-dir",
        type=str,
        help="Directory containing SWC files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        help="Directory to save merged SWC files",
    )
    parser.add_argument(
        "--ignore-list",
        type=str,
        nargs="+",
        default=["unique", "base"],
        help="List of substrings to ignore in filenames",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    main(args.swc_dir, args.out_dir, args.ignore_list)
