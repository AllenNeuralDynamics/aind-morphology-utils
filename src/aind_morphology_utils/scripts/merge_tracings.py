import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import networkx as nx

from aind_morphology_utils.swc import NeuronGraph, StructureTypes


def group_neuron_files(
    file_names: List[str],
) -> Dict[Tuple[str, str, str], List[str]]:
    """
    Group neuron files by neuron ID and tracer initials,
    where the group corresponds to different compartment tracings of the same cell.
    For example, the files "N001-653158-dendrite-CA.swc" and "N001-653158-axon-CA.swc"
    would be grouped together.

    Parameters
    ----------
    file_names : List[str]
        List of absolute file paths.

    Returns
    -------
    Dict[Tuple[str, str, str], List[str]]
        A dictionary where the key is a tuple (neuron ID, sample, tracer initials) and the value is a list of file paths.
    """
    grouped_files = defaultdict(list)
    for file_path in file_names:
        file_name = os.path.basename(file_path)
        parts = file_name.lower().replace("_", "-").split("-")
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


def merge_swcs(files: List[str],
               axon_file: Optional[str] = None,
               dendrite_file: Optional[str] = None,
               merge_nearest: bool = True) -> NeuronGraph:
    """
    Merge the given graphs based on whether they represent axon or dendrite. The root 
    of the axon graph (parent == -1) will be connected to the nearest node in the dendrite graph.

    Parameters
    ----------
    files : List[str]
        List of file names corresponding to the graphs.
    axon_file : Optional[str], default=None
        File name for the axon graph. If None, it will be inferred from `files`.
    dendrite_file : Optional[str], default=None
        File name for the dendrite graph. If None, it will be inferred from `files`.
    merge_nearest : bool, default=True
        If True, the axon root will be connected to the nearest node in the dendrite graph.
        Otherwise, it will be connected to the dendrite root.

    Returns
    -------
    NeuronGraph
        The merged graph.

    Raises
    ------
    ValueError
        If an axon or dendrite file is not found, or if the expected number of files is not provided.
    """
    if len(files) != 2:
        raise ValueError(f"Expected 2 graphs to merge, got {len(files)}")

    if dendrite_file is None:
        dendrite_file = next(
            (f for f in files if "dendrite" in Path(f).name.lower()), None
        )

    if axon_file is None:
        axon_file = next((f for f in files if "axon" in Path(f).name.lower()), None)

    if not dendrite_file or not axon_file:
        raise ValueError("Could not find both axon and dendrite files")

    dendrite = NeuronGraph.from_swc(dendrite_file)
    axon = NeuronGraph.from_swc(axon_file)

    if not dendrite.nodes:
        raise ValueError(f"Dendrite graph from {dendrite_file} is empty. Cannot merge.")
    if not axon.nodes:
        raise ValueError(f"Axon graph from {axon_file} is empty. Cannot merge.")

    # Set structure types for dendrite and axon
    dendrite.set_constant_structure_type(StructureTypes.BASAL_DENDRITE.value)
    axon.set_constant_structure_type(StructureTypes.AXON.value)

    # Find the conventional dendrite root (often represents soma or connection point)
    dendrite_root = min(
        dendrite.nodes
    )  # Assumes the root is the node with the lowest ID

    # Relabel axon nodes to ensure unique IDs after merge
    first_new_axon_label = max(dendrite.nodes) + 1
    axon = nx.convert_node_labels_to_integers(
        axon, first_label=first_new_axon_label
    )
    # Find the root of the relabeled axon graph
    axon_root = min(axon.nodes) # This is the node that needs to be connected

    # Get the spatial coordinates of the axon root node
    axon_root_data = axon.nodes[axon_root]
    axon_root_coords = (axon_root_data['x'], axon_root_data['y'], axon_root_data['z'])

    if merge_nearest:
        nearest_dendrite_node = None
        min_dist_sq = float('inf')

        for dend_node, data in dendrite.nodes(data=True):
            dend_coords = (data['x'], data['y'], data['z'])
            # Calculate squared Euclidean distance (faster than sqrt).
            # There usually won't be more than a couple thousand nodes, 
            # so this is probably efficient enough
            dist_sq = (axon_root_coords[0] - dend_coords[0])**2 + \
                    (axon_root_coords[1] - dend_coords[1])**2 + \
                    (axon_root_coords[2] - dend_coords[2])**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest_dendrite_node = dend_node

        if nearest_dendrite_node is None:
            # This should not happen if the dendrite graph is not empty
            raise RuntimeError("Could not find nearest dendrite node. Dendrite graph might be malformed or empty.")
        
        merge_node = nearest_dendrite_node
    else:
        # If not merging by nearest, use the dendrite root directly
        merge_node = dendrite_root

    # Create a union of the two graphs
    merged_graph = nx.union(dendrite, axon)

    # Connect the axon root to the NEAREST node in the dendrite tracing
    merged_graph.add_edge(
        merge_node, 
        axon_root
    )

    # Set the structure type of the conventional dendrite root node to SOMA
    # This assumes the node with the minimum ID in the original dendrite file represents the soma location.
    if dendrite_root in merged_graph.nodes:
        merged_graph.nodes[dendrite_root]["struct_type"] = StructureTypes.SOMA.value
    else:
        # This case indicates an unexpected issue, dendrite_root should always be in the merged graph
        print(f"Warning: Original dendrite root {dendrite_root} not found after merge. Cannot set SOMA type.")

    return merged_graph


def merge_swcs_in_folder(swc_dir: str, out_dir: str, ignore_list: List[str]) -> None:
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
        neuron_id_str = f"{group_key[0].upper()}-{group_key[1]}-{group_key[2].upper()}"
        print(f"Processing group: {neuron_id_str} with files: {files}")
        try:
            if len(files) > 2:
                print(f"Warning: Expected 2 graphs for {neuron_id_str}, found {len(files)}. Skipping merge.")
                continue # Skip instead of raising ValueError to process other files
                # raise ValueError(f"Expected 2 graphs to merge, got {len(files)}")
            elif len(files) == 2:
                merged_graph = merge_swcs(files)
                print(f"Successfully merged {neuron_id_str}.")
            elif len(files) == 1:
                print(f"Warning: Only found one file for {neuron_id_str}. Saving as is.")
                merged_graph = NeuronGraph.from_swc(files[0])
                # Set structure type based on filename (important for single files)
                if "axon" in Path(files[0]).name.lower():
                    merged_graph.set_constant_structure_type(
                        StructureTypes.AXON.value
                    )

                elif "dendrite" in Path(files[0]).name.lower():
                    merged_graph.set_constant_structure_type(
                        StructureTypes.BASAL_DENDRITE.value
                    )
                    # If it's only dendrite, set its root type to SOMA
                    if merged_graph.nodes:
                        dendrite_root = min(merged_graph.nodes)
                        merged_graph.nodes[dendrite_root]["struct_type"] = StructureTypes.SOMA.value
                else:
                     print(f"Warning: Could not determine type (axon/dendrite) for single file {files[0]}. Structure types may be incorrect.")

            else:
                # This case means len(files) == 0, which shouldn't happen with group_neuron_files logic
                print(f"Warning: No files found for group key {group_key}. Skipping.")
                continue

            merged_graph.save_swc(
                os.path.join(
                    out_dir, f"{neuron_id_str}.swc"
                )
            )
            print(f"Saved merged file for {neuron_id_str}.")

        except Exception as e:
            print(f"Error processing group {neuron_id_str}: {e}")
            # Continue processing other files even if one group fails
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge neuron SWC files.")
    parser.add_argument(
        "--swc-dir",
        type=str,
        required=True, # Make input dir required
        help="Directory containing SWC files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True, # Make output dir required
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

    print(f"Starting SWC merge process.")
    print(f"Input directory: {args.swc_dir}")
    print(f"Output directory: {args.out_dir}")
    print(f"Ignoring files containing: {args.ignore_list}")

    os.makedirs(args.out_dir, exist_ok=True)
    merge_swcs_in_folder(args.swc_dir, args.out_dir, args.ignore_list)
    print("SWC merge process finished.")