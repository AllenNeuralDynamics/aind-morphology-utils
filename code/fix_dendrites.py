import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

import networkx as nx
from networkx.algorithms.dag import dag_longest_path

from aind_morphology_utils.swc import NeuronGraph, StructureTypes
from aind_morphology_utils.writers import MouseLightJsonWriter
from aind_morphology_utils.ccf_annotation import CCFMorphologyMapper
from aind_morphology_utils.utils import read_swc

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def fix_structure_assignment(swc_in_path: str, swc_out_path: str) -> None:
    """
    Load an SWC file into a NeuronGraph, and then for each branch coming directly from
    the soma (i.e., the root), determine the longest branch in terms of number of nodes.
    The branch with the most nodes is set to AXON, while all other branches are marked as 
    BASAL_DENDRITE. The soma (root) remains with structure type SOMA.
    
    Parameters
    ----------
    swc_in_path : str
        Path to the input SWC file.
    swc_out_path : str
        Path to write the modified SWC file.
        
    Raises
    ------
    ValueError
        If no root (soma) is found or if the longest branch cannot be determined.
    """
    # Load the neuron graph from the SWC file.
    graph = NeuronGraph.from_swc(swc_in_path)

    # Identify the root node (soma): nodes with no incoming edges.
    roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    if not roots:
        raise ValueError(f"No root node (soma) found in the neuron graph for {swc_in_path}.")
    soma = roots[0]
    # Ensure the soma is always marked as SOMA.
    graph.nodes[soma]["struct_type"] = StructureTypes.SOMA.value

    # Get all immediate children of the soma (each represents a main branch).
    soma_children = list(graph.successors(soma))
    if not soma_children:
        graph.save_swc(swc_out_path)
        return

    def mark_subtree_iterative(start, structure_type: int) -> None:
        """
        Iteratively mark the given node and all its descendants with the given structure type.
        This avoids deep recursion issues in large graphs.
        """
        stack = [start]
        while stack:
            node = stack.pop()
            graph.nodes[node]["struct_type"] = structure_type
            stack.extend(list(graph.successors(node)))

    # Determine the longest branch from the soma based on node count.
    branch_lengths = {}
    axon_branch = None
    longest_branch_length = 0

    for child in soma_children:
        # Compute the branch subgraph: child + all nodes reachable from the child.
        branch_nodes = set(nx.descendants(graph, child))
        branch_nodes.add(child)
        branch_subgraph = graph.subgraph(branch_nodes)
        # Use dag_longest_path to get the longest path (by number of nodes).
        path = dag_longest_path(branch_subgraph)
        path_length = len(path)
        branch_lengths[child] = path_length

        if path_length > longest_branch_length:
            longest_branch_length = path_length
            axon_branch = child

    if axon_branch is None:
        raise ValueError(f"Could not determine the longest branch from the soma in {swc_in_path}.")

    # Mark the branches: the branch with the longest path (node count) is the axon;
    # all other branches are basal dendrites.
    for child in soma_children:
        if child == axon_branch:
            mark_subtree_iterative(child, StructureTypes.AXON.value)
        else:
            mark_subtree_iterative(child, StructureTypes.BASAL_DENDRITE.value)

    # Save the corrected SWC structure to the provided output file.
    graph.save_swc(swc_out_path)
    _LOGGER.info(f"Processed {swc_in_path} and saved fixed SWC to {swc_out_path}")


def process_file(in_swc: Path, out_swc: Path, out_json: Path) -> None:
    """
    Process an individual SWC file: fix the structure assignment, annotate morphology,
    and write out both the fixed SWC and a JSON representation.
    """
    try:
        fix_structure_assignment(str(in_swc), str(out_swc))
        morph = read_swc(str(out_swc))
        mapper = CCFMorphologyMapper(resolution=10)
        mapper.annotate_morphology(morph)
        writer = MouseLightJsonWriter(morph)
        writer.write(str(out_json))
        _LOGGER.info(f"Successfully processed {in_swc}")
    except Exception as ex:
        _LOGGER.exception(f"Error processing {in_swc}: {ex}")


def main():
    # Define the input and output root directories.
    input_root = Path("/root/capsule/results")

    c=0
    out_dir = Path("/results/fixed-types")
    for _ in [1]:
        aligned_dir = input_root / "aligned"
        assert aligned_dir.is_dir()
        c+=1

        output_swc_root = out_dir / "aligned"
        output_json_root = out_dir / "json"

        # Create output directories if they do not exist.
        output_swc_root.mkdir(parents=True, exist_ok=True)
        output_json_root.mkdir(parents=True, exist_ok=True)

        # Recursively find all SWC files in the input directory.
        swc_files = list(aligned_dir.rglob("*.swc"))
        if not swc_files:
            _LOGGER.error(f"No SWC files found under {input_root}")
            return

        _LOGGER.info(f"Found {len(swc_files)} SWC files to process.")

        # Prepare the list of tasks to process.
        tasks = []
        for swc_file in swc_files:
            # Determine relative path from input root and create corresponding output paths.
            relative_path = swc_file.relative_to(aligned_dir)
            out_swc = output_swc_root / relative_path
            out_json = output_json_root / relative_path.with_suffix(".json")
            # Make sure the output directories exist.
            out_swc.parent.mkdir(parents=True, exist_ok=True)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((swc_file, out_swc, out_json))

        # Use a process pool to process files concurrently.
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(process_file, in_swc, out_swc, out_json)
                for in_swc, out_swc, out_json in tasks
            ]

            # Optionally, wait for all futures to complete and log any errors.
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    _LOGGER.error(f"Error in processing: {exc}")

if __name__ == "__main__":
    main()
