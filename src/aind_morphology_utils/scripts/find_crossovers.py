import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree as KDTree
from sklearn.cluster import DBSCAN

from aind_morphology_utils.swc import NeuronGraph


def graph_to_points(graph: nx.Graph) -> np.ndarray:
    """
    Extracts node coordinates from a graph into a NumPy array.

    Parameters
    ----------
    graph : nx.Graph
        A NetworkX graph (e.g., a NeuronGraph) where each node has 'x', 'y', and 'z' attributes.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (N, 3) where N is the number of nodes, and each row contains [x, y, z].
    """
    points = []
    for n in sorted(graph.nodes()):
        node = graph.nodes()[n]
        points.append([node['x'], node['y'], node['z']])
    return np.array(points)


def graphs_to_kdtree(graphs: List[NeuronGraph]) -> Optional[KDTree]:
    """
    Builds a KDTree from a collection of neuron graphs.

    Parameters
    ----------
    graphs : list of NeuronGraph
        A list of neuron graphs, each containing nodes with 3D coordinates.

    Returns
    -------
    KDTree or None
        A KDTree built from all the graph points combined, or None if no points are found.
    """
    all_points = []
    for g in graphs:
        points = graph_to_points(g)
        all_points.extend(points)
    if not all_points:
        return None
    return KDTree(np.array(all_points))


def cluster_points(points: np.ndarray, eps: float, min_samples: int) -> List[np.ndarray]:
    """
    Clusters points using DBSCAN and returns representative points for each cluster.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 3) containing points to cluster.
    eps : float
        The DBSCAN 'eps' parameter.
    min_samples : int
        The DBSCAN 'min_samples' parameter.

    Returns
    -------
    List[np.ndarray]
        A list of representative points (each a length-3 ndarray) for each cluster.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    unique_clusters = set(labels)

    representatives = []
    for cluster_id in unique_clusters:
        cluster_points = points[labels == cluster_id]

        if len(cluster_points) == 1:
            # Single point cluster, return the point itself
            chosen_point = cluster_points[0]
        else:
            # Multiple points: find centroid and choose closest point
            centroid = np.mean(cluster_points, axis=0)
            diffs = cluster_points - centroid
            dists = np.linalg.norm(diffs, axis=1)
            best_idx = np.argmin(dists)
            chosen_point = cluster_points[best_idx]

        representatives.append(chosen_point)

    return representatives


def find_crossovers(
    g: NeuronGraph,
    kdt: KDTree,
    radius: float,
    cluster_dist: float = 20.0,
    min_samples: int = 3
) -> List[List[float]]:
    """
    Finds crossover points between a source neuron graph and other neurons represented by a KDTree.

    Parameters
    ----------
    g : NeuronGraph
        The source neuron graph.
    kdt : KDTree
        A KDTree built from the other neuron graphs' points.
    radius : float
        The spatial radius to use when querying nearby points from `kdt`.
    cluster_dist : float, optional
        The epsilon parameter for the DBSCAN clustering, by default 20.0.
    min_samples : int, optional
        The minimum number of samples in a neighborhood for a point to be considered a core point
        by DBSCAN, by default 3.

    Returns
    -------
    list of list of float
        A list of chosen crossover coordinates. Each coordinate is [x, y, z].
    """
    source_points = graph_to_points(g)

    # Query nearest neighbor in all other graphs for each source point
    dd, ii = kdt.query(source_points, k=1)

    # Filter results by radius
    mask = dd <= radius
    ii = ii[mask]

    if len(ii) == 0:
        return []

    target_coords = kdt.data[ii]

    # Cluster target coordinates and extract representatives
    reps = cluster_points(target_coords, eps=cluster_dist,
                          min_samples=min_samples)
    return [r.tolist() for r in reps]


def load_graphs_from_dir(swcdir: str) -> Dict[int, NeuronGraph]:
    """
    Loads neuron graphs from a directory, skipping empty graphs.

    Parameters
    ----------
    swcdir : str
        Directory containing SWC files.

    Returns
    -------
    Dict[int, NeuronGraph]
        A dictionary mapping integer labels to loaded NeuronGraphs.
    """
    all_graphs: Dict[int, NeuronGraph] = {}
    for i, swc in enumerate(Path(swcdir).iterdir()):
        g = NeuronGraph.from_swc(swc)
        if nx.is_empty(g):
            print(f"Skipping empty graph {swc}")
            continue
        all_graphs[i] = g
    return all_graphs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find crossovers between neuron graphs.")
    parser.add_argument(
        "--swcdir",
        type=str,
        default="/root/capsule/data/exaSPIM_653980_block_000/swcs/final-trees",
        help="Directory containing SWC files."
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=5,
        help="Radius for querying neighbors in the KDTree."
    )
    parser.add_argument(
        "--cluster_dist",
        type=float,
        default=20,
        help="Epsilon parameter for DBSCAN clustering."
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=3,
        help="Minimum samples parameter for DBSCAN clustering."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="crossovers.txt",
        help="Output filename to save the crossover points."
    )

    args = parser.parse_args()

    swcdir = args.swcdir
    radius = args.radius
    cluster_dist = args.cluster_dist
    min_samples = args.min_samples
    output_file = args.output

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Load graphs
    all_graphs = load_graphs_from_dir(swcdir)
    all_keys = list(all_graphs.keys())
    all_crossovers: List[List[float]] = []

    # Process each graph, build KDTree from the rest, and find crossovers
    for graph_label in all_keys:
        g = all_graphs.pop(graph_label)
        rest_graphs = list(all_graphs.values())
        if rest_graphs:
            # FIXME: this could be sped up by building
            # one kdtree with all graphs up front, then filtering
            # queries for visted graphs.
            start_kdtree = time.time()
            kdt = graphs_to_kdtree(rest_graphs)
            end_kdtree = time.time()
            # print(f"graphs_to_kdtree took {end_kdtree - start_kdtree:.4f} seconds")
            if kdt is None:
                continue
            all_crossovers.extend(
                find_crossovers(
                    g,
                    kdt,
                    radius,
                    cluster_dist=cluster_dist,
                    min_samples=min_samples
                )
            )

    # Save all crossovers to a file
    # Each line: "x y z"
    with open(output_file, "w") as f:
        for coord in all_crossovers:
            f.write(f"{coord[0]} {coord[1]} {coord[2]}\n")

    t1 = time.time()
    print("Total time:", t1 - t0)


if __name__ == "__main__":
    main()
