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
        A NumPy array of shape (N, 3) where each row contains [x, y, z].
    """
    points = []
    for n in sorted(graph.nodes()):
        node = graph.nodes()[n]
        points.append([node['x'], node['y'], node['z']])
    return np.array(points)


def cluster_points(points: np.ndarray, eps: float, min_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clusters points using DBSCAN and returns representative points for each cluster
    along with their indices in the original `points` array.

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
    Tuple[np.ndarray, np.ndarray]
        The first element (np.ndarray) is an array of shape (K, 3), where K is the number
        of clusters (including the noise cluster if `-1` is present). Each row represents
        the chosen representative point for that cluster.  
        The second element (np.ndarray) is a 1D array of length K containing the indices
        of these representative points in the original `points` array.
    """
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_

    # Identify each cluster by its ID (including -1 for noise, if any)
    unique_clusters = set(labels)

    representative_points = []
    representative_indices = []

    for cluster_id in unique_clusters:
        # Find all indices in this cluster
        cluster_idxs = np.where(labels == cluster_id)[0]  # Indices in the original 'points' array
        cluster_pts = points[cluster_idxs]

        if len(cluster_pts) == 1:
            # Single point in this cluster
            chosen_point = cluster_pts[0]
            chosen_index = cluster_idxs[0]
        else:
            # Multiple points: find the centroid and choose the closest point
            centroid = np.mean(cluster_pts, axis=0)
            diffs = cluster_pts - centroid
            dists = np.linalg.norm(diffs, axis=1)
            best_idx = np.argmin(dists)
            chosen_point = cluster_pts[best_idx]
            chosen_index = cluster_idxs[best_idx]

        representative_points.append(chosen_point)
        representative_indices.append(chosen_index)

    # Convert to NumPy arrays for convenient downstream usage
    representative_points = np.array(representative_points)
    representative_indices = np.array(representative_indices, dtype=int)

    return representative_points, representative_indices



def find_crossovers(
    g: NeuronGraph,
    kdt: KDTree,
    radius: float,
    cluster_dist: float,
    min_samples: int,
    all_point_labels: np.ndarray,
    current_graph_label: int,
    visited: set
) -> List[List[float]]:
    """
    Finds crossover points between a source neuron graph and other neurons represented by a KDTree,
    excluding points from graphs that have already been visited or from the same graph.

    This version picks only the nearest valid neighbor from each source point's neighborhood.

    Parameters
    ----------
    g : NeuronGraph
        The source neuron graph.
    kdt : KDTree
        A KDTree built from all the graphs' points combined.
    radius : float
        The spatial radius to use when querying nearby points from `kdt`.
    cluster_dist : float
        The epsilon parameter for the DBSCAN clustering.
    min_samples : int
        The minimum number of samples in a neighborhood for DBSCAN.
    all_point_labels : np.ndarray
        An array parallel to kdt.data, indicating the graph index for each point.
    current_graph_label : int
        The label/index of the current graph being processed.
    visited : set
        A set of graph labels that have been processed.

    Returns
    -------
    list of list of float
        A list of chosen crossover coordinates. Each coordinate is [x, y, z].
    """
    source_points = graph_to_points(g)

    neighborhoods = kdt.query_ball_point(source_points, r=radius, return_sorted=True)

    if len(neighborhoods) == 0:
        return []

    valid_indices = []
    valid_sources = []

    # For each source point's neighborhood, pick the nearest valid neighbor
    # 'valid' means it's not from a visited graph or the same graph
    for i, nhood in enumerate(neighborhoods):
        chosen_idx = None
        for idx in nhood:
            graph_label = all_point_labels[idx]
            if graph_label not in visited and graph_label != current_graph_label:
                chosen_idx = idx
                break
        if chosen_idx is not None:
            valid_indices.append(chosen_idx)
            valid_sources.append(i)

    if len(valid_indices) == 0:
        return []

    target_coords = kdt.data[valid_indices]
    source_coords = source_points[valid_sources]

    # Cluster the chosen valid neighbors and find representatives
    reps, reps_inds = cluster_points(target_coords, eps=cluster_dist, min_samples=min_samples)
    source_reps = source_coords[reps_inds]

    return [(p[0].tolist(), p[1].tolist()) for p in zip(source_reps, reps)]


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
        help="Directory containing SWC files."
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=5.0,
        help="Radius for querying neighbors in the KDTree."
    )
    parser.add_argument(
        "--cluster_dist",
        type=float,
        default=20.0,
        help="Epsilon parameter for DBSCAN clustering."
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=5,
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

    # Build a single KDTree from all graphs and keep track of their labels
    all_points = []
    all_point_labels = []
    for i, g in all_graphs.items():
        pts = graph_to_points(g)
        all_points.append(pts)
        # Create a label array for these points
        all_point_labels.append(np.full(len(pts), i, dtype=int))

    if len(all_points) == 0:
        print("No graphs loaded.")
        return

    all_points = np.vstack(all_points)
    all_point_labels = np.concatenate(all_point_labels)

    kdt = KDTree(all_points)

    all_crossovers: List[List[float]] = []
    visited = set()

    for graph_label in all_keys:
        g = all_graphs[graph_label]

        visited.add(graph_label)

        cr = find_crossovers(
            g,
            kdt,
            radius,
            cluster_dist=cluster_dist,
            min_samples=min_samples,
            all_point_labels=all_point_labels,
            current_graph_label=graph_label,
            visited=visited
        )
        all_crossovers.extend(cr)

    # Each line: "x y z"
    with open(output_file, "w") as f:
        for coord in all_crossovers:
            f.write(f"{coord[0]} {coord[1]}\n")

    t1 = time.time()
    print("Total time:", t1 - t0)


if __name__ == "__main__":
    main()
