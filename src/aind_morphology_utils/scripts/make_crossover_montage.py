import argparse
import math
from pathlib import Path
from typing import Tuple, List, Optional

import dask.array as da
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tensorstore as ts
import zarr
from skimage.color import label2rgb, gray2rgb
from skimage.draw import line, disk
from skimage.exposure import rescale_intensity
from skimage.measure import label
from skimage.morphology import remove_small_objects

from aind_morphology_utils.swc import NeuronGraph


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the script.

    Returns
    -------
    argparse.Namespace
        An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Script to read a set of crossover coordinates, "
            "extract subvolumes from a 3D label mask, and create MIP images. "
            "Optionally, remove small objects from the subvolumes before saving a montage."
        )
    )
    parser.add_argument(
        "--coords-path",
        type=str,
        help="Path to the text file containing the crossover coordinates."
    )
    parser.add_argument(
        "--label-mask",
        type=str,
        help="Path or URI to the label mask (Neuroglancer Precomputed)."
    )
    parser.add_argument(
        "--mip-size",
        type=int,
        default=128,
        help=(
            "Full MIP size (in both X and Y dimensions). "
            "The script will use half of this value on each side of the coordinate."
        )
    )
    parser.add_argument(
        "--mip-depth",
        type=int,
        default=64,
        help="Full MIP depth (in Z dimension). The script will use half of this value above and below the coordinate."
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=0,
        help="Minimum size for removing small labels."
    )
    parser.add_argument(
        "--output-filepath",
        type=str,
        help="Path to save the montage output image."
    )

    return parser.parse_args()


def read_coords(filename: str) -> np.ndarray:
    """
    Reads lines of coordinates from a file and returns them as a NumPy array.

    Each line in the file should have space-separated float values:
    x_value y_value z_value

    Parameters
    ----------
    filename : str
        Path to the coordinates file.

    Returns
    -------
    np.ndarray
        An array of shape (N, 3), where each row is (x, y, z).
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            # Strip whitespace
            line = line.strip()
            # Split by space and convert to float
            coords = [float(x.strip()) for x in line.split(' ')]
            data.append(coords)

    return np.array(data, dtype=float)


def subgraph_in_bounding_box(
    G: nx.Graph,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float
) -> Optional[nx.Graph]:
    """
    Extracts a subgraph of G that lies within a given 3D bounding box.

    The function also shifts node coordinates in the subgraph
    so that the bounding box corner is at (0, 0, 0).

    Parameters
    ----------
    G : nx.Graph
        The input graph. Each node is expected to have 'x', 'y', 'z' attributes.
    x_min : float
        Minimum x-coordinate of the bounding box.
    x_max : float
        Maximum x-coordinate of the bounding box.
    y_min : float
        Minimum y-coordinate of the bounding box.
    y_max : float
        Maximum y-coordinate of the bounding box.
    z_min : float
        Minimum z-coordinate of the bounding box.
    z_max : float
        Maximum z-coordinate of the bounding box.

    Returns
    -------
    nx.Graph or None
        A subgraph of G that lies within the bounding box, or None if no nodes found.
    """
    nodes_in_box = []
    for n, data in G.nodes(data=True):
        x, y, z = data['x'], data['y'], data['z']
        if (x_min <= x <= x_max) and (y_min <= y <= y_max) and (z_min <= z <= z_max):
            nodes_in_box.append(n)

    # Create an induced subgraph with just those nodes
    if not nodes_in_box:
        return None

    subG = G.subgraph(nodes_in_box).copy()
    for node in subG.nodes():
        subG.nodes()[node]['x'] -= x_min
        subG.nodes()[node]['y'] -= y_min
        subG.nodes()[node]['z'] -= z_min

    return subG


def load_graphs(swcdir: str) -> dict:
    """
    Loads SWC files from a directory into NeuronGraph objects.

    Parameters
    ----------
    swcdir : str
        Directory containing SWC files.

    Returns
    -------
    dict
        A dictionary mapping graph labels (file stem) to NeuronGraph objects.
    """
    all_graphs = {}
    for swc in Path(swcdir).iterdir():
        g = NeuronGraph.from_swc(swc)
        if nx.is_empty(g):
            print(f"Skipping empty graph {swc}")
            continue
        # The label for this entire graph
        graph_label = swc.stem
        g.label = graph_label
        all_graphs[graph_label] = g
    return all_graphs


def paint_subgraph_on_mip(
    mip: np.ndarray,
    subgraphs: List[nx.Graph],
    thickness: int = 1
) -> np.ndarray:
    """
    Paints the edges of multiple subgraphs onto a MIP image.
    Each subgraph is painted in a distinct color.
    The thickness parameter controls how thick the lines appear.

    Parameters
    ----------
    mip : np.ndarray
        A 2D grayscale image (e.g., a MIP) with shape (Y, X).
    subgraphs : list of networkx.Graph
        A list of subgraphs whose nodes have spatial coordinates ('x', 'y', 'z').
    thickness : int
        The radius of the disk drawn around each line pixel.
        thickness=1 means a single-pixel thick line. Increasing this value will make the line thicker.

    Returns
    -------
    np.ndarray
        An RGB image (uint8) with the subgraph edges painted in distinct colors.
    """
    if mip.dtype != np.uint8:
        # Scale the MIP to uint8 if not already
        mip = rescale_intensity(mip, in_range='image', out_range='uint8').astype(np.uint8)

    # Convert grayscale MIP to RGB
    mip_rgb = gray2rgb(mip)

    # Define a set of colors for different subgraphs
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128)   # Purple
    ]

    for i, subG in enumerate(subgraphs):
        color = colors[i % len(colors)]
        # Paint each edge of the subgraph
        for e in subG.edges():
            source = subG.nodes[e[0]]
            target = subG.nodes[e[1]]

            y0, x0 = int(round(source.get('y'))), int(round(source.get('x')))
            y1, x1 = int(round(target.get('y'))), int(round(target.get('x')))

            rr, cc = line(y0, x0, y1, x1)

            # For each pixel in the line, draw a disk to create thickness
            for yy, xx in zip(rr, cc):
                rr_disk, cc_disk = disk((yy, xx), radius=thickness)
                # Ensure we don't paint out-of-bounds
                valid = (
                    (rr_disk >= 0) & (rr_disk < mip_rgb.shape[0]) &
                    (cc_disk >= 0) & (cc_disk < mip_rgb.shape[1])
                )
                mip_rgb[rr_disk[valid], cc_disk[valid], :] = color

    return mip_rgb


def main() -> None:
    """
    Main function that reads crossover coordinates, loads a label mask,
    extracts subvolumes around each coordinate, optionally removes small objects,
    creates MIP images, and saves a montage of these MIPs.
    """
    args = parse_args()

    coords_path = args.coords_path
    label_mask = args.label_mask

    # Convert the user-specified size/depth into half-sizes
    half_xy = args.mip_size // 2
    half_z = args.mip_depth // 2

    min_size = args.min_size
    desired_shape = (args.mip_size,) * 2
    output_filepath = args.output_filepath

    coords = read_coords(coords_path)
    coords = coords.astype(int)

    dataset_future = ts.open({
        'driver': 'neuroglancer_precomputed',
        'kvstore': label_mask,
        # Use 100MB in-memory cache.
        'context': {
            'cache_pool': {
                'total_bytes_limit': 100_000_000
            }
        },
        'recheck_cached_data': 'open',
    })
    z = dataset_future.result().T[0, ...]
    print(f"Loaded dataset with shape {z.shape} from {label_mask}.")

    mips = []
    for c in coords:
        cz, cy, cx = np.flip(c)  # (z, y, x)

        z_start = max(cz - half_z, 0)
        z_end = min(cz + half_z, z.shape[0])
        y_start = max(cy - half_xy, 0)
        y_end = min(cy + half_xy, z.shape[1])
        x_start = max(cx - half_xy, 0)
        x_end = min(cx + half_xy, z.shape[2])

        subvol = z[z_start:z_end, y_start:y_end, x_start:x_end].read().result().astype(int)
        if min_size > 0:
            subvol = remove_small_objects(subvol, min_size=min_size)

        mip = np.max(subvol, axis=0)

        if mip.shape != desired_shape:
            # Skip if the extracted subvolume doesn't match desired shape
            continue

        print(f"Extracted MIP of shape {mip.shape} for coordinate {c}.")
        mips.append(mip)

    # We'll use tab10 colors for the labels, with black for background (label=0)
    base_cmap = plt.colormaps.get_cmap('tab10')
    main_colors = [base_cmap(i)[:3] for i in range(1, base_cmap.N)]

    num_mips = len(mips)
    cols = math.ceil(math.sqrt(num_mips))
    rows = math.ceil(num_mips / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), dpi=200,
                             subplot_kw={'xticks': [], 'yticks': []})

    # Ensure axes is a 2D array
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for i, mip in enumerate(mips):
        r = i // cols
        c = i % cols
        ax = axes[r, c]

        # Convert labeled image to RGB using label2rgb
        # bg_label=0 will use bg_color=(0,0,0)
        mip_rgb = label2rgb(mip, bg_label=0, bg_color=(0, 0, 0), colors=main_colors)
        ax.imshow(mip_rgb, interpolation='nearest')

    # Turn off unused subplots
    for j in range(num_mips, rows * cols):
        r = j // cols
        c = j % cols
        axes[r, c].axis('off')

    plt.tight_layout()
    plt.savefig(output_filepath)
    print(f"Montage saved to {output_filepath}")


if __name__ == "__main__":
    main()
