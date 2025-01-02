import argparse
import math
import re
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
            "Script to read a set of coordinate pairs, check whether they share the same label "
            "in a 3D label mask, and create separate montages for 'same-label' vs. 'diff-label' pairs. "
            "Also saves those pairs to text files in bracketed form, optionally with scaling."
        )
    )

    parser.add_argument(
        "--coords-path",
        type=str,
        default="/root/capsule/crossovers.txt",
        help="Path to the text file containing bracketed 3D coordinate pairs."
    )
    parser.add_argument(
        "--label-mask",
        type=str,
        default=(
            "gs://allen-nd-goog/from_google/whole_brain/653980_b0/"
            "202412_73227862_855_mean80_mask40_dynamic/label_mask"
        ),
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
        default=1000,
        help="Minimum size for removing small labels in the subvolume."
    )
    parser.add_argument(
        "--scale",
        type=float,
        nargs=3,
        default=(0.748, 0.748, 1.0),
        help=(
            "Scaling factors (x_scale, y_scale, z_scale) to apply to the coordinates "
            "before saving them. Used to convert from voxel to physical units."
        )
    )
    parser.add_argument(
        "--same-output-filepath",
        type=str,
        default="/results/same_labels.png",
        help="Path to save the montage of pairs that share the same label."
    )
    parser.add_argument(
        "--diff-output-filepath",
        type=str,
        default="/results/diff_labels.png",
        help="Path to save the montage of pairs that have different labels."
    )
    parser.add_argument(
        "--same-output-coords",
        type=str,
        default="/results/same_coords.txt",
        help="Path to save the text file of source–target pairs that share the same label."
    )
    parser.add_argument(
        "--diff-output-coords",
        type=str,
        default="/results/diff_coords.txt",
        help="Path to save the text file of source–target pairs that have different labels."
    )

    return parser.parse_args()


def read_coords(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads lines of bracketed 3D coordinates from a file and returns two NumPy arrays:
    one for source coordinates and one for target coordinates.

    Each line in the file should have exactly two bracketed coordinate triplets, for example:
    [524.0, 2315.0, 878.0] [522.0, 2312.0, 879.0]

    Parameters
    ----------
    filename : str
        Path to the coordinates file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (sources, targets)
        - sources is shape (N, 3)
        - targets is shape (N, 3)
        where N is the number of lines in the file.
    """
    pattern = r"\[(.*?)\]"

    source_data = []
    target_data = []

    with open(filename, 'r') as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            matches = re.findall(pattern, line)

            if len(matches) != 2:
                raise ValueError(
                    f"Line {line_idx} does not have exactly two bracketed coordinates: '{line}'"
                )

            # Parse the first bracketed coordinate (source)
            source_coords = [float(x.strip()) for x in matches[0].split(',')]
            # Parse the second bracketed coordinate (target)
            target_coords = [float(x.strip()) for x in matches[1].split(',')]

            source_data.append(source_coords)
            target_data.append(target_coords)

    # Convert to numpy arrays
    sources = np.array(source_data, dtype=float)
    targets = np.array(target_data, dtype=float)

    return sources, targets


def create_montage(mips: List[np.ndarray], title: str, output_file: str) -> None:
    """
    Given a list of 2D label images (mips), create a montage and save as an image.

    Parameters
    ----------
    mips : List[np.ndarray]
        List of 2D label images (e.g., after taking a max projection).
    title : str
        A title to display above the figure.
    output_file : str
        Path to save the resulting montage figure.
    """
    if not mips:
        print(f"No MIPs to display for {title}. Skipping montage.")
        return

    base_cmap = plt.colormaps.get_cmap('tab10')
    main_colors = [base_cmap(i)[:3] for i in range(1, base_cmap.N)]

    num_mips = len(mips)
    cols = math.ceil(math.sqrt(num_mips))
    rows = math.ceil(num_mips / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), dpi=200,
                             subplot_kw={'xticks': [], 'yticks': []})
    fig.suptitle(title, fontsize=16)

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
        # (with black background for label=0)
        mip_rgb = label2rgb(mip, bg_label=0, bg_color=(0, 0, 0), colors=main_colors)
        ax.imshow(mip_rgb, interpolation='nearest')

    # Hide any unused subplots
    for j in range(num_mips, rows * cols):
        r = j // cols
        c = j % cols
        axes[r, c].axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)
    print(f"{title} montage saved to {output_file}")


def save_pairs_to_file(
    filename: str,
    source_coords: np.ndarray,
    target_coords: np.ndarray,
    indices: List[int],
    scale: np.ndarray
) -> None:
    """
    Save the specified source-target pairs to a file in bracketed format, optionally applying scaling.

    Parameters
    ----------
    filename : str
        Path to the output text file.
    source_coords : np.ndarray
        Array of shape (N, 3) for source coordinates.
    target_coords : np.ndarray
        Array of shape (N, 3) for target coordinates.
    indices : List[int]
        Indices of the pairs to be saved.
    scale : np.ndarray
        Scaling factors for x, y, z. E.g. [xy_scale, xy_scale, z_scale].
    """
    if not indices:
        print(f"No pairs to save for {filename}.")
        return

    with open(filename, 'w') as f:
        for i in indices:
            s = source_coords[i] * scale
            t = target_coords[i] * scale
            # Format to mirror the input style, e.g.:
            # [524.0, 2315.0, 878.0] [522.0, 2312.0, 879.0]
            f.write(f"[{s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}] [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]\n")

    print(f"Saved {len(indices)} pairs to {filename}")


def main() -> None:
    args = parse_args()

    # Unpack user inputs
    coords_path = args.coords_path
    label_mask = args.label_mask
    half_xy = args.mip_size // 2
    half_z = args.mip_depth // 2
    min_size = args.min_size
    scale = np.array(args.scale, dtype=float)

    same_output_filepath = args.same_output_filepath
    diff_output_filepath = args.diff_output_filepath

    same_output_coords = args.same_output_coords
    diff_output_coords = args.diff_output_coords

    # Read source and target coordinates
    source_coords, target_coords = read_coords(coords_path)
    source_coords = source_coords.astype(int)
    target_coords = target_coords.astype(int)

    # Open the label mask
    dataset_future = ts.open({
        'driver': 'neuroglancer_precomputed',
        'kvstore': label_mask,
        'context': {
            'cache_pool': {
                'total_bytes_limit': 100_000_000
            }
        },
        'recheck_cached_data': 'open',
    })
    z = dataset_future.result().T[0, ...]  # shape is (Z, Y, X)
    print(f"Loaded dataset with shape {z.shape} from {label_mask}.")

    # Classify each source-target pair as "same label" or "diff label"
    same_indices = []
    diff_indices = []
    for i, (src, tgt) in enumerate(zip(source_coords, target_coords)):
        # Flip from (x, y, z) to (z, y, x)
        src_z, src_y, src_x = np.flip(src)
        tgt_z, tgt_y, tgt_x = np.flip(tgt)

        # Check bounds
        if (0 <= src_z < z.shape[0] and 0 <= src_y < z.shape[1] and 0 <= src_x < z.shape[2] and
            0 <= tgt_z < z.shape[0] and 0 <= tgt_y < z.shape[1] and 0 <= tgt_x < z.shape[2]):
            label_src = z[src_z, src_y, src_x].read().result()
            label_tgt = z[tgt_z, tgt_y, tgt_x].read().result()
            if label_src == label_tgt:
                same_indices.append(i)
            else:
                diff_indices.append(i)
        else:
            print(f"Warning: Source or target out of bounds for pair {i}: src={src}, tgt={tgt}")

    def extract_mips_from_indices(idxs: List[int]) -> List[np.ndarray]:
        """
        Return a list of MIPs around the source coordinates for given indices.
        """
        mips_list = []
        for i in idxs:
            src = source_coords[i]  # (x, y, z)
            sz, sy, sx = np.flip(src)  # (z, y, x)

            z_start = max(sz - half_z, 0)
            z_end   = min(sz + half_z, z.shape[0])
            y_start = max(sy - half_xy, 0)
            y_end   = min(sy + half_xy, z.shape[1])
            x_start = max(sx - half_xy, 0)
            x_end   = min(sx + half_xy, z.shape[2])

            subvol = z[z_start:z_end, y_start:y_end, x_start:x_end].read().result().astype(int)
            if min_size > 0:
                subvol = remove_small_objects(subvol, min_size=min_size)

            # Max-intensity projection along Z
            mip = np.max(subvol, axis=0)
            if mip.shape == (args.mip_size, args.mip_size):
                mips_list.append(mip)
            else:
                print(f"Skipping pair {i} because extracted MIP shape {mip.shape} != ({args.mip_size}, {args.mip_size})")
        return mips_list

    # Extract MIPs for same-label pairs
    same_mips = extract_mips_from_indices(same_indices)
    # Extract MIPs for different-label pairs
    diff_mips = extract_mips_from_indices(diff_indices)

    # Create two montages
    create_montage(same_mips, "Same-Label Pairs", same_output_filepath)
    create_montage(diff_mips, "Different-Label Pairs", diff_output_filepath)

    # Finally, save the pairs to text files in bracketed format
    save_pairs_to_file(same_output_coords, source_coords, target_coords, same_indices, scale)
    save_pairs_to_file(diff_output_coords, source_coords, target_coords, diff_indices, scale)


if __name__ == "__main__":
    main()
