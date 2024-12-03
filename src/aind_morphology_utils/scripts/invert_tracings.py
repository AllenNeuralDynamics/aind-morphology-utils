import argparse
import logging
import os
import shutil
from glob import glob
from pathlib import Path

from aind_morphology_utils import coordinate_mapping
from aind_morphology_utils.ccf_annotation import CCFMorphologyMapper
from aind_morphology_utils.converters import NRRDToOMEZarr
from aind_morphology_utils.coordinate_mapping import OMEZarrTransform
from aind_morphology_utils.scripts.merge_tracings import merge_swcs_in_folder
from aind_morphology_utils.utils import read_swc
from aind_morphology_utils.writers import MouseLightJsonWriter

logging.basicConfig(
    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S'
)
_LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--affine-path',
        type=str,
    )
    parser.add_argument(
        '--warp-path',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--swc-dir',
        type=str,
    )
    parser.add_argument(
        '--image-path',
        type=str,
    )
    parser.add_argument('--output-dir', type=str)
    parser.add_argument(
        '--transform-res', type=float, nargs='+', default=[25.0, 25.0, 25.0]
    )
    parser.add_argument(
        '--input-res',
        type=float,
        nargs='+',
        default=[0.421875, 0.421875, 0.5625]
    )
    parser.add_argument(
        '--swc-scale', type=float, nargs='+', default=[0.748, 0.748, 1]
    )
    parser.add_argument('--flip-axes', type=int, nargs='+', default=[])
    parser.add_argument('--log-level', type=str, default=logging.INFO)
    return parser.parse_args()


def main() -> None:
    """
    Main function to process swc files
    """
    args = _parse_args()
    _LOGGER.setLevel(args.log_level)

    _LOGGER.info(f"args: {args}")

    neuron_folder = args.swc_dir
    _LOGGER.info(f"Neuron folder: {neuron_folder}")

    affine_path = args.affine_path
    _LOGGER.info(f"affine path: {affine_path}")

    inverse_warp_path = args.warp_path
    _LOGGER.info(f"inverse warp path: {inverse_warp_path}")

    image_path = args.image_path
    _LOGGER.info(f"image path: {image_path}")

    output_folder = args.output_dir
    _LOGGER.info(f"output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    ants_transform = coordinate_mapping.AntsInverseTransform(
        affine_path=affine_path,
        inverse_warp_path=inverse_warp_path,
        image_path=image_path,
        transform_res=args.transform_res,
        input_res=args.input_res,
        swc_scale=args.swc_scale,
        flip_axes=args.flip_axes,
    )

    all_swcs = glob(os.path.join(neuron_folder, '**', '*.swc'), recursive=True)

    inverted_dir = os.path.join(output_folder, 'inverse-warp')
    os.makedirs(inverted_dir, exist_ok=True)

    for swc_file in all_swcs:
        _LOGGER.info(f"processing {swc_file}")

        transformed = read_swc(swc_file, add_offset=True)

        transformed = ants_transform.transform(transformed)

        transformed_path = os.path.join(
            inverted_dir, os.path.relpath(
                swc_file, neuron_folder
            )
        )
        os.makedirs(os.path.dirname(transformed_path), exist_ok=True)
        transformed.save(transformed_path)


if __name__ == "__main__":
    main()
