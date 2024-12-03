import argparse
import logging
import os
import shutil
from glob import glob
from pathlib import Path

from aind_morphology_utils import coordinate_mapping
from aind_morphology_utils.utils import read_obj, write_obj

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
        '--mesh-dir',
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
        '--mesh-scale', type=float, nargs='+', default=[0.748, 0.748, 1]
    )
    parser.add_argument('--flip-axes', type=int, nargs='+', default=[])
    parser.add_argument('--log-level', type=str, default=logging.INFO)
    return parser.parse_args()


def main() -> None:
    """
    Main function to process mesh files
    """
    args = _parse_args()
    _LOGGER.setLevel(args.log_level)

    _LOGGER.info(f"args: {args}")

    mesh_folder = args.mesh_dir
    _LOGGER.info(f"Mesh folder: {mesh_folder}")

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
        swc_scale=args.mesh_scale,
        flip_axes=args.flip_axes,
    )

    all_objs = glob(os.path.join(mesh_folder, '**', '*.obj'), recursive=True)

    inverted_dir = os.path.join(output_folder, 'inverse-warp-objs')
    os.makedirs(inverted_dir, exist_ok=True)

    for obj_file in all_objs:
        _LOGGER.info(f"processing {obj_file}")

        verts, normals, faces = read_obj(obj_file)

        transformed_verts = ants_transform.transform_array(verts)

        transformed_path = os.path.join(
            inverted_dir, os.path.relpath(
                obj_file, mesh_folder
            )
        )
        os.makedirs(os.path.dirname(transformed_path), exist_ok=True)
        write_obj(transformed_path, transformed_verts, normals, faces)


if __name__ == "__main__":
    main()
