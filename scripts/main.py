import logging
import tempfile
from glob import glob
import os
import argparse
from pathlib import Path

from aind_morphology_utils import coordinate_mapping
from aind_morphology_utils.ccf_annotation import CCFMorphologyMapper
from aind_morphology_utils.coordinate_mapping import HDF5Transform
from aind_morphology_utils.utils import read_swc
from aind_morphology_utils.writers import MouseLightJsonWriter
from merge_tracings import merge_swcs_in_folder


logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
_LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--reg-dir', type=str,
                        default=r"C:\Users\cameron.arshadi\Documents\registrations\651324\registration")
    parser.add_argument('--slicer-transform', type=str, default=r"C:\Users\cameron.arshadi\Documents\registrations\651324\registration\Transform.651324.h5")
    parser.add_argument('--swc-dir', type=str,
                        default=r'C:\Users\cameron.arshadi\Documents\registrations\651324\exaSPIM_651324_2023-03-06_15-13-25\Complete')
    parser.add_argument('--image-path', type=str,
                        default="s3://aind-open-data/exaSPIM_651324_2023-03-06_15-13-25_fusion_2023-03-28_17-06-00/fused.zarr")
    parser.add_argument('--output-dir', type=str, default=r'..\results\651324-ccf_coords')
    parser.add_argument('--output-json', default=True, action='store_true')
    parser.add_argument('--transform-res', type=float, nargs='+', default=[25.0, 25.0, 25.0])
    parser.add_argument('--input-res', type=float, nargs='+', default=[0.421875, 0.421875, 0.5625])
    parser.add_argument('--swc-scale', type=float, nargs='+', default=[0.748, 0.748, 1])
    parser.add_argument('--flip-axes', type=int, nargs='+', default=[0])
    parser.add_argument('--input-scale', type=int, default=5)
    parser.add_argument('--affine-only', default=True, action='store_true')
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

    ants_registration_folder = args.reg_dir
    _LOGGER.info(f"registration folder: {ants_registration_folder}")

    slicer_transform_file = args.slicer_transform
    _LOGGER.info(f"slicer transform file: {slicer_transform_file}")

    image_path = args.image_path
    _LOGGER.info(f"image path: {image_path}")

    output_folder = args.output_dir
    _LOGGER.info(f"output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    ants_transform = None
    if ants_registration_folder is not None:
        ants_transform = coordinate_mapping.AntsTransform(
            registration_folder=ants_registration_folder,
            image_path=image_path,
            transform_res=args.transform_res,
            input_res=args.input_res,
            swc_scale=args.swc_scale,
            flip_axes=args.flip_axes,
            input_scale=args.input_scale,
            affine_only=args.affine_only
        )

    slicer_transform = None
    if slicer_transform_file is not None:
        slicer_transform = HDF5Transform(slicer_transform_file)

    with tempfile.TemporaryDirectory() as temp_dir:
        merge_swcs_in_folder(neuron_folder, temp_dir, ignore_list=['unique', 'base'])

        all_swcs = glob(os.path.join(temp_dir, '**', '*.swc'), recursive=True)

        for swc_file in all_swcs:
            _LOGGER.info(f"processing {swc_file}")

            transformed = read_swc(swc_file, add_offset=True)

            if ants_transform is not None:
                transformed = ants_transform.transform(transformed)
            if slicer_transform is not None:
                transformed = slicer_transform.transform(transformed)

            if args.output_json:
                _LOGGER.info(f"writing {swc_file} as JSON")
                # Map CCF regions and save as JSON
                ccf_mapper = CCFMorphologyMapper(resolution=args.transform_res[0])
                ccf_mapper.annotate_morphology(transformed)
                writer = MouseLightJsonWriter(
                    transformed,
                    id_string=Path(swc_file).stem,
                )
                writer.write(os.path.join(args.output_dir, Path(swc_file).stem + '.json'))

            else:
                # Save as SWC
                transformed.save(os.path.join(args.output_dir, Path(swc_file).name))


if __name__ == "__main__":
    main()
