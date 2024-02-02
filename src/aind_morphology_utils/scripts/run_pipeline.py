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
        '--reg-dir',
        type=str,
    )
    parser.add_argument(
        '--slicer-transform',
        type=str,
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
    parser.add_argument('--output-json', default=False, action='store_true')
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
    parser.add_argument('--input-scale', type=int, default=5)
    parser.add_argument('--affine-only', default=False, action='store_true')
    parser.add_argument('--log-level', type=str, default=logging.INFO)
    return parser.parse_args()


def process_slicer_transform(
    slicer_transform_path: str, output_folder: str
) -> OMEZarrTransform:
    """
    Process the slicer transform file and save it as a zarr file.

    Parameters
    ----------
    slicer_transform_path : str
        The path to the slicer transform, either Zarr or NRRD.
    output_folder : str
        The folder to save the zarr file to.

    Returns
    -------
    OMEZarrTransform
        The OMEZarrTransform object.
    """
    out_zarr = os.path.join(output_folder, 'slicer_transform.ome.zarr')
    if slicer_transform_path.endswith('.nrrd'):
        converter = NRRDToOMEZarr(slicer_transform_path)
        converter.save(out_zarr)
        return OMEZarrTransform(out_zarr)
    elif slicer_transform_path.endswith('.zarr'):
        shutil.copytree(slicer_transform_path, out_zarr)
        return OMEZarrTransform(out_zarr)
    else:
        raise ValueError(
            f"Unknown slicer transform file type: {slicer_transform_path}"
        )


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
        slicer_transform = process_slicer_transform(
            slicer_transform_file, output_folder
        )

    all_swcs = glob(os.path.join(neuron_folder, '**', '*.swc'), recursive=True)

    aligned_dir = os.path.join(output_folder, 'aligned')
    os.makedirs(aligned_dir, exist_ok=True)

    for swc_file in all_swcs:
        _LOGGER.info(f"processing {swc_file}")

        transformed = read_swc(swc_file, add_offset=True)

        if ants_transform is not None:
            transformed = ants_transform.transform(transformed)
        if slicer_transform is not None:
            transformed = slicer_transform.transform(transformed)

        transformed_path = os.path.join(
            aligned_dir, os.path.relpath(
                swc_file, neuron_folder
            )
        )
        os.makedirs(os.path.dirname(transformed_path), exist_ok=True)
        transformed.save(transformed_path)

    merged_dir = os.path.join(output_folder, 'merged')
    os.makedirs(merged_dir, exist_ok=True)
    merge_swcs_in_folder(
        aligned_dir, merged_dir, ignore_list=['unique', 'base']
    )

    if args.output_json:
        json_dir = os.path.join(output_folder, 'json')
        os.makedirs(json_dir, exist_ok=True)
        for swc_file in glob(os.path.join(merged_dir, '*.swc')):
            _LOGGER.info(f"processing {swc_file}")
            transformed = read_swc(swc_file, add_offset=True)
            # Map CCF regions and save as JSON
            ccf_mapper = CCFMorphologyMapper(resolution=args.transform_res[0])
            ccf_mapper.annotate_morphology(transformed)
            writer = MouseLightJsonWriter(
                transformed, id_string=Path(swc_file).stem, )
            writer.write(os.path.join(json_dir, Path(swc_file).stem + '.json'))


if __name__ == "__main__":
    main()
