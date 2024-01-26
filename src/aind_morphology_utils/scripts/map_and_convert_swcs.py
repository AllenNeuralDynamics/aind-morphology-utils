import argparse
import os
import logging
from pathlib import Path

from allensdk.core.swc import read_swc

from aind_morphology_utils.ccf_annotation import CCFMorphologyMapper
from aind_morphology_utils.writers import MouseLightJsonWriter


def map_and_convert_swc_dir(
    swc_dir: str, out_json_dir: str, resolution: int = 25
) -> None:
    """
    Function to map CCF annotations to and convert .swc files into MouseLight .json format.

    Parameters
    ----------
    swc_dir : str
        Directory where .swc files are located.
    out_json_dir : str
        Output directory where .json files will be written.
    resolution : int, optional
        Resolution in microns for the CCFMorphologyMapper, by default 25.

    Returns
    -------
    None
    """
    for file in Path(swc_dir).iterdir():
        if file.suffix == ".swc":
            logging.info(f"Processing {file.name}")
            morphology = read_swc(str(file))

            mapper = CCFMorphologyMapper(resolution=resolution)  # microns
            mapper.annotate_morphology(morphology)

            writer = MouseLightJsonWriter(morphology, id_string=file.stem)
            json_output_file = os.path.join(out_json_dir, f"{file.stem}.json")
            writer.write(json_output_file)


def main() -> None:
    """
    Main function which handles the command line arguments and calls the conversion function.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Directory of .swc files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output directory for .json files",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=25,
        help="CCF resolution in microns",
    )
    parser.add_argument(
        "-l", "--log-level", type=str, default="INFO", help="Logging level"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.makedirs(args.output, exist_ok=True)

    map_and_convert_swc_dir(args.input, args.output, args.resolution)


if __name__ == "__main__":
    main()
