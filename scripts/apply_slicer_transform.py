from aind_morphology_utils.coordinate_mapping import OMEZarrTransform


def transform_swc_files(
    transform_file: str, swcs_folder: str, output_folder: str
) -> None:
    """
    Apply the transformation defined in the OME-Zarr file to all SWC files
    in the specified folder and save the transformed SWC files in the output folder.

    Parameters
    ----------
    transform_file : str
        The path to the OME-Zarr file containing the transform.
    swcs_folder : str
        The directory containing the SWC files to be transformed.
    output_folder : str
        The directory where the transformed SWC files should be saved.

    Returns
    -------
    None
    """
    transformer = OMEZarrTransform(transform_file)
    transformer.transform_swc_files(swcs_folder, output_folder)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transform SWC files using a given OME-Zarr transformation file."
    )
    parser.add_argument(
        "--transform_file",
        help="Path to the OME-Zarr file containing the transformation.",
    )
    parser.add_argument(
        "--swcs_folder",
        help="Path to the folder containing the SWC files to transform.",
    )
    parser.add_argument(
        "--output_folder",
        help="Path to the folder where the transformed SWC files will be saved.",
    )
    args = parser.parse_args()

    transform_swc_files(
        args.transform_file, args.swcs_folder, args.output_folder
    )
