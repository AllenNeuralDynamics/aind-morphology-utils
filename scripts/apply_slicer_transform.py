from aind_morphology_utils.coordinate_mapping import HDF5Transform


def main():
    args = parse_args()

    transformer = HDF5Transform(args.transform_file)
    transformer.transform_swc_files(args.swcs, args.output_folder)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='Apply ants transform to SWC files.')
    parser.add_argument('--transform_file', type=str, required=True,
                        help='Path to the HDF5 file containing the transform.')
    parser.add_argument('--swcs', type=str, required=True,
                        help='Path to the folder containing the SWC files.')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to the folder where the transformed SWC files will be saved.')

    return parser.parse_args()


if __name__ == '__main__':
    main()
