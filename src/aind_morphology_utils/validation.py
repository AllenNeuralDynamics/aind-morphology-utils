import argparse
import math
import os
import re
from collections import defaultdict
from typing import List, Dict

import imageio
import s3fs
import zarr
from skimage.exposure import rescale_intensity

from aind_morphology_utils.swc import NeuronGraph
from aind_morphology_utils.utils import collect_swcs


def has_one_root(graph: NeuronGraph) -> bool:
    """
    Validate that the graph has exactly one root node.

    Parameters
    ----------
    graph : NeuronGraph
        The neuron graph to validate.

    Returns
    -------
    bool
        True if the graph has exactly one root node, False otherwise.
    """
    roots = [node for node in graph.nodes if graph.in_degree(node) == 0]
    return len(roots) == 1


def has_valid_name(swc_file: str) -> bool:
    """
    Validate that the SWC file name is in the correct format.

    Parameters
    ----------
    swc_file : str
        The name of the SWC file to validate.

    Returns
    -------
    bool
        True if the SWC file name is valid, False otherwise.
    """
    pattern = r"^N\d{1,}(-|_)\d{6}(-|_)(axon|dendrite|dendrites)(-|_)([A-Za-z]{2,3}|consensus)\.swc$"
    return re.match(pattern, swc_file, re.IGNORECASE) is not None


def validate_swcs(swc_paths: List[str], image_path: str, output_dir: str) -> Dict[str, Dict[str, str]]:
    """
    Validate the SWC files in the given list.

    Parameters
    ----------
    swc_paths : List[str]
        List of paths to SWC files to be validated.
    image_path : str
        The path to the Zarr image file.
    output_dir : str
        The directory to save output files.

    Returns
    -------
    dict
        A dictionary containing validation results for each SWC file.
    """
    d = defaultdict(lambda: defaultdict(dict))
    for swc_path in swc_paths:
        graph = NeuronGraph.from_swc(swc_path)
        neuron_name = graph.get_name()
        d[neuron_name]["has_one_root"] = has_one_root(graph)
        d[neuron_name]["has_valid_name"] = has_valid_name(os.path.basename(swc_path))
        d[neuron_name]["soma_mip"] = get_soma_mip(image_path, graph, output_dir)
    return d


def get_soma_mip(
        image_path: str,
        graph: NeuronGraph,
        output_dir: str,
        crop_size: int = 128,
        mip_depth: int = 10
) -> str:
    """
    Get a MIP of the soma coordinate from the Zarr image.

    Parameters
    ----------
    image_path : str
        The path to the Zarr image file.
    graph : NeuronGraph
        The neuron graph.
    output_dir : str
        The directory to save output files.
    crop_size : int, optional
        The size of the crop around the soma coordinate (default is 128).
    mip_depth : int, optional
        The number of slices on each side for the MIP

    Returns
    -------
    str
        The path to the saved MIP image.
    """
    # Load the image
    s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name='us-west-2'))
    store = s3fs.S3Map(root=image_path, s3=s3, check=False)
    z = zarr.open(store, mode='r')
    arr = z['0']

    # get OME-Zarr scale metadata from root group
    metadata = z.attrs.asdict()
    scale = metadata['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']

    root = [node for node in graph.nodes if graph.in_degree(node) == 0][0]
    soma = graph.nodes[root]
    z = int(soma["z"] / scale[2])
    y = int(soma["y"] / scale[3])
    x = int(soma["x"] / scale[4])

    s = math.ceil(crop_size / 2)

    mip = arr[0, 0, z - mip_depth:z + mip_depth, y - s:y + s, x - s:x + s].max(axis=0)

    mip = rescale_intensity(mip, out_range=(0, 255)).astype('uint8')

    mip_dir = os.path.join(output_dir, "mip")
    os.makedirs(mip_dir, exist_ok=True)

    mip_path = os.path.join(mip_dir, f"{graph.get_name()}_soma_mip.png")
    imageio.imwrite(mip_path, mip)

    return mip_path


def create_html_report(data: Dict[str, Dict[str, str]], output_dir: str) -> None:
    """
    Create an HTML report from the validation data.

    Parameters
    ----------
    data : dict
        A dictionary containing validation results.
    output_dir : str
        The directory to save the report.
    """
    os.makedirs(output_dir, exist_ok=True)

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SWC Validation Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                font-size: 18px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                font-size: 18px;
            }
            table, th, td {
                border: 1px solid black;
            }
            th, td {
                padding: 5px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            img {
                max-width: 128px;
                height: auto;
            }
            .true {
                color: green;
            }
            .false {
                color: red;
            }
        </style>
    </head>
    <body>
        <h1>SWC Validation Report</h1>
        <table>
            <tr>
                <th>Neuron</th>
                <th>Has One Root</th>
                <th>Has Valid Name</th>
                <th>Soma MIP</th>
            </tr>
    """

    for neuron, results in data.items():
        soma_mip_path = results["soma_mip"]

        has_one_root_class = "true" if results["has_one_root"] else "false"
        has_valid_name_class = "true" if results["has_valid_name"] else "false"

        # Append the data row to the HTML content
        html_content += f"""
        <tr>
            <td><b>{neuron}</b></td>
            <td class="{has_one_root_class}"><b>{results["has_one_root"]}</b></td>
            <td class="{has_valid_name_class}"><b>{results["has_valid_name"]}</b></td>
            <td><img src="{soma_mip_path}" alt="Soma MIP"></td>
        </tr>
        """

    html_content += """
        </table>
    </body>
    </html>
    """

    report_path = os.path.join(output_dir, 'report.html')
    with open(report_path, 'w') as f:
        f.write(html_content)

    print(f"Report saved to {report_path}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Validate SWC files.")
    parser.add_argument("--directory", type=str, help="The directory containing SWC files.")
    parser.add_argument("--output", type=str, help="The output directory")
    parser.add_argument("--zarr", type=str, help="The path to the Zarr image file.")
    return parser.parse_args()


def main() -> None:
    """
    Main function to validate SWC files and generate a report.
    """
    args = parse_args()
    path = args.directory
    output_path = args.output
    swc_paths = collect_swcs(path)
    d = validate_swcs(swc_paths, args.zarr, output_path)
    create_html_report(d, output_path)


if __name__ == "__main__":
    main()
