import json
import logging
from pathlib import Path
from typing import Any

from allensdk.core import swc
from allensdk.core.swc import Morphology, read_swc

from aind_morphology_utils.ccf_annotation import CCFMorphologyMapper
from aind_morphology_utils.utils import (
    get_ccf_id,
    get_structure_types,
    rgb_to_hex,
)

_LOGGER = logging.getLogger(__name__)


class MouseLightJsonWriter:
    """Class for writing a MouseLight-style JSON file from a Morphology"""

    @staticmethod
    def write(morphology: Morphology, output_path: str) -> None:
        """
        Write the Morphology object data to a JSON file.

        Parameters
        ----------
        morphology : Morphology
           The Morphology object to write.
        output_path : str
           The path of the output JSON file.
        """
        data = MouseLightJsonWriter._build_dict(morphology, output_path)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def _populate_soma(d: dict, morphology: Morphology) -> None:
        """
        Populate the soma information in the given dictionary.

        Parameters
        ----------
        d : dict
            The dictionary to populate.
        morphology : Morphology
            The Morphology object containing soma information.

        """
        # FIXME: assume the root node corresponds to the soma
        #  in the future we should explicity use the soma .swc type identifier
        #  for the root node in the swc.
        soma_node = morphology.node(0)
        ccf_region_id = get_ccf_id(soma_node)
        d["soma"] = {
            "x": soma_node["x"],
            "y": soma_node["y"],
            "z": soma_node["z"],
            "allenId": ccf_region_id,
        }

    @staticmethod
    def _populate_samples(
        d: dict, morphology: Morphology, structure_type: int = -1
    ) -> None:
        """
        Populate the dictionary with the Morphology compartments

        Parameters
        ----------
        d : dict
            The dictionary to populate.
        morphology : Morphology
            The Morphology object.
        structure_type : int, optional
            The type of structure (defaults to -1, undefined structure). If -1,
            all compartments will be written as "axon" (structure ID 2).

        """
        if structure_type in (
            Morphology.DENDRITE,
            Morphology.BASAL_DENDRITE,
            Morphology.APICAL_DENDRITE,
        ):
            structure_name = "dendrite"
        else:
            # FIXME: assume it's axon
            structure_name = "axon"
        d[structure_name] = []
        if structure_type == -1:
            compartments = morphology.compartment_list
        else:
            compartments = morphology.compartment_list_by_type(structure_type)
        for c in compartments:
            ccf_region_id = get_ccf_id(c)
            sample = {
                "sampleNumber": c[swc.NODE_ID] + 1,  # enforce starting at 1
                "structureIdentifier": Morphology.AXON
                if structure_type == -1
                else structure_type,
                "x": c[swc.NODE_X],
                "y": c[swc.NODE_Y],
                "z": c[swc.NODE_Z],
                "radius": c[swc.NODE_R],
                "parentNumber": c[swc.NODE_PN] + 1
                if c[swc.NODE_PN] != -1
                else -1,
                "allenId": ccf_region_id,
            }
            d[structure_name].append(sample)

    @staticmethod
    def _populate_allen_info(d: dict, unique_structures: Any) -> None:
        """
        Populate the CCF structure information in the given dictionary.

        Parameters
        ----------
        d : dict
            The dictionary to populate.
        unique_structures : Any
            A collection of unique structure dictionaries.

        """
        d["allenInformation"] = [
            {
                "allenId": structure["id"],
                "name": structure["name"],
                "safeName": structure["name"],
                "acronym": structure["acronym"],
                "graphOrder": structure["graph_order"],
                "structureIdPath": "/".join(
                    str(s) for s in structure["structure_id_path"]
                ),
                "colorHex": rgb_to_hex(tuple(structure["rgb_triplet"])),
            }
            for structure in unique_structures
        ]

    @staticmethod
    def _find_unique_structures(morphology: Morphology) -> None:
        """
        Find the unique CCF structures in the given Morphology object.

        Parameters
        ----------
        morphology : Morphology
            The Morphology object to analyze.

        Returns
        -------
        unique_structures : list
            A list of unique structures.

        """
        ccf_region_set = set()
        unique_structures = []
        for c in morphology.compartment_list:
            try:
                ccf_id = c["allenInformation"]["id"]
                if ccf_id in ccf_region_set:
                    continue
                else:
                    ccf_region_set.add(ccf_id)
                    unique_structures.append(c["allenInformation"])
            except KeyError:
                continue
        return unique_structures

    @staticmethod
    def _build_dict(morphology: Morphology, output_path: str) -> dict:
        """
        Build the MLJson dictionary from the given Morphology object.

        Parameters
        ----------
        morphology : Morphology
            The Morphology object to analyze.
        output_path : str
            The path of the output file, used to populate the idString field.

        Returns
        -------
        data : dict
            The built dictionary of data.

        """
        data = {}
        data["comment"] = ""
        data["neurons"] = []
        for i in range(morphology.num_trees):
            neuron_dict = {}

            # Top-level metadata
            neuron_dict["idString"] = Path(output_path).stem
            neuron_dict["DOI"] = "n/a"
            neuron_dict["sample"] = {}
            neuron_dict["label"] = {}
            # FIXME: hardcoded to CCFv3
            neuron_dict["annotationSpace"] = {
                "version": 3,
                "description": "Annotation Space: CCFv3.0 Axes> X: Anterior-Posterior; Y: Inferior-Superior; "
                "Z:Left-Right",
            }

            MouseLightJsonWriter._populate_soma(neuron_dict, morphology)

            structure_types = get_structure_types(morphology)
            if Morphology.DENDRITE in structure_types:
                MouseLightJsonWriter._populate_samples(
                    neuron_dict, morphology, Morphology.DENDRITE
                )
            if Morphology.AXON in structure_types:
                MouseLightJsonWriter._populate_samples(
                    neuron_dict, morphology, Morphology.AXON
                )
            else:
                MouseLightJsonWriter._populate_samples(neuron_dict, morphology)

            # populate allen information
            ccf_region_set = MouseLightJsonWriter._find_unique_structures(
                morphology
            )
            MouseLightJsonWriter._populate_allen_info(
                neuron_dict, ccf_region_set
            )

            data["neurons"].append(neuron_dict)

        return data


def main():
    mapper = CCFMorphologyMapper(resolution=25)
    swc = (
        r"C:\Users\cameron.arshadi\Desktop\exaSPIM_609281_2022-11-03_13-49-18\ccf_coords\Neuron_02_2022-11-03"
        r".swc_ccf10.swc"
    )
    out_json = (
        r"C:\Users\cameron.arshadi\Desktop\exaSPIM_609281_2022-11-03_13-49-18\ccf_coords\Neuron_02_2022-11-03"
        r".swc_ccf10.json"
    )
    morph = read_swc(swc)
    mapper.annotate_morphology(morph)
    MouseLightJsonWriter.write(morph, out_json)


if __name__ == "__main__":
    main()
