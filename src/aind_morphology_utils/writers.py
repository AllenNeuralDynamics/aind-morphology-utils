import json
import logging
from pathlib import Path
from typing import Any, List, Optional

from allensdk.core import swc
from allensdk.core.swc import Morphology, read_swc, Compartment

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
    def write(
        morphology: Morphology,
        output_path: str,
        id_string: Optional[str] = None,
        sample: Optional[dict] = None,
        label: Optional[dict] = None,
        comment: Optional[str] = None,
    ) -> None:
        """
        Write the Morphology object data to a JSON file.

        Parameters
        ----------
        morphology : Morphology
           The Morphology object to write.
        output_path : str
           The path of the output JSON file.
        """
        data = MouseLightJsonWriter._build_dict(
            morphology, output_path, id_string, sample, label, comment
        )
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def _get_soma_node(morphology: Morphology):
        """
        Get the soma node from a given morphology.

        This function returns the soma node if it exists in the morphology.
        If the soma is not specified, it defaults to returning the first node (node 0).

        Parameters
        ----------
        morphology : Morphology
            The Morphology object from which to extract the soma node.

        Returns
        -------
        Compartment
            The soma node if it exists, else the first node in the morphology.
        """
        if morphology.soma is not None:
            return morphology.soma
        else:
            # FIXME: this might not always make sense
            return morphology.node(0)

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
        soma_node = MouseLightJsonWriter._get_soma_node(morphology)
        ccf_id = get_ccf_id(soma_node)
        d["soma"] = {
            "x": soma_node["x"],
            "y": soma_node["y"],
            "z": soma_node["z"],
            "allenId": ccf_id,
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
        if structure_name not in d:
            d[structure_name] = []
        if structure_type == -1:
            compartments = morphology.compartment_list
        else:
            compartments = morphology.compartment_list_by_type(structure_type)
        # Root node is the first sample for each structure type
        compartments.insert(0, MouseLightJsonWriter._get_soma_node(morphology))
        for c in MouseLightJsonWriter.remap_ids(compartments):
            ccf_region_id = get_ccf_id(c)
            if c[swc.NODE_TYPE] == Morphology.SOMA:
                stype = Morphology.SOMA
            elif structure_type == -1:
                stype = Morphology.AXON
            else:
                stype = structure_type
            sample = {
                "sampleNumber": c[swc.NODE_ID],  # enforce starting at 1
                "structureIdentifier": stype,
                "x": c[swc.NODE_X],
                "y": c[swc.NODE_Y],
                "z": c[swc.NODE_Z],
                "radius": c[swc.NODE_R],
                "parentNumber": c[swc.NODE_PN],
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
                "name": "Whole Brain"
                if structure["name"] == "root"
                else structure["name"],
                "safeName": structure["name"],
                "acronym": "wholebrain"
                if structure["acronym"] == "root"
                else structure["acronym"],
                "graphOrder": structure["graph_order"],
                "structureIdPath": "/"
                + "/".join(str(s) for s in structure["structure_id_path"])
                + "/",  # add beginning and trailing slash to match MouseLight JSON files
                "colorHex": rgb_to_hex(tuple(structure["rgb_triplet"]))
                .upper()
                .lstrip("#"),
            }
            for structure in sorted(unique_structures, key=lambda x: x["id"])
        ]

    @staticmethod
    def _find_unique_structures(morphology: Morphology) -> list:
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
    def remap_ids(node_list: List[Compartment]) -> List[Compartment]:
        """
        Remap 'id' and 'parent' fields in a list of Compartment objects for contiguous and ascending ids.

        This function returns a new list of Compartment objects where the 'id' field starts from 1
        and is contiguous and ascending. The 'parent' field is updated to reflect the new 'id' for
        the changed nodes.

        Parameters
        ----------
        node_list : List[Compartment]
            A list of Compartment objects.

        Returns
        -------
        new_node_list : List[Compartment]
            A new list of Compartment objects with 'id' and 'parent' fields remapped.
        """
        id_mapping = {
            node["id"]: new_id
            for new_id, node in enumerate(node_list, start=1)
        }

        new_node_list = [
            Compartment(
                **{
                    "id": id_mapping[node["id"]],
                    "parent": id_mapping[node["parent"]]
                    if node["parent"] in id_mapping
                    else -1,
                    **{
                        k: v
                        for k, v in node.items()
                        if k not in ["id", "parent"]
                    },
                }
            )
            for node in node_list
        ]

        return new_node_list

    @staticmethod
    def _build_dict(
        morphology: Morphology,
        output_path: str,
        id_str: Optional[str] = None,
        sample: Optional[dict] = None,
        label: Optional[dict] = None,
        comment: Optional[str] = None,
    ) -> dict:
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
        data["comment"] = comment or ""
        data["neurons"] = []
        for i in range(morphology.num_trees):
            tree = Morphology(morphology.tree(i))

            # Top-level metadata
            # FIXME: missing/hardcoded fields
            neuron_dict = {
                "idString": id_str or Path(output_path).stem + f"-{i}",
                "DOI": "n/a",
                "sample": sample,
                "label": label,
                "annotationSpace": {
                    "version": 3,
                    "description": "Annotation Space: CCFv3.0 Axes> X: Anterior-Posterior; Y: Inferior-Superior; "
                    "Z:Left-Right",
                },
            }

            MouseLightJsonWriter._populate_soma(neuron_dict, tree)

            relevant_types = {
                Morphology.AXON,
                Morphology.DENDRITE,
                Morphology.BASAL_DENDRITE,
                Morphology.APICAL_DENDRITE,
            }
            types_in_tree = get_structure_types(tree)
            relevant_types_in_tree = [
                t for t in types_in_tree if t in relevant_types
            ]
            # FIXME: this is brittle. Will not handle cases where a subset of nodes have a defined type
            for t in relevant_types_in_tree:
                MouseLightJsonWriter._populate_samples(neuron_dict, tree, t)
            if not relevant_types_in_tree:
                MouseLightJsonWriter._populate_samples(neuron_dict, tree)

            # populate CCF information
            ccf_region_set = MouseLightJsonWriter._find_unique_structures(tree)
            MouseLightJsonWriter._populate_allen_info(
                neuron_dict, ccf_region_set
            )

            data["neurons"].append(neuron_dict)

        return data


def main():
    mapper = CCFMorphologyMapper(resolution=25)
    swc = r"C:\Users\cameron.arshadi\Downloads\exaSPIM_651324_Neuron_7.swc"
    out_json = (
        r"C:\Users\cameron.arshadi\Downloads\exaSPIM_651324_Neuron_7.json"
    )
    morph = read_swc(swc)
    mapper.annotate_morphology(morph)
    MouseLightJsonWriter.write(morph, out_json)


if __name__ == "__main__":
    main()
