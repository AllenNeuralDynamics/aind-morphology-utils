import logging
from enum import Enum
from typing import Dict

import networkx as nx
from allensdk.core.swc import Morphology

_LOGGER = logging.getLogger(__name__)


class StructureTypes(Enum):
    """
    An enumeration of structure types.
    """
    UNDEFINED = 0
    SOMA = 1
    AXON = 2
    BASAL_DENDRITE = 3
    APICAL_DENDRITE = 4
    FORK_POINT = 5
    END_POINT = 6
    CUSTOM = 7


class NeuronGraph(nx.DiGraph):
    """
    A specialized DiGraph for representing neuron reconstructions.
    """

    def __init__(self):
        super().__init__()
        self.offset = [0.0, 0.0, 0.0]  # X Y Z

    
    def from_allensdk_morphology(self, morphology: Morphology)-> "NeuronGraph":
        """
        Load an an allensdk morphology object into a NeuronGraph.
        """
        #graph = cls()  # Instantiate a new NeuronGraph object

        print("Now starting this....")
        for compartment in morphology.compartment_list:
            node_id = compartment['id']
            struct_type = compartment['type']
            x = compartment['x']
            y = compartment['y']
            z = compartment['z']
            radius = compartment['radius']
            parent_id = compartment['parent']
            
            try:
                self.add_node(
                                node_id,
                                struct_type=struct_type,
                                x=x,
                                y=y,
                                z=z,
                                radius=radius,
                            )
            except:
                print("Could not add node!")
            
        for compartment in morphology.compartment_list:
            node_id = compartment['id']
            parent_id = compartment['parent']
            try:
                if parent_id != -1:
                    self.add_edge(parent_id, node_id)
            except:
                print("Couldn't add edge!!!")
        
        #return graph


    @classmethod
    def from_swc(cls, swc_file_path: str) -> "NeuronGraph":
        """
        Load an SWC file into a NeuronGraph.

        Parameters
        ----------
        swc_file_path : str
            Path to the SWC file to be loaded.

        Returns
        -------
        NeuronGraph
            A NeuronGraph object populated with data from the SWC file.

        Raises
        ------
        IOError
            If there is an error reading the SWC file.
        ValueError
            If there is an invalid line format in the SWC file.
        """
        graph = cls()  # Instantiate a new NeuronGraph object

        def parse_offset(line: str) -> list:
            """Parse the OFFSET line and return the offset values as a list of floats."""
            try:
                # Split the line and filter out non-numeric parts
                parts = line.split()
                offset_values = parts[2:]
                return [float(value) for value in offset_values]
            except ValueError:
                _LOGGER.error(
                    f"Error: Invalid OFFSET format in SWC file: {line}"
                )
                raise

        def parse_line(line: str, offset) -> tuple:
            """Parse a line of SWC data into node parameters, applying offset to coordinates."""
            try:
                parts = line.split()
                node_id = int(parts[0])
                struct_type = int(parts[1])
                x, y, z = [
                    float(coord) + offset_val
                    for coord, offset_val in zip(parts[2:5], offset)
                ]
                radius = float(parts[5])
                parent_id = int(parts[6])
                return node_id, struct_type, x, y, z, radius, parent_id
            except ValueError:
                _LOGGER.error(
                    f"Error: Invalid line format in SWC file: {line}"
                )
                raise

        try:
            with open(swc_file_path, "r") as file:
                for line in file:
                    if line.startswith("#"):
                        if "OFFSET" in line:
                            graph.offset = parse_offset(line)
                        continue
                    (
                        node_id,
                        struct_type,
                        x,
                        y,
                        z,
                        radius,
                        parent_id,
                    ) = parse_line(line, graph.offset)
                    graph.add_node(
                        node_id,
                        struct_type=struct_type,
                        x=x,
                        y=y,
                        z=z,
                        radius=radius,
                    )
                    if parent_id != -1:
                        graph.add_edge(parent_id, node_id)
        except IOError as e:
            _LOGGER.error(f"Error reading file {swc_file_path}: {e}")
            raise

        return graph

    def set_structure_types(self, structure_type_map: Dict[int, int]) -> None:
        """
        Set structure types for nodes in the graph based on a given mapping.

        Parameters
        ----------
        structure_type_map : Dict[int, int]
            A dictionary mapping node IDs to structure types.
        """
        for node_id, struct_type in structure_type_map.items():
            if node_id in self.nodes():
                self.nodes[node_id]["struct_type"] = struct_type
            else:
                _LOGGER.warning(
                    f"Warning: Node {node_id} not found in the graph."
                )

    def set_constant_structure_type(self, structure_type: int) -> None:
        """
        Set a constant structure type for all nodes in the graph.

        Parameters
        ----------
        structure_type : int
            The structure type to be set for all nodes.
        """
        for node in self.nodes():
            self.nodes[node]["struct_type"] = structure_type

    def get_lines(node, mydict, lines):
        if node['parent_id'] == -1:
            nodeid, x, y, z, radius, struct_type, parent_id = (
                            node[attr]
                            for attr in ["node_id", "x", "y", "z", "radius", "struct_type", "parent_id"]
                        )
        else:
            get_lines(mydict[node['parent_id']], mydict, lines)
            nodeid, x, y, z, radius, struct_type, parent_id = (
                            mydict[node['parent_id']][attr]
                            for attr in ["node_id", "x", "y", "z", "radius", "struct_type", "parent_id"]
                        )
            #print(nodeid, x, y, z, radius, struct_type, parent_id)
            lines.append(
                    f"{int(nodeid)} {int(struct_type)} {float(x)} {float(y)} {float(z)} {float(radius)} {int(parent_id)}\n"
                )
        return lines
    


    def save_swc(self, swc_path: str, sort_by_parentid = True) -> None:
        """
        Save the graph as an SWC file.

        Parameters
        ----------
        swc_path : str
            The path to the file to be written.

        Raises
        ------
        IOError
            If the file cannot be opened for writing.
        KeyError
            If required node attributes are missing.
        """
        try:
            print(self.nodes[0].keys())
            mydict = {}
            print(self.nodes[0]['node_id'])
            for i in range(len(self.nodes)):
                print(i)
                mydict[self.nodes[i]['node_id']] = self.nodes[i]
            lines = []
            newlines = get_lines(self.nodes[0],mydict, lines)
            '''#node list with parent id added
            nodelist = []
            for nodeid in self.nodes:
                node = self.nodes[nodeid]
                node['node_id'] = nodeid
                node['parent_id'] = next(iter(self.predecessors(nodeid)), -1)
                nodelist.append(node)

            if sort_by_parentid:
                nodelist.sort(key=lambda s: s['node_id'])
                print("sorted")

            #populate lines
            lines = []
            for attrs in nodelist:
                try:
                    node, x, y, z, radius, struct_type, parent_id = (
                        attrs[attr]
                        for attr in ["node_id", "x", "y", "z", "radius", "struct_type", "parent_id"]
                    )
                except KeyError as e:
                    raise KeyError(
                        f"Missing required attribute {e} for node"
                    )

                lines.append(
                    f"{int(node)} {int(struct_type)} {float(x)} {float(y)} {float(z)} {float(radius)} {int(parent_id)}\n"
                )

            with open(swc_path, "w") as file:
                file.writelines(lines)'''

            '''lines = []
            for node in sorted(self.nodes()):
                attrs = self.nodes[node]
                try:
                    x, y, z, radius, struct_type = (
                        attrs[attr]
                        for attr in ["x", "y", "z", "radius", "struct_type"]
                    )
                except KeyError as e:
                    raise KeyError(
                        f"Missing required attribute {e} for node {node}"
                    )

                parent_id = next(iter(self.predecessors(node)), -1)
                lines.append(
                    f"{int(node)} {int(struct_type)} {float(x)} {float(y)} {float(z)} {float(radius)} {int(parent_id)}\n"
                )

            with open(swc_path, "w") as file:
                file.writelines(lines)'''

        except IOError as e:
            _LOGGER.error(f"Error saving file {swc_path}: {e}")
            raise
