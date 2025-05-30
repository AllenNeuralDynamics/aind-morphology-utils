import unittest
from unittest import mock

import networkx as nx
import pandas as pd
from pandas.testing import assert_frame_equal

from aind_morphology_utils.swc import NeuronGraph


class TestNeuronGraph(unittest.TestCase):
    sample_swc_data = "\n".join(
        [
            "# Sample SWC data",
            "1 1 0.0 0.0 0.0 0.5 -1",
            "2 3 1.0 0.0 0.0 0.3 1",
            "3 3 1.0 1.0 0.0 0.4 1",
            "4 2 0.0 1.0 0.0 0.6 1",
        ]
    )

    def create_graph_from_mock_data(self, swc_data=None):
        if swc_data is None:
            swc_data = self.sample_swc_data
        mock_open = mock.mock_open(read_data=swc_data)
        with mock.patch("builtins.open", mock_open):
            return NeuronGraph.from_swc("dummy_path.swc")

    def test_from_swc(self):
        """Test loading a graph from an SWC file."""
        graph = self.create_graph_from_mock_data()
        self.assertEqual(len(graph.nodes), 4)
        self.assertEqual(len(graph.edges), 3)

    def test_set_constant_structure_type(self):
        """Test setting a constant structure type for all nodes."""
        graph = self.create_graph_from_mock_data()
        graph.set_constant_structure_type(2)
        for _, node_data in graph.nodes(data=True):
            self.assertEqual(node_data["struct_type"], 2)

    def test_set_structure_types(self):
        """Test setting structure types for nodes based on a mapping."""
        graph = self.create_graph_from_mock_data()
        structure_type_map = {1: 1, 2: 2, 3: 2, 4: 3}
        graph.set_structure_types(structure_type_map)
        for node_id, node_data in graph.nodes(data=True):
            self.assertEqual(
                node_data["struct_type"], structure_type_map[node_id]
            )

    def test_connectivity(self):
        """Test the connectivity of the graph."""
        graph = self.create_graph_from_mock_data()
        expected_parents = {2: 1, 3: 1, 4: 1}
        for node, parent in expected_parents.items():
            self.assertIn(
                parent,
                graph.predecessors(node),
                f"Node {node} should have {parent} as its parent.",
            )

    def test_empty_swc_data(self):
        """Test loading an empty SWC file."""
        empty_graph = self.create_graph_from_mock_data(swc_data="")
        self.assertEqual(len(empty_graph.nodes), 0)
        self.assertEqual(len(empty_graph.edges), 0)

    def test_malformed_swc_line(self):
        """Test handling of malformed line in SWC file."""
        malformed_swc_data = (
            "# Sample SWC data\n1 1 0.0 0.0 not_a_number 0.5 -1"
        )
        with self.assertRaises(ValueError):
            self.create_graph_from_mock_data(swc_data=malformed_swc_data)

    def test_nodes_with_no_edges(self):
        """Test handling nodes with no connecting edges."""
        swc_data_with_isolated_node = (
            "# Sample SWC data\n1 1 0.0 0.0 0.0 0.5 -1\n5 3 2.0 2.0 2.0 0.3 -1"
        )
        graph = self.create_graph_from_mock_data(
            swc_data=swc_data_with_isolated_node
        )
        self.assertIn(5, graph.nodes)
        self.assertEqual(len(list(graph.neighbors(5))), 0)

    def test_node_attributes(self):
        """Test correct attributes of nodes."""
        graph = self.create_graph_from_mock_data()
        node_1_attrs = graph.nodes[1]
        self.assertEqual(node_1_attrs["struct_type"], 1)
        self.assertEqual(node_1_attrs["x"], 0.0)
        self.assertEqual(node_1_attrs["y"], 0.0)
        self.assertEqual(node_1_attrs["z"], 0.0)
        self.assertEqual(node_1_attrs["radius"], 0.5)

    def test_graph_properties(self):
        """Test properties of the graph."""
        graph = self.create_graph_from_mock_data()
        self.assertEqual(
            nx.number_connected_components(graph.to_undirected()), 1
        )

    def test_graph_modifications(self):
        """Test modifications to the graph."""
        graph = self.create_graph_from_mock_data()
        graph.add_node(5, struct_type=3, x=2.0, y=2.0, z=2.0, radius=0.4)
        graph.add_edge(1, 5)
        self.assertIn(5, graph.nodes)
        self.assertIn((1, 5), graph.edges)

    def test_save_as_swc(self):
        """Test saving a graph as an SWC file."""
        graph = self.create_graph_from_mock_data()
        mock_open = mock.mock_open()
        with mock.patch("builtins.open", mock_open):
            graph.save_swc("dummy_path.swc")
        expected_lines = [
            "1 1 0.0 0.0 0.0 0.5 -1\n",
            "2 3 1.0 0.0 0.0 0.3 1\n",
            "3 3 1.0 1.0 0.0 0.4 1\n",
            "4 2 0.0 1.0 0.0 0.6 1\n",
        ]
        mock_open().writelines.assert_called_once_with(expected_lines)

    def test_save_as_swc_missing_attributes(self):
        """Test handling missing attributes when saving as SWC."""
        graph = self.create_graph_from_mock_data()
        graph.add_node(
            5, x=2.0, y=2.0, z=2.0
        )  # Missing struct_type and radius
        with self.assertRaises(KeyError):
            graph.save_swc("dummy_path.swc")

    def test_save_as_swc_io_error(self):
        """Test handling I/O error when saving as SWC."""
        graph = self.create_graph_from_mock_data()
        with mock.patch("builtins.open", mock.mock_open()) as mocked_open:
            mocked_open.side_effect = IOError("Failed to open file")
            with self.assertRaises(IOError):
                graph.save_swc("dummy_path.swc")

    def test_to_dataframe(self):
        """Test converting the graph to a pandas DataFrame with exact match."""

        graph = self.create_graph_from_mock_data()
        df = graph.to_dataframe()
        
        expected_df = pd.DataFrame.from_dict(
            {
                0: {"node_id":1, "struct_type": 1, "x": 0.0, "y": 0.0, "z": 0.0, "radius": 0.5, "parent_id": -1},
                1: {"node_id":2, "struct_type": 3, "x": 1.0, "y": 0.0, "z": 0.0, "radius": 0.3, "parent_id": 1},
                2: {"node_id":3, "struct_type": 3, "x": 1.0, "y": 1.0, "z": 0.0, "radius": 0.4, "parent_id": 1},
                3: {"node_id":4, "struct_type": 2, "x": 0.0, "y": 1.0, "z": 0.0, "radius": 0.6, "parent_id": 1},
            },
            orient="index",
        ).astype(df.dtypes.to_dict()) 
        assert_frame_equal(df.sort_index(), expected_df.sort_index())
        
        
if __name__ == "__main__":
    unittest.main()
