from unittest import TestCase

from edge_prop.data_loader import DataLoader
from edge_prop.graph_wrappers import BinaryLabeledGraph
from edge_prop.constants import DATASET2PATH


class TestDataLoader(TestCase):
    def setUp(self) -> None:
        self.data_loader = DataLoader(DATASET2PATH['epinions'])

    def test_load_data(self):
        graph, true_lables, test_indices = self.data_loader.load_data()

        self.assertIsInstance(graph, BinaryLabeledGraph)
        self.assertGreater(graph.n_nodes, 100)
        self.assertGreater(graph.n_edges, 100)
