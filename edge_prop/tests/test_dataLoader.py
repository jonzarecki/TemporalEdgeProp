from unittest import TestCase

from edge_prop.data_loader import DataLoader
from edge_prop.models import BinaryLabeledGraph


class TestDataLoader(TestCase):
    def setUp(self) -> None:
        path = r'/data/home/morpheus/edge_prop_data/soc-sign-epinions.txt'
        self.data_loader = DataLoader(path)

    def test_load_data(self):
        graph = self.data_loader.load_data()

        self.assertIsInstance(graph, BinaryLabeledGraph)
        self.assertGreater(graph.n_nodes, 100)
        self.assertGreater(graph.n_edges, 100)


