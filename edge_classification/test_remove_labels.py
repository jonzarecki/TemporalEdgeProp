from unittest import TestCase
import networkx as nx

from edge_classification.data_preperation import remove_labels
from edge_classification.graph_wrappers.binary_labeled_graph import BinaryLabeledGraph


class TestRemove_labels(TestCase):
    def setUp(self) -> None:
        g = nx.Graph()
        g.add_edge(0, 1, label=1)
        g.add_edge(1, 2, label=1)
        g.add_edge(3, 4, label=-1)
        g.add_edge(4, 5, label=1)
        g.add_edge(5, 6, label=-1)
        self.graph = BinaryLabeledGraph(g, 'label')

    def test_remove_labels(self):
        new_graph, y_true, label_mask = remove_labels(self.graph, 0.8)

        self.assertTrue(0 in new_graph.edge_label_dict.values())
        self.assertTrue(0 not in y_true)
        self.assertEqual(label_mask.sum(), 4)
