import unittest
import numpy as np
import networkx as nx
from datetime import datetime, timedelta

from edge_classification.edge_propagation import GraphEdgePropagation
from edge_classification.graph_wrappers.binary_labeled_graph import BinaryLabeledGraph
from edge_classification.graph_wrappers.binary_labeled_temporal_graph import BinaryLabeledTemporalGraph
from edge_classification.temporal_edge_propagation import TemporalGraphEdgePropagation


class TestEdgePropAlgs(unittest.TestCase):
    def test_normal_edge_prop(self):
        """
        Easy test for sanity for normal EdgeProp
        """
        g = nx.Graph()
        g.add_edge(0, 1, label=0)
        g.add_edge(1, 2, label=1)
        g.add_edge(3, 4, label=0)
        g.add_edge(4, 5, label=0)
        g.add_edge(5, 6, label=-1)
        g = BinaryLabeledGraph(g, 'label')
        true_labels = np.array([1, 1, -1, -1, -1])

        edge_prop_model = GraphEdgePropagation(y_attr='label')
        edge_prop_model.fit(g)

        pred = edge_prop_model.predict()

        self.assertTrue(np.array_equal(pred, true_labels))

    @staticmethod
    def days_from_now(days: int) -> timedelta:
        return datetime.now() - timedelta(days=days)


    def test_temporal_edge_prop(self):
        """
        Easy test for sanity for Temporal EdgeProp
        """

        g = nx.Graph()
        g.add_edge(0, 1, label=1, in_test=0, time=self.days_from_now(1))
        g.add_edge(1, 2, label=0, in_test=1, time=self.days_from_now(2))
        g.add_edge(2, 3, label=-1, in_test=0, time=self.days_from_now(0))
        g.add_edge(4, 5, label=0, in_test=1, time=self.days_from_now(10))
        g.add_edge(5, 6, label=1, in_test=0, time=self.days_from_now(2))
        g.add_edge(4, 7, label=-1, in_test=0, time=self.days_from_now(5))
        g = BinaryLabeledTemporalGraph(g, 'label', 'time')
        true_labels = np.array([1, -1])

        edge_prop_model = TemporalGraphEdgePropagation(y_attr='label', in_test='in_test', time_attr='time')
        edge_prop_model.fit(g)

        pred = edge_prop_model.predict()

        self.assertTrue(np.array_equal(pred, true_labels))


if __name__ == '__main__':
    unittest.main()
