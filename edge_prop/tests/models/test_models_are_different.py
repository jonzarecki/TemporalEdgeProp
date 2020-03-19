import unittest
import numpy as np
import networkx as nx
from datetime import datetime, timedelta

from edge_prop.models.base_model import BaseModel
from edge_prop.models.dense_baseline import DenseBasline
from edge_prop.models.dense_edge_propagation import DenseEdgeProp
from edge_prop.models.edge_propagation import GraphEdgePropagation
from edge_prop.graph_wrappers import BinaryLabeledTemporalGraph, BinaryLabeledGraph
from edge_prop.models.temporal_edge_propagation import TemporalGraphEdgePropagation

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

    def test_are_different(self):
        """
        Easy test for sanity for normal EdgeProp
        """
        g = nx.Graph()
        g.add_edge(0, 1, label=1)
        g.add_edge(1, 2, label=BaseModel.NO_LABEL)
        g.add_edge(2, 3, label=BaseModel.NO_LABEL)
        g.add_edge(3, 4, label=0)
        g.add_edge(3, 5, label=BaseModel.NO_LABEL)
        graph = BinaryLabeledGraph(g, 'label')
        true_labels = np.array([-1, -1, 1, 1, 1])

        baseline = DenseBasline(graph.y_attr, max_iter=1000, alpha=0.8)
        baseline.fit(graph)
        y_pred2 = baseline.predict_proba()

        edge_prop = DenseEdgeProp(graph.y_attr, max_iter=1000, alpha=0.8)
        edge_prop.fit(graph)

        y_pred = edge_prop.predict_proba()
        print(y_pred[:,:,0])
        print(y_pred2[:,:,0])
        self.assertFalse(np.array_equal(y_pred2, y_pred))


    def test_are_different2(self):
        """
        Easy test for sanity for normal EdgeProp
        """
        g = nx.Graph()
        g.add_edge(0, 1, label=0)
        g.add_edge(1, 2, label=DenseEdgeProp.NO_LABEL)
        g.add_edge(2, 3, label=1)
        g.add_edge(2, 4, label=DenseEdgeProp.NO_LABEL)
        graph = BinaryLabeledGraph(g, 'label')
        true_labels = np.array([-1, -1, 1, 1, 1])

        baseline = DenseBasline(graph.y_attr, max_iter=1000, alpha=0.8)
        baseline.fit(graph)
        y_pred2 = baseline.predict_proba()

        edge_prop = DenseEdgeProp(graph.y_attr, max_iter=1000, alpha=0.8)
        edge_prop.fit(graph)

        y_pred = edge_prop.predict_proba()
        print(y_pred[:,:,0])
        print(y_pred2[:,:,0])
        self.assertFalse(np.array_equal(edge_prop.predict(), baseline.predict()))

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
