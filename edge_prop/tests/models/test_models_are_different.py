import unittest
import numpy as np
import networkx as nx
from datetime import datetime, timedelta

from edge_prop.models import DenseBasline, DenseEdgeProp
from edge_prop.graph_wrappers import BaseGraph
from edge_prop.constants import NO_LABEL

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
        g = BaseGraph(g)
        true_labels = np.array([1, 1, -1, -1, -1])

        edge_prop_model = DenseBasline()
        edge_prop_model.fit(g, 'label')

        pred = edge_prop_model.predict()

        self.assertTrue(np.array_equal(pred, true_labels))

    def test_are_different(self):
        """
        Easy test for sanity for normal EdgeProp
        """
        g = nx.Graph()
        g.add_edge(0, 1, label=1)
        g.add_edge(1, 2, label=NO_LABEL)
        g.add_edge(2, 3, label=NO_LABEL)
        g.add_edge(3, 4, label=0)
        g.add_edge(3, 5, label=NO_LABEL)
        graph = BaseGraph(g)
        true_labels = np.array([-1, -1, 1, 1, 1])

        baseline = DenseBasline(max_iter=1000, alpha=0.8)
        baseline.fit(graph, 'label')
        y_pred2 = baseline.predict_proba()

        edge_prop = DenseEdgeProp(max_iter=1000, alpha=0.8)
        edge_prop.fit(graph, 'label')

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
        g.add_edge(1, 2, label=NO_LABEL)
        g.add_edge(2, 3, label=1)
        g.add_edge(2, 4, label=NO_LABEL)
        graph = BaseGraph(g)
        true_labels = np.array([-1, -1, 1, 1, 1])

        baseline = DenseBasline(max_iter=1000, alpha=0.8)
        baseline.fit(graph, 'label')
        y_pred2 = baseline.predict_proba()

        edge_prop = DenseEdgeProp(max_iter=1000, alpha=0.8)
        edge_prop.fit(graph, 'label')

        y_pred = edge_prop.predict_proba()
        print(y_pred[:,:,0])
        print(y_pred2[:,:,0])
        self.assertFalse(np.array_equal(edge_prop.predict(), baseline.predict()))

    @staticmethod
    def days_from_now(days: int) -> timedelta:
        return datetime.now() - timedelta(days=days)

