from unittest import TestCase
import networkx as nx
import numpy as np
from edge_prop.graph_wrappers import BaseGraph
from edge_prop.models import DenseEdgeProp
from edge_prop.constants import NO_LABEL


class TestDenseEdgeProp(TestCase):
    def setUp(self) -> None:
        g = nx.Graph()
        g.add_edge(0, 1, label=[0])
        g.add_edge(1, 2, label=[1])
        g.add_edge(0, 2, label=[1])
        g.add_edge(3, 4, label=[NO_LABEL])
        g.add_edge(4, 5, label=[2])
        g.add_edge(5, 6, label=[NO_LABEL])
        self.graph = BaseGraph(g)
        self.true_labels = np.array([0, 1, 1, 2, 2, 2])

        self.edge_prop_model = DenseEdgeProp()

    def test_fit(self):
        self.edge_prop_model.fit(self.graph, 'label')
        self.assertIsNotNone(self.edge_prop_model.edge_distributions)
        self.assertEqual(self.edge_prop_model.edge_distributions.sum(), self.graph.n_edges * 2)

    def test_predict(self):
        self.edge_prop_model.fit(self.graph, 'label')
        results = self.edge_prop_model.predict()
        self.assertListEqual(list(results), list(self.true_labels))


    def test_predict2(self):
        g = nx.Graph()
        g.add_edge(0, 2, label=[1])
        g.add_edge(1, 2, label=[NO_LABEL])
        g.add_edge(2, 3, label=[NO_LABEL])
        g.add_edge(3, 4, label=[0])
        g.add_edge(3, 5, label=[NO_LABEL])
        graph = BaseGraph(g)
        print(graph.edge_order)
        true_labels = np.array([1, 1, 0, 0, 0])
        edge_prop_model = DenseEdgeProp()
        edge_prop_model.fit(graph, 'label')
        results = edge_prop_model.predict()
        self.assertListEqual(list(results), list(true_labels))

    def test_predict3(self):
        g = nx.Graph()
        g.add_edge(0, 1, label=[0])
        g.add_edge(1, 2, label=[NO_LABEL])
        g.add_edge(2, 3, label=[1])
        graph = BaseGraph(g)
        print(graph.edge_order)
        true_labels = np.array([0, 0, 1])
        edge_prop_model = DenseEdgeProp()
        edge_prop_model.fit(graph, 'label')
        results = edge_prop_model.predict()
        self.assertListEqual(list(results), list(true_labels))

    def test_predict4(self):
        g = nx.Graph()
        g.add_edge(0, 1, label=[NO_LABEL])
        g.add_edge(1, 2, label=[1])
        g.add_edge(3, 4, label=[NO_LABEL])
        g.add_edge(4, 5, label=[NO_LABEL])
        g.add_edge(5, 6, label=[0])
        graph = BaseGraph(g)
        true_labels = np.array([1, 1, 0, 0, 0])
        edge_prop_model = DenseEdgeProp()
        edge_prop_model.fit(graph, 'label')
        results = edge_prop_model.predict()
        self.assertListEqual(list(results), list(true_labels))
