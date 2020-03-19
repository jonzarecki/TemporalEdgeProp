from unittest import TestCase
import networkx as nx
import numpy as np
from edge_prop.graph_wrappers import BinaryLabeledGraph
from edge_prop.models.dense_edge_propagation import DenseEdgeProp


class TestDenseEdgeProp(TestCase):
    def setUp(self) -> None:
        g = nx.Graph()
        g.add_edge(0, 1, label=0)
        g.add_edge(1, 2, label=1)
        g.add_edge(0, 2, label=1)
        g.add_edge(3, 4, label=DenseEdgeProp.NO_LABEL)
        g.add_edge(4, 5, label=2)
        g.add_edge(5, 6, label=DenseEdgeProp.NO_LABEL)
        self.graph = BinaryLabeledGraph(g, 'label')  # TODO: not binary anymore
        self.true_labels = np.array([0, 1, 1, 2, 2, 2])

        self.edge_prop_model = DenseEdgeProp(y_attr='label')

    def test_fit(self):
        self.edge_prop_model.fit(self.graph)
        self.assertIsNotNone(self.edge_prop_model.edge_distributions)
        self.assertEqual(self.edge_prop_model.edge_distributions.sum(), self.graph.n_edges * 2)

    def test_predict(self):
        self.edge_prop_model.fit(self.graph)
        results = self.edge_prop_model.predict()
        self.assertListEqual(list(results), list(self.true_labels))


    def test_predict2(self):
        g = nx.Graph()
        g.add_edge(0, 2, label=1)
        g.add_edge(1, 2, label=DenseEdgeProp.NO_LABEL)
        g.add_edge(2, 3, label=DenseEdgeProp.NO_LABEL)
        g.add_edge(3, 4, label=0)
        g.add_edge(3, 5, label=DenseEdgeProp.NO_LABEL)
        graph = BinaryLabeledGraph(g, 'label')
        print(graph.edge_order)
        true_labels = np.array([1, 1, 0, 0, 0])
        edge_prop_model = DenseEdgeProp(y_attr='label')
        edge_prop_model.fit(graph)
        results = edge_prop_model.predict()
        self.assertListEqual(list(results), list(true_labels))

    def test_predict3(self):
        g = nx.Graph()
        g.add_edge(0, 1, label=0)
        g.add_edge(1, 2, label=DenseEdgeProp.NO_LABEL)
        g.add_edge(2, 3, label=1)
        graph = BinaryLabeledGraph(g, 'label')
        print(graph.edge_order)
        true_labels = np.array([0, 0, 1])
        edge_prop_model = DenseEdgeProp(y_attr='label')
        edge_prop_model.fit(graph)
        results = edge_prop_model.predict()
        self.assertListEqual(list(results), list(true_labels))

    def test_predict4(self):
        g = nx.Graph()
        g.add_edge(0, 1, label=DenseEdgeProp.NO_LABEL)
        g.add_edge(1, 2, label=1)
        g.add_edge(3, 4, label=DenseEdgeProp.NO_LABEL)
        g.add_edge(4, 5, label=DenseEdgeProp.NO_LABEL)
        g.add_edge(5, 6, label=0)
        graph = BinaryLabeledGraph(g, 'label')  # TODO: not binary anymore
        true_labels = np.array([1, 1, 0, 0, 0])
        edge_prop_model = DenseEdgeProp(y_attr='label')
        edge_prop_model.fit(graph)
        results = edge_prop_model.predict()
        self.assertListEqual(list(results), list(true_labels))
