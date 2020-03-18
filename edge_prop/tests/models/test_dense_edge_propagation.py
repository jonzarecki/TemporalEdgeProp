from unittest import TestCase
import networkx as nx
import numpy as np
from edge_prop.graph_wrappers import BinaryLabeledGraph
from edge_prop.models.dense_edge_propagation import DenseEdgeProp


class TestDenseEdgeProp(TestCase):
    def setUp(self) -> None:
        g = nx.Graph()
        no_label = DenseEdgeProp.NO_LABEL
        g.add_edge(0, 1, label=0)
        g.add_edge(1, 2, label=1)
        g.add_edge(0, 2, label=1)
        g.add_edge(3, 4, label=no_label)
        g.add_edge(4, 5, label=2)
        g.add_edge(5, 6, label=no_label)
        self.graph = BinaryLabeledGraph(g, 'label')
        self.true_labels = np.array([0, 1, 1, 2, 2, 2])

        self.edge_prop_model = DenseEdgeProp(y_attr='label')

    def test_fit(self):
        self.edge_prop_model.fit(self.graph)
        #TODO add asserts

    def test_predict(self):
        self.edge_prop_model.fit(self.graph)
        results = self.edge_prop_model.predict()
        #TODO add asserts


