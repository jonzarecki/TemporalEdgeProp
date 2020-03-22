from unittest import TestCase
import unittest
import numpy as np
import networkx as nx
from datetime import datetime, timedelta

from edge_prop.common.graph_label_prop.label_prop import GraphLabelPropagation


class TestLabelProp(TestCase):
    def test_label_prop(self):
        """
        Easy test for sanity for normal LabelProp
        """
        g = nx.Graph()
        g.add_node(0, label=-1)
        g.add_edge(0, 1)
        g.add_node(2, label=1)
        g.add_edge(1, 2)
        g.add_edge(2, 4)
        g.add_edge(2, 3)
        true_labels = np.array([-1, 1, 1, 1, 1])

        initial_node_labels = nx.get_node_attributes(g, 'label')
        labels_sorted = [initial_node_labels.setdefault(n, 0) for n in sorted(g.nodes)]
        # prop_model = GraphLabelPropagation(y_attr='label')  #TODO: might be a good idea
        prop_model = GraphLabelPropagation()
        prop_model.fit(g, labels_sorted)

        pred = prop_model.predict()

        self.assertTrue(np.array_equal(pred, true_labels))