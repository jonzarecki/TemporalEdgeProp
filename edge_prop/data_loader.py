import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

from edge_prop.graph_wrappers.binary_labeled_graph import BinaryLabeledGraph


class DataLoader:
    def __init__(self, path: str, test_size=0.2):
        self.path = path
        self.test_size = test_size

    def load_data(self, trunc_nodes:int = None):
        graph = nx.read_edgelist(self.path, comments='#', data=[('label', int)])
        if trunc_nodes is not None:
            graph.remove_nodes_from(map(str, range(trunc_nodes, graph.number_of_nodes())))

        edge2label = nx.get_edge_attributes(graph, 'label')
        edges = list(edge2label.keys())
        true_lables = np.array(list(edge2label.values()))
        true_lables[true_lables < 0] = true_lables.max() + 1

        indices = np.arange(len(true_lables))
        train_indices, test_indices = train_test_split(indices, test_size=self.test_size)

        edge2label = {}
        edge2label.update({edges[i]: true_lables[i] for i in train_indices})
        edge2label.update({edges[i]: 0 for i in test_indices})
        nx.set_edge_attributes(graph, edge2label, 'label')

        binary_labeled_graph = BinaryLabeledGraph(graph, 'label')

        return binary_labeled_graph, true_lables
