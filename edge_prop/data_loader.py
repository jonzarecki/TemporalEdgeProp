from os.path import join
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from edge_prop.graph_wrappers.binary_labeled_graph import BinaryLabeledGraph
from edge_prop.models.dense_edge_propagation import DenseEdgeProp


class DataLoader:
    def __init__(self, path: str, test_size: float = 0.2, no_label: int = DenseEdgeProp.NO_LABEL,
                 dtype_tuples=[('label', int)]):
        self.path = path
        self.test_size = test_size
        self.no_label = no_label
        self.dtype_tuples = dtype_tuples

    def load_data(self, trunc_nodes: int = None):
        if 'aminer' in self.path:
            sources_train, destinations_train, labels_train, _, _ = self._get_triples(join(self.path, 'train.txt'))
            edges_train = list(zip(sources_train, destinations_train))
            labels_train = [[y] if not isinstance(y, list) else y for y in labels_train]
            edges_train = edges_train[:1000] #TODO: remove only first 1000 edges
            labels_train = labels_train[:1000] #TODO: remove only first 1000 edges
            sources_test, destinations_test, labels_test, _, _ = self._get_triples(join(self.path, 'valid.txt'))
            edges_test = list(zip(sources_test, destinations_test))
            labels_test = [[y] if not isinstance(y, list) else y for y in labels_test]
            edges_test = edges_test[:1000] #TODO: remove only first 1000 edges
            labels_test = labels_test[:1000] #TODO: remove only first 1000 edges


            edges = np.concatenate([edges_train, edges_test])
            graph = nx.from_edgelist(edges)

            edge2label = {}
            edge2label.update({edge: label for edge, label in zip(edges_train, labels_train)})
            edge2label.update({edge: label for edge, label in zip(edges_test, labels_test)})
            nx.set_edge_attributes(graph, edge2label, 'label')

            edge2label = nx.get_edge_attributes(graph, 'label')
            edges = list(edge2label.keys())
            true_lables = np.array(list(edge2label.values()))
            test_indices = [i for i, edge in enumerate(edges) if edge in edges_test]

            edge2label = {}
            edge2label.update({edge: label for edge, label in zip(edges_train, labels_train)})
            edge2label.update({edge: [self.no_label] for edge, label in zip(edges_test, labels_test)})
            nx.set_edge_attributes(graph, edge2label, 'label')
        else:
            graph = nx.read_edgelist(self.path, comments='#', data=self.dtype_tuples)
            if trunc_nodes is not None:
                graph.remove_nodes_from(map(str, range(trunc_nodes, graph.number_of_nodes())))

            edge2label = nx.get_edge_attributes(graph, 'label')
            edges = list(edge2label.keys())
            true_lables = np.array([list(edge2label.values())]).T
            if len(np.unique(true_lables)) == 2:  # binary case
                true_lables[true_lables < 0] = 0

            indices = np.arange(len(true_lables))
            train_indices, test_indices = train_test_split(indices, test_size=self.test_size)

            edge2label = {}
            edge2label.update({edges[i]: true_lables[i] for i in train_indices})
            edge2label.update({edges[i]: [self.no_label] for i in test_indices})
            nx.set_edge_attributes(graph, edge2label, 'label')

        true_lables = MultiLabelBinarizer().fit_transform(true_lables)
        binary_labeled_graph = BinaryLabeledGraph(graph, 'label')

        return binary_labeled_graph, true_lables, test_indices

    @staticmethod
    def _get_triples(path):
        headList = []
        tailList = []
        relationList = []
        headSet = []
        tailSet = []
        f = open(path, "r")
        content = f.readline()
        global tripleTotal, entityTotal, tagTotal
        tripleTotal, entityTotal, tagTotal = [int(i) for i in content.strip().split()]
        for i in range(entityTotal):
            headSet.append(set())
            tailSet.append(set())
        while (True):
            content = f.readline()
            if content == "":
                break
            values = content.strip().split()
            values = [(int)(i) for i in values]
            headList.append(values[0])
            tailList.append(values[1])
            headSet[values[0]].add(values[1])
            tailSet[values[1]].add(values[0])
            relationList.append(values[2:])
        f.close()
        return headList, tailList, relationList, headSet, tailSet