from os.path import join
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from edge_prop.graph_wrappers import BaseGraph
from edge_prop.constants import LABEL_GT, LABEL_TRAIN, NO_LABEL


class DataLoader:
    def __init__(self, path: str, test_size: float = 0.2, dtype_tuples=[(LABEL_GT, int)]):
        self.path = path
        self.test_size = test_size
        self.dtype_tuples = dtype_tuples

    def load_data(self, trunc_nodes: int = None):
        if 'aminer' in self.path:
            graph = self._load_aminer(self.path)
        elif 'epinions' in self.path:
            graph = self._load_epinions(self.path, trunc_nodes)
        else:
            raise Exception('No such dataset exists')

        g = BaseGraph(graph)
        test_indices = np.array(
            [i for i, (_, label) in enumerate(g.get_edge_attributes(LABEL_TRAIN)) if label == [NO_LABEL]])
        test_indices = np.array([i for i, (_, label) in enumerate(g.get_edge_attributes(LABEL_GT)) if
                                 label != [NO_LABEL] and i in test_indices])

        true_labels = np.array([label for _, label in g.get_edge_attributes(LABEL_GT)])
        y_test = MultiLabelBinarizer().fit_transform(true_labels)
        for cur_y_test, true_label in zip(y_test, true_labels):
            if not (sum(cur_y_test[true_label]) == len(true_label) == sum(cur_y_test)):
                raise Exception('Classes got binarized not in the right order')

        return g, y_test, test_indices

    def _load_epinions(self, path, trunc_nodes):
        graph = nx.read_edgelist(path, comments='#', data=self.dtype_tuples)
        if trunc_nodes is not None:
            graph.remove_nodes_from(map(str, range(trunc_nodes, graph.number_of_nodes())))
        edge2label = nx.get_edge_attributes(graph, LABEL_GT)
        edge2label = {edge: [0 if label < 0 else label] for edge, label in edge2label.items()}
        nx.set_edge_attributes(graph, edge2label, LABEL_GT)
        _, test_edges = train_test_split(edge2label.keys(), test_size=self.test_size)
        edge2label.update({edge: [NO_LABEL] for i, edge in test_edges})
        nx.set_edge_attributes(graph, edge2label, LABEL_TRAIN)
        return graph

    def _load_aminer(self, path):
        edges_train, labels_train = DataLoader._get_triples(join(path, 'train.txt'))
        edges_test, labels_test = DataLoader._get_triples(join(path, 'valid.txt'))
        graph = nx.from_edgelist(np.concatenate([edges_train, edges_test]))
        edge2label = {}
        edge2label.update({edge: label for edge, label in zip(edges_train, labels_train)})
        edge2label.update({edge: label for edge, label in zip(edges_test, labels_test)})
        nx.set_edge_attributes(graph, edge2label, LABEL_GT)
        edge2label.update({edge: [NO_LABEL] for edge in edges_test})
        nx.set_edge_attributes(graph, edge2label, LABEL_TRAIN)
        return graph

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

        edges = list(zip(headList, tailList))
        relationList = [[label] if not isinstance(label, list) else label for label in relationList]

        return edges, relationList
