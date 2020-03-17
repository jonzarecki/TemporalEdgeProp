import networkx as nx

from edge_prop.graph_wrappers.binary_labeled_graph import BinaryLabeledGraph


class DataLoader:
    def __init__(self, path:str):
        self.path = path

    def load_data(self):

        graph = nx.read_edgelist(self.path, comments = '#', data=[('label', int)])
        binary_labeled_graph = BinaryLabeledGraph(graph, 'label')
        return binary_labeled_graph