import networkx as nx
import numpy as np
from sparse import COO


class BaseGraph:
    """
    Util class for keeping useful info on graphs

    g_nx - the graphs as nx object
    node_order - sorted order of the nodes
    n_nodes - number of nodes
    node_to_idxs - dictionary that maps node to it's idx

    edge_order - sorted order of the edges
    n_edges - number of edges
    """

    def __init__(self, g: nx.Graph, **kwargs):
        self.graph_nx = g

        self.node_order = sorted(g.nodes)
        self.n_nodes = g.number_of_nodes()
        self.node_to_idx = {node: i for (i, node) in enumerate(self.node_order)}

        self.edge_order = sorted(g.edges)
        self.n_edges = g.number_of_edges()

        # only used when this is an aggregated graph from BinaryLabeledTemporalGraph.build_aggregated_graph()
        self.agg_e_maps_to = None
        self.edge_timestamp_in_order = None

    def adjacency_matrix(self, sparse=True):
        sadj_mat = nx.adjacency_matrix(self.graph_nx, nodelist=self.node_order)

        if not sparse:
            return np.asarray(sadj_mat.todense(), dtype=bool)
        else:
            return COO.from_scipy_sparse(sadj_mat)

    def get_edge_attributes_ordered(self, label):
        edge_label_dict = nx.get_edge_attributes(self.graph_nx, label)
        edge_labels = [(e, edge_label_dict[e]) for e in self.edge_order]
        return edge_labels
