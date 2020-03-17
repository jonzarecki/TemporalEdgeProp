import networkx as nx

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
        self.g_nx = g

        self.node_order = sorted(g.nodes)
        self.n_nodes = g.number_of_nodes()
        self.node_to_idx = {node: i for (i, node) in enumerate(self.node_order)}

        self.edge_order = sorted(g.edges)
        self.n_edges = g.number_of_edges()

        # only used when this is an aggregated graph from BinaryLabeledTemporalGraph.build_aggregated_graph()
        self.agg_e_maps_to = None
        self.edge_timestamp_in_order = None
