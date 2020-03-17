import networkx as nx
import numpy as np

from sklearn.semi_supervised._label_propagation import BaseLabelPropagation


class TemporalGraphLabelPropagation(BaseLabelPropagation):
    """
        Class for the temporal LabelProp alg'
        - node_time: the node attribute name containing the time in datetime format

    """
    def __init__(self, node_time: str, *args, **kwargs):
        super(TemporalGraphLabelPropagation, self).__init__(*args, **kwargs)
        self.node_time = node_time

    def _build_graph(self):
        """
        here we assume self.X already hold our affinity matyrix as calculated by networkx
        """
        return self.X_

    @staticmethod
    def decay_function(times_list: np.ndarray, measure_from: float) -> np.ndarray:
        """
        Calculates the decayed time function for all times in $timeslist in comparison with measure_from timestamp
        Args:
            times_list: list of timestamps in np array
            measure_from: timestamp to start measuring from

        Returns:
            Decayed scores for all times in $times_list
        """
        alpha = 1.
        beta = -0.0005
        return alpha * np.exp(beta * np.abs(measure_from - times_list) / (60**2))

    def fit(self, g: nx.Graph, y):
        """
        Uses the laplacian matrix to acy as affinity matrix for the label-prop algorithm
        Args:
            g: The graph in nx format
            y: the edge label we expect

        Returns:
            self
        """
        weight_name = 'TGLP_weight'
        edge_weight = {}
        for n1, n2, attrs in g.edges(data=True):
            edge_weight[(n1, n2)] = (self.decay_function(g.nodes[n1][self.node_time]) +
                                     self.decay_function(g.nodes[n2][self.node_time])) / 2.0

        nx.set_edge_attributes(g, edge_weight, weight_name)
        X = - nx.normalized_laplacian_matrix(g, nodelist=sorted(g.nodes), weight=weight_name).toarray()
        X = X - np.diag(np.diagonal(X))
        retval = super(TemporalGraphLabelPropagation, self).fit(X, y)
        nx.set_edge_attributes(g, {}, weight_name)  # delete attr
        return retval
