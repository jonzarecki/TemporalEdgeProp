import networkx as nx
import numpy as np

from edge_prop.models import BaseModel
from edge_prop.graph_wrappers import BinaryLabeledGraph
from edge_prop.common.utils import initialize_distributions, perform_edge_prop_on_graph


class GraphEdgePropagation(BaseModel):
    """
    Edge Propgation

    EXPECTS non-multi edge graphs
    Parameters
    ------------
    y_attr: The edge attr containing the label

    max_iter: integer
            Change maximum number of iterations allowed

    tol: float
            Convergence tolerance: threshold to consider the system at a steady state

    """
    _variant = 'propagation'

    def __init__(self, y_attr: str = 'label', max_iter=50, tol=1e-3):
        super(GraphEdgePropagation, self).__init__(y_attr, max_iter, tol)

    def predict(self):
        """
        Predict labels across all edges

        Parameters
        ----------

        Returns
        ------
        y : array_like, shape = [n_edges]
            Predictions for entire graph

        """
        return np.sign(self.edge_prop_results)  # only for 2-class for now

    def fit(self, g: BinaryLabeledGraph):
        """
        Uses the laplacian matrix to act as affinity matrix for the label-prop alg'
        :param g: The graph


        Returns
        -------
        self : returns a pointer to self
        """

        label_distributions, edge_exists, labeled, y_static = initialize_distributions(g)
        # actual graph construction
        graph_matrix = - nx.normalized_laplacian_matrix(g.graph_nx, nodelist=g.node_order)
        graph_matrix.setdiag(0)  # graph_matrix = graph_matrix - np.diag(np.diagonal(graph_matrix))

        graph_matrix = graph_matrix.toarray()
        label_distributions = perform_edge_prop_on_graph(graph_matrix, label_distributions, edge_exists,
                                                         labeled, y_static)

        # set the results
        self.edge_prop_results = np.zeros(g.n_edges)  # will hold the results
        for i, (u, v) in enumerate(g.edge_order):
            edge_idxs = g.node_to_idx[u], g.node_to_idx[v]
            self.edge_prop_results[i] = label_distributions[edge_idxs]
        return self

class GraphEdgePropagation(BaseModel):
    """
    Edge Propgation

    EXPECTS non-multi edge graphs
    Parameters
    ------------
    y_attr: The edge attr containing the label

    max_iter: integer
            Change maximum number of iterations allowed

    tol: float
            Convergence tolerance: threshold to consider the system at a steady state

    """
    _variant = 'propagation'

    def __init__(self, y_attr: str = 'label', max_iter=50, tol=1e-3):
        super(GraphEdgePropagation, self).__init__(y_attr, max_iter, tol)

    def predict(self):
        """
        Predict labels across all edges

        Parameters
        ----------

        Returns
        ------
        y : array_like, shape = [n_edges]
            Predictions for entire graph

        """
        return np.sign(self.edge_prop_results)  # only for 2-class for now

    def fit(self, g: BinaryLabeledGraph):
        """
        Uses the laplacian matrix to act as affinity matrix for the label-prop alg'
        :param g: The graph


        Returns
        -------
        self : returns a pointer to self
        """

        label_distributions, edge_exists, labeled, y_static = initialize_distributions(g)
        # actual graph construction
        graph_matrix = - nx.normalized_laplacian_matrix(g.graph_nx, nodelist=g.node_order)
        graph_matrix.setdiag(0)  # graph_matrix = graph_matrix - np.diag(np.diagonal(graph_matrix))

        graph_matrix = graph_matrix.toarray()
        label_distributions = perform_edge_prop_on_graph(graph_matrix, label_distributions, edge_exists,
                                                         labeled, y_static)

        # set the results
        self.edge_prop_results = np.zeros(g.n_edges)  # will hold the results
        for i, (u, v) in enumerate(g.edge_order):
            edge_idxs = g.node_to_idx[u], g.node_to_idx[v]
            self.edge_prop_results[i] = label_distributions[edge_idxs]
        return self
