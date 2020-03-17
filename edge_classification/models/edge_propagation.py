import gc
import os
import warnings
from abc import ABCMeta
from typing import Tuple

import scipy

import networkx as nx
import numpy as np
import six
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import safe_sparse_dot

from edge_classification.graph_wrappers.binary_labeled_graph import BinaryLabeledGraph

EDGEPROP_BASE_DIR = os.path.dirname(__file__) + "/"


class GraphEdgePropagation(six.with_metaclass(ABCMeta), BaseEstimator, ClassifierMixin):
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

    def __init__(self, y_attr: str, max_iter=50, tol=1e-3):
        self.y_attr = y_attr
        self.max_iter = max_iter
        self.tol = tol
        self.edge_prop_results = None  # keeps results after run

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
        graph_matrix = - nx.normalized_laplacian_matrix(g.g_nx, nodelist=g.node_order)
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


def initialize_distributions(g: BinaryLabeledGraph) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix,
                                                             scipy.sparse.dok_matrix, scipy.sparse.csr_matrix]:
    """
    Initializes the distribution for edge_prop and temporal edge_prop
    :param g: Labeled graph to initialize the distributions on
    :return:
        - label_distributions - csr_matrix on which we calculate the -1,1 score for each edge
        - edge_exists - csr_matrix which as 1 only where an edge exists in $g
        - labeled - dok_matrix which has 1 only where a labeled edge exists
        - y_static - csr_matrix which has the initial label distributions (only from labeled edges)
    """
    gc.collect()
    label_distributions = scipy.sparse.lil_matrix((g.n_nodes, g.n_nodes), dtype=float)
    edge_exists = scipy.sparse.dok_matrix((g.n_nodes, g.n_nodes), dtype=bool)
    for ((u, v), label) in g.edge_labels:
        edge = g.node_to_idx[u], g.node_to_idx[v]
        reverse_edge = tuple(reversed(edge))
        label_distributions[edge] = label
        label_distributions[reverse_edge] = label
        edge_exists[edge] = 1
        edge_exists[reverse_edge] = 1
    label_distributions = label_distributions.tocsr()
    edge_exists = edge_exists.tocsr()
    y_static = scipy.sparse.dok_matrix(label_distributions)
    labeled = (y_static != 0)

    return label_distributions, edge_exists, labeled, y_static


def perform_edge_prop_on_graph(graph_matrix: scipy.sparse.csr_matrix, label_distributions: scipy.sparse.csr_matrix,
                               edge_exists: scipy.sparse.csr_matrix, labeled: scipy.sparse.dok_matrix,
                               y_static: scipy.sparse.csr_matrix, max_iter=50, tol=1e-3) -> scipy.sparse.csr_matrix:
    """
    Performs the EdgeProp algorithm on the given graph.
    returns the label distribution (|N|, |N|) matrix with scores between -1, 1 stating the calculated label distribution.


    :param graph_matrix: matrix representing the info flow in the graph
    :param label_distributions: csr_matrix on which we calculate the -1,1 scores for each edge
    :param edge_exists: csr_matrix which has 1 only where an edge exists in $g
    :param labeled: dok_matrix which has 1 only where a labeled edge exists
    :param y_static: csr_matrix which has the initial label distributions (only from labeled edges)
    :param max_iter: maximum iteration number for EdgeProp
    :param tol: epsilon number to check whether the distribution matrix did not change
    """
    labeled_arr = y_static[labeled]
    label_distributions = label_distributions.copy()
    l_previous = None
    for n_iter in range(max_iter):
        if n_iter != 0 and np.abs(label_distributions - l_previous).sum() < tol:  # did not change
            break  # end the loop, finished
        l_previous = label_distributions.copy()
        step1 = safe_sparse_dot(l_previous, graph_matrix)
        B = np.asarray(np.sum(step1, axis=0))  # sum aggregates the data into nodes
        # expand for data on edges, average of the 2 nodes
        mat = (edge_exists.multiply(B) + edge_exists.multiply(B.T)) / 2.0  # weird to keep sparsity
        mat[mat > 1.0] = 1.0
        mat[mat < -1.0] = -1.0
        mat[labeled] = labeled_arr  # keep original labeled edges
        label_distributions = mat
    else:
        warnings.warn("max_iter was reached without convergence", category=ConvergenceWarning)
        n_iter += 1

    return label_distributions
