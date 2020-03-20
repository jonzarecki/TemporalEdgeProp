import abc
import warnings
from abc import ABCMeta, ABC

import six
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import networkx as nx
from sparse import DOK, COO

from edge_prop.graph_wrappers import BaseGraph, BinaryLabeledGraph


class BaseModel(six.with_metaclass(ABCMeta), BaseEstimator, ClassifierMixin):
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
    NO_LABEL = -1

    def __init__(self, y_attr: str, max_iter: int = 50, tol: float = 1e-3, alpha: float = 1):
        self.y_attr = y_attr
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter

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
        results = np.zeros((self.graph.n_edges, self.edge_distributions.shape[2]), dtype=np.int)  # will hold the results
        for i, (u, v) in enumerate(self.graph.edge_order):
            edge_idxs = self.graph.node_to_idx[u], self.graph.node_to_idx[v]
            dist = self.edge_distributions[edge_idxs]  # label distribution
            # if len(dist[dist == dist.max()]) > 1:
            #     warnings.warn(f"edge {(u, v)} doesn't have a definitive max: {dist}", category=RuntimeWarning)
            results[i] = dist#.argmax()
        # results = np.ones_like(self.edge_distributions[:, :, 0]) * self.NO_LABEL
        # edge_exists = self.edge_distributions.sum(axis=-1) != 0
        # results[edge_exists] = self.edge_distributions.argmax(axis=-1)[edge_exists]
        return results

    def predict_proba(self):
        return self.edge_distributions

    def fit(self, g: BinaryLabeledGraph):
        """
        Uses the laplacian matrix to act as affinity matrix for the label-prop alg'
        :param g: The graph


        Returns
        -------
        self : returns a pointer to self
        """
        self.graph = g
        self._classes = self._get_classes(g)
        adj_mat = g.adjacency_matrix(sparse=False)
        y = self._create_y(g)

        self.edge_distributions = self._perform_edge_prop_on_graph(adj_mat, y, max_iter=self.max_iter, tol=self.tol)
        return self

    @abc.abstractmethod
    def _perform_edge_prop_on_graph(self, adj_mat: np.ndarray, y: np.ndarray, max_iter=100,
                                    tol=1e-1) -> np.ndarray:
        """
        Performs the EdgeProp algorithm on the given graph.
        returns the label distribution (|N|, |N|) matrix with scores between -1, 1 stating the calculated label distribution.
        """
        pass

    def _get_classes(self, g: BinaryLabeledGraph) -> np.ndarray:
        classes = np.unique([label for _, y in g.edge_labels for label in y])
        classes = classes[classes != self.NO_LABEL]
        return classes

    def _create_y(self, g):
        y = np.zeros((g.n_nodes, g.n_nodes, len(self._classes)))
        for ((u, v), labels) in g.edge_labels:
            edge = g.node_to_idx[u], g.node_to_idx[v]
            reverse_edge = tuple(reversed(edge))
            for label in labels:
                if label != self.NO_LABEL:
                    y[edge][label] = 1/len(labels)
                    y[reverse_edge][label] = 1/len(labels)
        return y
