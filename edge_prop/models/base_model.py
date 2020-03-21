import abc
import warnings
from abc import ABCMeta

import six
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sparse import DOK, COO

from edge_prop.graph_wrappers import BaseGraph


class BaseModel(six.with_metaclass(ABCMeta), BaseEstimator, ClassifierMixin):
    """
    Edge Propgation

    EXPECTS non-multi edge graphs
    Parameters
    ------------
    max_iter: integer
            Change maximum number of iterations allowed

    tol: float
            Convergence tolerance: threshold to consider the system at a steady state

    """
    _variant = 'propagation'
    NO_LABEL = -1

    def __init__(self, max_iter: int = 50, tol: float = 1e-5, alpha: float = 1):
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter

        self.sparse = False

    def predict(self, indices=None):
        """
        Predict labels across all edges

        Parameters
        ----------

        Returns
        ------
        y : array_like, shape = [n_edges]
            Predictions for entire graph

        """
        if indices is None:
            indices = range(self.graph.n_edges)

        results = []
        for i in indices:
            u, v = self.graph.edge_order[i]
            dist = self.get_edge_distributions(u, v)  # label distribution
            if len(dist[dist == dist.max()]) > 1:
                warnings.warn(f"edge {(u, v)} doesn't have a definitive max: {dist}", category=RuntimeWarning)
            results.append(dist.argmax())
        results = np.array(results, dtype=np.int)
        # results = np.ones_like(self.edge_distributions[:, :, 0]) * self.NO_LABEL
        # edge_exists = self.edge_distributions.sum(axis=-1) != 0
        # results[edge_exists] = self.edge_distributions.argmax(axis=-1)[edge_exists]
        return results

    def predict_proba(self, indices=None):
        if indices is None:
            indices = range(self.graph.n_edges)

        results = []
        for i in indices:
            u, v = self.graph.edge_order[i]
            results.append(self.get_edge_distributions(u, v))
        results = np.array(results)

        return results

    def get_edge_distributions(self, u, v):
        u_index, v_index = self.graph.node_to_idx[u], self.graph.node_to_idx[v]
        if self.sparse:
            return self.edge_distributions[u_index, v_index].todense()
        else:
            return self.edge_distributions[u_index, v_index]

    def fit(self, g: BaseGraph, label):
        """
        Uses the laplacian matrix to act as affinity matrix for the label-prop alg'
        :param g: The graph


        Returns
        -------
        self : returns a pointer to self
        """
        self.graph = g
        adj_mat = g.adjacency_matrix(sparse=self.sparse)
        y = self._create_y(g, label)

        self.edge_distributions = self._perform_edge_prop_on_graph(adj_mat, y, max_iter=self.max_iter, tol=self.tol)

        return self

    @abc.abstractmethod
    def _perform_edge_prop_on_graph(self, adj_mat: np.ndarray, y: np.ndarray, max_iter=100,
                                    tol=1e-1) -> COO:
        """
        Performs the EdgeProp algorithm on the given graph.
        returns the label distribution (|N|, |N|) matrix with scores between -1, 1 stating the calculated label distribution.
        """
        pass

    @staticmethod
    def _get_classes(g: BaseGraph, label) -> np.ndarray:
        edge_labels = g.get_edge_attributes(label)
        classes = np.unique([label for _, y in edge_labels for label in y])
        classes = classes[classes != BaseModel.NO_LABEL]
        return classes

    @staticmethod
    def _create_y(g, label):
        classes = BaseModel._get_classes(g, label)
        edge_labels = g.get_edge_attributes(label)

        y = np.zeros((g.n_nodes, g.n_nodes, len(classes)))
        for ((u, v), labels) in edge_labels:
            edge = g.node_to_idx[u], g.node_to_idx[v]
            reverse_edge = tuple(reversed(edge))
            for label in labels:
                if label != BaseModel.NO_LABEL:
                    y[edge][label] = 1/len(labels)
                    y[reverse_edge][label] = 1/len(labels)
        return y
