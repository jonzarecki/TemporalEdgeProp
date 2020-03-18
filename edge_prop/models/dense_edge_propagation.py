import gc
import os
import warnings
from abc import ABCMeta
from typing import Tuple, List

import scipy

import networkx as nx
import numpy as np
import six
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import safe_sparse_dot
import torch
from tqdm.autonotebook import tqdm

from edge_prop.graph_wrappers import BinaryLabeledGraph
from edge_prop.models.base_model import BaseModel

EDGEPROP_BASE_DIR = os.path.dirname(__file__) + "/"


class DenseEdgeProp(BaseModel):
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

    def __init__(self, y_attr: str, max_iter: int = 50, tol: float = 1e-3):
        super().__init__(y_attr)
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
        return self.edge_prop_results.argmax(axis=-1)

    def fit(self, g: BinaryLabeledGraph):
        """
        Uses the laplacian matrix to act as affinity matrix for the label-prop alg'
        :param g: The graph


        Returns
        -------
        self : returns a pointer to self
        """
        self._classes = self._get_classes(g)
        adj_mat = np.asarray(nx.adjacency_matrix(g.graph_nx).todense())
        y = self._create_y(g)

        label_distributions = self._perform_edge_prop_on_graph(adj_mat, y, max_iter=self.max_iter, tol=self.tol)

        # set the results
        self.edge_prop_results = label_distributions
        # self.edge_prop_results = np.zeros(g.n_edges)  # will hold the results
        # for i, (u, v) in enumerate(g.edge_order):
        #     edge_idxs = g.node_to_idx[u], g.node_to_idx[v]
        #     self.edge_prop_results[i] = label_distributions[edge_idxs]
        return self

    def _get_classes(self, g: BinaryLabeledGraph) -> np.ndarray:
        classes = np.unique([label for (edge, label) in g.edge_labels])
        classes = classes[classes != self.NO_LABEL]
        return classes

    def _perform_edge_prop_on_graph(self, adj_mat: np.ndarray, y: np.ndarray, max_iter=50,
                                    tol=1e-3) -> np.ndarray:
        """
        Performs the EdgeProp algorithm on the given graph.
        returns the label distribution (|N|, |N|) matrix with scores between -1, 1 stating the calculated label distribution.
        """

        label_distributions = y.copy()
        l_previous = None
        for n_iter in tqdm(range(max_iter), desc='Fitting model', unit='iter'):
            if n_iter != 0 and np.abs(label_distributions - l_previous).sum() < tol:  # did not change
                break  # end the loop, finished
            l_previous = label_distributions.copy()
            B = np.sum(np.dot(adj_mat, label_distributions), axis=0)
            D = np.sum(adj_mat, axis=0)

            mat = (B[:, np.newaxis, :] + B[np.newaxis, :, :]) / (D[:, np.newaxis] + D[np.newaxis, :])[:, :, np.newaxis]

            mat[adj_mat == 0] = 0
            mat[adj_mat != 0] = mat[adj_mat != 0] / np.sum(mat[adj_mat != 0], axis=-1, keepdims=True)
            mat[mat != mat] = 0

            label_distributions = mat
        else:
            warnings.warn("max_iter was reached without convergence", category=ConvergenceWarning)
            n_iter += 1

        return label_distributions

    def _create_y(self, g):
        y = np.zeros((g.n_nodes, g.n_nodes, len(self._classes)))
        for ((u, v), label) in g.edge_labels:
            edge = g.node_to_idx[u], g.node_to_idx[v]
            reverse_edge = tuple(reversed(edge))
            if label != self.NO_LABEL:
                y[edge][label] = 1
                y[reverse_edge][label] = 1
        return y
