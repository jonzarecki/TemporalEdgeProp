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


class DenseBasline(BaseModel):
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
    def _perform_edge_prop_on_graph(self, adj_mat: np.ndarray, y: np.ndarray, max_iter=50,
                                    tol=1e-3) -> np.ndarray:
        """
        Performs the EdgeProp algorithm on the given graph.
        returns the label distribution (|N|, |N|) matrix with scores between -1, 1 stating the calculated label distribution.
        """
        A = adj_mat
        Y = y
        last_Y = np.sum(np.dot(A, Y), axis=0)
        last_Y = last_Y / np.sum(last_Y, axis=-1)[:, np.newaxis]
        last_Y[last_Y != last_Y] = 0

        for i in range(max_iter):
            B = np.dot(A, last_Y)
            D = np.sum(A, axis=0)
            last_Y = B / D[:, np.newaxis]

        #     last_Y=last_Y[:,np.newaxis,:]*last_Y[np.newaxis,:,:]
        last_Y = last_Y[:, np.newaxis, :] + last_Y[np.newaxis, :, :]
        last_Y[A == 0] = 0
        last_Y[A != 0] = last_Y[A != 0] / np.sum(last_Y[A != 0], axis=-1)[:, np.newaxis]
        last_Y[last_Y != last_Y] = 0

        # save original labels
        edge_exists = y.sum(axis=-1) > 0
        last_Y[edge_exists] = y[edge_exists] * self.alpha + last_Y[edge_exists] * (1 - self.alpha)

        #
        # label_distributions = y.copy()
        # l_previous = None
        # for n_iter in tqdm(range(max_iter), desc='Fitting model', unit='iter'):
        #     if n_iter != 0 and np.abs(label_distributions - l_previous).sum() < tol:  # did not change
        #         break  # end the loop, finished
        #     l_previous = label_distributions.copy()
        #     B = np.sum(np.dot(adj_mat, label_distributions), axis=0)
        #     D = np.sum(adj_mat, axis=0)
        #
        #     mat = (B[:, np.newaxis, :] + B[np.newaxis, :, :]) / (D[:, np.newaxis] + D[np.newaxis, :])[:, :, np.newaxis]
        #
        #     mat[adj_mat == 0] = 0
        #     mat[adj_mat != 0] = mat[adj_mat != 0] / np.sum(mat[adj_mat != 0], axis=-1, keepdims=True)
        #     mat[mat != mat] = 0
        #
        #     edge_exists = y.sum(axis=-1) > 0
        #     mat[edge_exists] = y[edge_exists] * self.alpha + mat[edge_exists] * (1 - self.alpha)
        #     label_distributions = mat
        # else:
        #     warnings.warn("max_iter was reached without convergence", category=ConvergenceWarning)
        #     n_iter += 1

        return last_Y