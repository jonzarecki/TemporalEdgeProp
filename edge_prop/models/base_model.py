import abc
import io
import logging
import os
import warnings
from abc import ABCMeta

import six
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sparse import DOK, COO
from torch.utils.tensorboard import SummaryWriter
from matplotlib.pyplot import cm

from edge_prop.graph_wrappers import BaseGraph
from edge_prop.constants import NO_LABEL, EDGEPROP_BASE_DIR, TENSORBOARD_DIR
from edge_prop.common.metrics import get_all_metrics
from edge_prop.visualization.adj_mat_to_image import graph2image


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

    def __init__(self, max_iter: int = 50, tol: float = 1e-5, alpha: float = 1, tb_exp_name: str = None):
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.tb_exp_name = tb_exp_name

        self.verbose = False
        if tb_exp_name is not None:
            path = os.path.join(TENSORBOARD_DIR, tb_exp_name)  # , str(datetime.datetime.now()))
            self.writer = SummaryWriter(path)
            self.verbose = True

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
            assert np.allclose(self.get_edge_distributions(u, v),
                               self.get_edge_distributions(v, u)), "graphs are undirectional, shouldn't happen"
            if len(dist[dist == dist.max()]) > 1:
                warnings.warn(f"edge {(u, v)} doesn't have a definitive max: {dist}", category=RuntimeWarning)
            results.append(self.classes[dist.argmax()])  # returned index and not the class
        results = np.array(results, dtype=np.int)
        # results = np.ones_like(self.edge_distributions[:, :, 0]) * NO_LABEL
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

    def fit(self, g: BaseGraph, label, val={}):
        """
        Uses the laplacian matrix to act as affinity matrix for the label-prop alg'
        :param g: The graph


        Returns
        -------
        self : returns a pointer to self
        """
        self.graph = g
        self.val = val
        self.adj_mat = g.adjacency_matrix(sparse=self.sparse)
        self.y = self._create_y(g, label)
        self.num_classes = self.y.shape[-1]

        self.edge_distributions = self._perform_edge_prop_on_graph(self.adj_mat, self.y, max_iter=self.max_iter,
                                                                   tol=self.tol)

        return self

    @abc.abstractmethod
    def _perform_edge_prop_on_graph(self, adj_mat: np.ndarray, y: np.ndarray, max_iter=100, tol=1e-1) -> COO:
        """
        Performs the EdgeProp algorithm on the given graph.
        returns the label distribution (|N|, |N|) matrix with scores between -1, 1 stating the calculated label distribution.
        """
        pass

    def _get_classes(self, g: BaseGraph, label) -> np.ndarray:
        edge_labels = g.get_edge_attributes_ordered(label)
        classes = np.unique([label for _, y in edge_labels for label in y])
        classes = classes[classes != NO_LABEL]
        return classes

    def _create_y(self, g, label):
        classes = self._get_classes(g, label)
        self.classes = classes
        edge_labels = g.get_edge_attributes_ordered(label)

        y = np.zeros((g.n_nodes, g.n_nodes, len(classes)))
        for ((u, v), labels) in edge_labels:
            edge = g.node_to_idx[u], g.node_to_idx[v]
            reverse_edge = tuple(reversed(edge))
            for label in labels:
                if label != NO_LABEL:
                    y[edge][label] = 1 / len(labels)
                    y[reverse_edge][label] = 1 / len(labels)
        return y

    def write_evaluation_to_tensorboard(self, global_step):
        # if self.num_classes > 2:
        #     graph_image = graph2image(self.edge_distributions.argmax(axis=-1), self.adj_mat, color_map=cm.gist_ncar)
        # else:
        #     graph_image = graph2image(self.edge_distributions[:, :, -1], self.adj_mat, color_map=cm.seismic)
        # self.writer.add_image("Graph", graph_image, global_step=global_step)

        for val_name, (val_indices, y_val) in self.val.items():
            y_pred = self.predict_proba(val_indices)
            metrics = get_all_metrics(y_pred, y_val)
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(f'{val_name}/{metric_name}', metric_value, global_step=global_step)

            pred_classes = np.argmax(y_pred, axis=-1)
            self.writer.add_histogram(f'{val_name}/predicted_class', pred_classes, global_step=global_step)

        self.writer.flush()
