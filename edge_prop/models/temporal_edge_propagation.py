import logging
import multiprocessing
import os
import pickle
from functools import partial
from typing import Tuple, List

import networkx as nx
import numpy as np
from sklearn.preprocessing import minmax_scale

from edge_prop.graph_wrappers import BinaryLabeledTemporalGraph, TemporalGraph
from edge_prop.models import GraphEdgePropagation
from edge_prop.common.utils import initialize_distributions, perform_edge_prop_on_graph, bulk_calc_temporal_edge_prop

from edge_prop.constants import DECAYED_TIME_WEIGHT, EDGEPROP_BASE_DIR


class TemporalGraphEdgePropagation(GraphEdgePropagation):
    """
    Temporal Edge Propagation

    Parameters
    ------------
    y_attr: The edge attr containing the label
    time_attr: The edge attr containing the time

    max_iter: integer
        Change maximum number of iterations allowed

    tol: float
        Convergence tolerance threshold to consider the system at a steady state

    """
    _variant = "propagation"

    def __init__(self, y_attr: str = 'label', in_test: str = 'in_test', time_attr: str = 'time', max_iter=50, tol=1e-7,
                 on_future=False, is_parallel=False, proc_num=10, chunk_size=100):
        super(TemporalGraphEdgePropagation, self).__init__(y_attr, max_iter, tol)
        self.chunk_size = chunk_size
        self.proc_num = proc_num
        self.on_future = on_future
        self.is_parallel = is_parallel
        self.time_attr = time_attr
        self.in_test = in_test

    def predict(self, X=None):
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

    def predict_proba(self, X=None):
        probas_1_cls = minmax_scale(self.edge_prop_results)
        return np.array([(1 - p, p) for p in probas_1_cls])

    def fit(self, g: BinaryLabeledTemporalGraph):
        """
        Uses the laplacian matrix to act as affinity matrix for the edge-prop alg'.
        For each edge we predict we use it's time as reference time and use exponential decay from that time
        to weigh the other edges.
        Uses _fit_for_past and _fit_for_future to actually predict according the params in __init__

        :param g: The graph
        :return: returns a pointer to self
        """
        path = EDGEPROP_BASE_DIR + "models/temporal_edge_prop/" + str(
            hash(tuple([itm[1] for itm in g.edge_labels]))) + ".bin"
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                self.edge_prop_results = pickle.load(f)
            return self
        logging.info(f"Temporal Edge Prop - {path}")
        if not self.on_future:
            results_combined = self._fit_for_past(g)
        else:  # on future
            results_combined = self._fit_for_future(g)

        self.edge_prop_results = [pred for _, pred in sorted(results_combined, key=lambda r: r[0])]

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.edge_prop_results, f)

        return self

    def _fit_for_past(self, g: BinaryLabeledTemporalGraph) -> List[Tuple[int, int]]:
        """
        Uses the laplacian matrix to act as affinity matrix for the edge-prop alg'.
        For each edge we predict we use it's time as reference time and use exponential decay from that time
        to weigh the other edges.
        Updates self.edge_prop_results with the results as a list of (edge_idx, prediction)
        runs edge-prop MULTIPLE times for each test edge

        :param g: The graph
        :return: the list of predictions from in (edge_idx, pred) format
        """
        items_in_test = [(i, e) for (i, e) in enumerate(g.edge_order) if g.g_nx.get_edge_data(*e)[self.in_test]]
        agg_g, g_edge_idx_to_agg_edge = g.build_aggregated_graph()
        if self.is_parallel:
            func = partial(bulk_calc_temporal_edge_prop, agg_g=agg_g, g_edge_times_dict=g.edge_times_dict,
                           g_edge_idx_to_agg_edge=g_edge_idx_to_agg_edge)
            with multiprocessing.Pool(processes=self.proc_num) as p:
                results = p.map(func, [items_in_test[st_i: (st_i + self.chunk_size)] for st_i
                                       in range(0, len(items_in_test) + self.chunk_size, self.chunk_size)])
            results_combined = sum(results, list())
        else:  # serial
            results_combined = bulk_calc_temporal_edge_prop(items_in_test, agg_g, g.edge_times_dict,
                                                            g_edge_idx_to_agg_edge)
        return results_combined

    def _fit_for_future(self, g: BinaryLabeledTemporalGraph):
        """
        Uses the laplacian matrix to act as affinity matrix for the edge-prop alg'.
        For each edge we predict we use it's time as reference time as
        the MAXIMUM time in the graph (meaning we predict for the future).
        Updates self.edge_prop_results with the results as a list of (edge_idx, prediction)
        runs edge-prop ONCE

        :param g: The graph
        :return: the list of predictions from in (edge_idx, pred) format
        """
        max_edge_time = max(g.edge_times_dict.values())
        agg_g, g_edge_idx_to_agg_edge = g.build_aggregated_graph()
        # initialize distributions
        agg_label_distributions, agg_edge_exists, agg_labeled, agg_y_static = initialize_distributions(agg_g)

        TemporalGraph.build_decayed_edge_weights(agg_g, reference_time=max_edge_time, dec_time_attr=DECAYED_TIME_WEIGHT)

        # actual graph construction
        agg_graph_matrix = - nx.normalized_laplacian_matrix(agg_g.g_nx, nodelist=agg_g.node_order,
                                                            weight=DECAYED_TIME_WEIGHT)
        agg_graph_matrix.setdiag(0)

        # run edge-prop for "future"
        agg_label_distributions = perform_edge_prop_on_graph(agg_graph_matrix, agg_label_distributions,
                                                             agg_edge_exists, agg_labeled, agg_y_static)

        # set the results
        results = []
        for i, e in enumerate(g.edge_order):
            inst_pred = agg_label_distributions[tuple(agg_g.node_to_idx[node] for node in g_edge_idx_to_agg_edge[i])]
            results.append((i, inst_pred))

        return results
