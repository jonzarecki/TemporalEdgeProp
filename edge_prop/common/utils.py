from typing import Tuple
import gc
import numpy as np
import networkx as nx
import scipy
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import safe_sparse_dot
import warnings

from edge_prop.graph_wrappers import TemporalGraph, BinaryLabeledGraph
from edge_prop.constants import DECAYED_TIME_WEIGHT

def perform_edge_prop_on_graph(graph_matrix: sparse.csr_matrix, label_distributions: sparse.csr_matrix,
                               edge_exists: sparse.csr_matrix, labeled: sparse.dok_matrix,
                               y_static: sparse.csr_matrix, max_iter=50, tol=1e-3) -> sparse.csr_matrix:
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


def initialize_distributions(g: BinaryLabeledGraph) -> Tuple[sparse.csr_matrix, sparse.csr_matrix,
                                                             sparse.dok_matrix, sparse.csr_matrix]:
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
    label_distributions = sparse.lil_matrix((g.n_nodes, g.n_nodes), dtype=float)
    edge_exists = sparse.dok_matrix((g.n_nodes, g.n_nodes), dtype=bool)
    for ((u, v), label) in g.edge_labels:
        edge = g.node_to_idx[u], g.node_to_idx[v]
        reverse_edge = tuple(reversed(edge))
        label_distributions[edge] = label
        label_distributions[reverse_edge] = label
        edge_exists[edge] = 1
        edge_exists[reverse_edge] = 1
    label_distributions = label_distributions.tocsr()
    edge_exists = edge_exists.tocsr()
    y_static = sparse.dok_matrix(label_distributions)
    labeled = (y_static != 0)

    return label_distributions, edge_exists, labeled, y_static


def bulk_calc_temporal_edge_prop(items_in_test: list, agg_g: BinaryLabeledGraph, g_edge_times_dict: dict,
                                 g_edge_idx_to_agg_edge: dict) -> list:
    """
    Used to calculate the TEP alg' for a bulk set of test edges
    :param items_in_test: The items we want to calculate TEP on in (edge_idx, edge) format
    :param agg_g: The aggregated graph object as calculated by build_aggregated_graph()
    :param g_edge_times_dict: dict that maps between edges and their time attribute
    :param g_edge_idx_to_agg_edge: dict that maps between an edge to it's agg edge (in agg_g)
    :return: the predictions from the TEP on the test edges in (edge_idx, pred) format
    """
    results = []
    # initialize distributions
    agg_label_distributions, agg_edge_exists, agg_labeled, agg_y_static = initialize_distributions(agg_g)

    for i, edge in items_in_test:
        TemporalGraph.build_decayed_edge_weights(agg_g, reference_time=g_edge_times_dict[edge],
                                                 dec_time_attr=DECAYED_TIME_WEIGHT)

        # actual graph construction
        agg_graph_matrix = - nx.normalized_laplacian_matrix(agg_g.g_nx, nodelist=agg_g.node_order,
                                                            weight=DECAYED_TIME_WEIGHT)
        agg_graph_matrix.setdiag(0)

        # run edge-prop
        agg_label_distributions = agg_y_static.tocsr(copy=True)  # don't use same label distribution for diff edges
        agg_label_distributions = perform_edge_prop_on_graph(agg_graph_matrix, agg_label_distributions,
                                                             agg_edge_exists, agg_labeled, agg_y_static)

        inst_pred = agg_label_distributions[tuple(agg_g.node_to_idx[node] for node in g_edge_idx_to_agg_edge[i])]
        results.append((i, inst_pred))

    return results
