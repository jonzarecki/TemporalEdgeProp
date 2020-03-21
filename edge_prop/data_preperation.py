from copy import deepcopy
from typing import Tuple

from edge_prop.graph_wrappers.base_graph import BaseGraph
import numpy as np
import networkx as nx

from edge_prop.graph_wrappers.binary_labeled_graph import BinaryLabeledGraph


def remove_labels(graph: BaseGraph, label, keep_labels_precent: float = 0.5) -> Tuple[
    BinaryLabeledGraph, np.ndarray, np.ndarray]:
    keep_label_indices = np.random.choice(range(graph.n_edges), size=int(keep_labels_precent * graph.n_edges),
                                          replace=False)
    label_mask = np.zeros(graph.n_edges, dtype=np.int)
    label_mask[keep_label_indices] = 1

    y_true = graph.get_edge_attributes(label)

    new_labels = {edge: label * label_mask[i] for i, (edge, label) in enumerate(graph.edge_labels)}

    new_graph = graph.graph_nx.copy()
    nx.set_edge_attributes(new_graph, new_labels, graph.y_attr)
    binary_graph = BinaryLabeledGraph(new_graph, graph.y_attr)

    return binary_graph, y_true, label_mask
