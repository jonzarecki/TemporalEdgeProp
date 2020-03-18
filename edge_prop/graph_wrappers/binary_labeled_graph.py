import networkx as nx
import numpy as np

from edge_prop.graph_wrappers import BaseGraph


class BinaryLabeledGraph(BaseGraph):
    """
    Utility class for keeping useful information on EDGE labeled graphs
    IMPORTANT: tag 0 is unlabeled

    edge_label_dict - dict that maps between the edge to it's tag
    edge_labels - list (in edge_order's oder) keeping (e, e's tag)

    classes - list keeping the unique tags (without 0 which is unlabeled)

    """

    def __init__(self, g: nx.Graph, y_attr: str, **kwargs):
        super().__init__(g, **kwargs)
        self.y_attr = y_attr

        self.edge_label_dict = nx.get_edge_attributes(g, y_attr)
        self.edge_labels = [(e, self.edge_label_dict[e]) for e in self.edge_order]
        # label construction
        # construct a categorical distribution for classification only
        classes = np.unique([y for (e, y) in self.edge_labels])
        classes = classes[classes != 0]