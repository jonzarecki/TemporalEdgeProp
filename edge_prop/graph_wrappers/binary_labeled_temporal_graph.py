from typing import Tuple
import numpy as np
import networkx as nx

from edge_prop.graph_wrappers import BinaryLabeledGraph
from edge_prop.graph_wrappers import TemporalGraph
from edge_prop.constants import AGG_TIMES_LIST_ATTR


class BinaryLabeledTemporalGraph(BinaryLabeledGraph, TemporalGraph):
    """
        Similar to BinaryLabeledGraph but also keeps information on edge times

        time_attr - the string attribute keeping the edge time (in datetime.datetime format)
        edge_time_dict - maps between edge and it's time
        edge_timestamp_in_order - list containing (in edge_order's order) all the edge's time in timestamp format

        edge_label_dict - dict that maps between the edge to it's tag
        edge_labels - list (in edge_order's oder) keeping (e, e's tag)

        classes - list keeping the unique tags (without 0 which is unlabeled)

        """
    def __init__(self, g: nx.Graph, y_attr: str, time_attr: str, **kwargs):
        super(BinaryLabeledTemporalGraph, self).__init__(g=g, y_attr=y_attr, time_attr=time_attr, **kwargs)

    def build_aggregated_graph(self) -> Tuple[BinaryLabeledGraph, dict]:
        """
        Aggregates this graphs multi-edges while keeping their time attributes as a list
        CURRENTLY LABELS ALL MULTI-EDGES AS 1 (IMPORTANT)
        """
        # dictionaries that map between the aggregated graphs and the original one
        agg_e_maps_to = dict()  # was UnorderedDict (?)
        edge_time_weights = dict()
        agg_edge_labels = dict()
        # build aggregated graph
        agg_g = nx.Graph()
        for i, e in enumerate(self.edge_order):
            agg_e = (e[0], e[1])
            if not agg_g.has_edge(e[0], e[1]):
                agg_g.add_edge(*agg_e)
                edge_time_weights[agg_e] = []
                agg_e_maps_to[agg_e] = []
                agg_edge_labels[agg_e] = -100
            edge_time_weights[agg_e].append(self.edge_times_dict[e].timestamp())
            agg_e_maps_to[agg_e].append(i)
            # assumes 1 is better than -1, can keep distribution of all labels for later
            # IMPORTANT
            agg_edge_labels[agg_e] = max(agg_edge_labels[agg_e], self.edge_label_dict[e])

        g_edge_idx_to_agg_edge = {}
        for agg_g_e in agg_g.edges:
            for g_edge_idx in agg_e_maps_to[agg_g_e]:
                g_edge_idx_to_agg_edge[g_edge_idx] = agg_g_e

        # add attributes to edges
        edge_time_weights = {e: np.array(t_list) for (e, t_list) in edge_time_weights.items()}
        nx.set_edge_attributes(agg_g, edge_time_weights, AGG_TIMES_LIST_ATTR)
        nx.set_edge_attributes(agg_g, agg_edge_labels, self.y_attr)

        # wrap with LabeledGraph
        agg_g_l = BinaryLabeledGraph(agg_g, self.y_attr)
        agg_g_l.agg_e_maps_to = agg_e_maps_to
        agg_g_l.edge_timestamp_in_order = self.edge_timestamp_in_order

        return agg_g_l, g_edge_idx_to_agg_edge