import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

import networkx as nx
import numpy as np

from edge_classification.graph_wrappers.base_graph import BaseGraph


class TemporalGraph(BaseGraph):
    """
        A graph that keeps information on edge time
        
        time_attr - the string attribute keeping the edge time (in datetime.datetime format)
        edge_time_dict - maps between edge and it's time
        edge_timestamp_in_order - list containing (in edge_order's order) all the edge's time in timestamp format
    
    """
    
    def __init__(self, g: nx.Graph, time_attr: str, **kwargs):
        super(TemporalGraph, self).__init__(g, **kwargs)

        self.time_attr = time_attr
        self.edge_times_dict: Dict[Any, datetime] = nx.get_edge_attributes(g, self.time_attr)
        self.edge_timestamp_in_order = np.array([self.edge_times_dict[e].timestamp() for e in self.edge_order])

    @staticmethod
    def decay_function(times_list: np.ndarray, measure_from: float) -> np.ndarray:
        """
        Calculates the decayed time function for all times in $timeslist in comparison with measure_from timestamp
        Args:
            times_list: list of timestamps in np array
            measure_from: timestamp to start measuring from

        Returns:
            Decayed scores for all times in $times_list
        """
        alpha = 1.
        beta = -0.0005
        return alpha * np.exp(beta * np.abs(measure_from - times_list) / (60**2))

    @staticmethod
    def build_decayed_edge_weights(agg_g: BaseGraph, reference_time: datetime, dec_time_attr: str):
        """
        Adds decayed edge weights to the already aggregated agg_g in comparison with $reference_time
        Args:
            agg_g:
            reference_time:
            dec_time_attr:

        Returns:

        """
        ref_timestamp = reference_time.timestamp()
        edge_weight_dict = {}
        agg_e_maps_to = agg_g.agg_e_maps_to  # we put it there in build_aggregated_graph()
        decayed_edge_times_in_order = TemporalGraph.decay_function(agg_g.edge_timestamp_in_order, ref_timestamp)
        for agg_e, g_idxs in agg_e_maps_to.to_dict().items():
            # sum of decayed weights IMPORTANT ALGORITHMIC CHOICE
            edge_weight_dict[agg_e] = decayed_edge_times_in_order[g_idxs].sum()
        nx.set_edge_attributes(agg_g.g_nx, edge_weight_dict, dec_time_attr)

    def split_into_time_chunks(self, delta=timedelta(days=7), verbose=True) -> List[nx.MultiDiGraph]:
        """
        Splits this graph into different chunks with a time chunks size of $delta
        Args:
            delta:
            verbose:

        Returns:
            List of separated graphs
        """
        max_time = max(self.edge_times_dict.values())
        min_time = min(self.edge_times_dict.values())
        graphs = []
        n_chunks = np.math.ceil((max_time - min_time) / delta)
        logging.info(f"split graph into: maxtime: {max_time} mintime: {min_time}")

        for i in range(1, n_chunks+1):
            chunk_st, chunk_end = min_time + (i - 1) * delta, min_time + i * delta
            logging.info(f"{i}: st - {chunk_st}  end - {chunk_end}")
            g = self.g_nx.edge_subgraph(filter(lambda e: chunk_st < self.edge_times_dict[e] < chunk_end, self.edge_order)).copy()
            graphs.append(g)

        # same number of nodes, different edges
        for g in graphs:
            for g2 in graphs:
                if g != g2:
                    g.add_node(g2)
        return graphs