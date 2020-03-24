from edge_prop.constants import EDGEPROP_BASE_DIR
import networkx as nx
import os
import matplotlib.pyplot as plt
from scipy import misc, ndimage
import numpy as np
from PIL import Image


def graph2image(probas_mat, adj_mat):
    graph = _create_graph(probas_mat, adj_mat)
    colors = nx.get_edge_attributes(graph, 'proba').values()

    # create matplotlib figure
    figure = _create_graph_figure(graph, colors)
    image = _plot_to_image(figure)
    return image


def _plot_to_image(figure):
    tmp_path = os.path.join(EDGEPROP_BASE_DIR, 'tmp.png')
    plt.savefig(tmp_path, format=format('png'), figure=figure)
    plt.close(figure)
    image = np.array(Image.open(tmp_path)).transpose()
    return image


def _create_graph_figure(graph, colors):
    options = {"edge_color": colors,
               "edge_cmap": plt.cm.seismic,
               "width": 4,
               "with_labels": False}
    figure = plt.figure()
    pos = nx.spring_layout(graph, seed=18)
    nx.draw(graph, pos, **options)
    return figure


def _create_graph(label_distribution, adj_mat) -> nx.Graph:
    rows, cols = np.where(adj_mat > 0)
    edges = list(zip(rows.tolist(), cols.tolist()))
    graph = nx.from_edgelist(edges)
    edge2label = {edge: label_distribution[edge] for edge in edges}
    nx.set_edge_attributes(graph, edge2label, 'proba')
    return graph
