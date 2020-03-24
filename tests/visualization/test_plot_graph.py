from unittest import TestCase

import sparse

from edge_prop.visualization.adj_mat_to_image import graph2image
import matplotlib.pyplot as plt

class TestPlot_graph(TestCase):
    def test_plot_graph(self):
        coords, data = [[1,2,0,0],[0,0,1,2]], [1,1,1,1]
        adj_mat = sparse.COO(coords, data)
        probas = [0.0, 1, 0, 1]
        probas_mat = sparse.COO(coords, probas)

        image = graph2image(probas_mat, adj_mat)

        plt.imshow(image.T)
        plt.show()
