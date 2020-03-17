import networkx as nx
import numpy as np

from sklearn.semi_supervised.label_propagation import BaseLabelPropagation


class GraphLabelPropagation(BaseLabelPropagation):
    _variant = 'propagation'

    def _build_graph(self):
        """
        here we assume self.X already hold our affinity matyrix as calculated by networkx
        """
        return self.X_

    def fit(self, g: nx.Graph, y):
        """
        Uses the laplacian matrix to acy as affinity matrix for the label-prop algorithm
        Args:
            g: The graph in nx format
            y: the edge label we expect

        Returns:
            self
        """
        X = - nx.normalized_laplacian_matrix(g, nodelist=sorted(g.nodes)).toarray()
        X = X - np.diag(np.diagonal(X))

        return super(GraphLabelPropagation, self).fit(X, y)
