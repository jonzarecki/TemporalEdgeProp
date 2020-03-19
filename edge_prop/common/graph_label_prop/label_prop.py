import networkx as nx
import numpy as np
from sklearn.semi_supervised import LabelPropagation

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
            #TODO: y is weird, should be an node attribute
            y : array_like, shape = [n_samples]
                n_labeled_samples (unlabeled points are marked as -1)
                All unlabeled samples will be transductively assigned labels
        Returns:
            self
        """
        X = - nx.normalized_laplacian_matrix(g, nodelist=sorted(g.nodes)).toarray()
        X = X - np.diag(np.diagonal(X))

        return super(GraphLabelPropagation, self).fit(X, y)

    def predict(self):
        return super(GraphLabelPropagation, self).predict(self.X_)