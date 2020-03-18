from abc import ABCMeta

import six
from sklearn.base import BaseEstimator, ClassifierMixin

from edge_prop.graph_wrappers import BaseGraph


class BaseModel(six.with_metaclass(ABCMeta), BaseEstimator, ClassifierMixin):
    """
    Edge Propgation

    EXPECTS non-multi edge graphs
    Parameters
    ------------
    y_attr: The edge attr containing the label

    max_iter: integer
            Change maximum number of iterations allowed

    tol: float
            Convergence tolerance: threshold to consider the system at a steady state

    """
    _variant = 'propagation'

    def __init__(self, y_attr: str, max_iter=50, tol=1e-3):
        self.y_attr = y_attr
        self.max_iter = max_iter
        self.tol = tol
        self.edge_distributions = None  # keeps results after run

    def predict(self):
        """
        Predict labels across all edges

        Parameters
        ----------

        Returns
        ------
        y : array_like, shape = [n_edges]
            Predictions for entire graph

        """
        raise NotImplementedError

    def fit(self, g: BaseGraph):
        """
        Fit the graph

        :param g: The graph


        Returns
        -------
        self : returns a pointer to self
        """
        raise NotImplementedError
