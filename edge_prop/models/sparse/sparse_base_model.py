
import numpy as np
from sparse import DOK, COO

from edge_prop.constants import NO_LABEL
from edge_prop.models import BaseModel


class SparseBaseModel(BaseModel):
    """
    Edge Propgation

    EXPECTS non-multi edge graphs
    Parameters
    ------------
    max_iter: integer
            Change maximum number of iterations allowed

    tol: float
            Convergence tolerance: threshold to consider the system at a steady state

    """

    def __init__(self, max_iter: int = 50, tol: float = 1e-5, alpha: float = 1):
        super(SparseBaseModel, self).__init__(max_iter, tol, alpha)
        self.sparse = True


    @staticmethod
    def _create_y(g, label):
        classes = BaseModel._get_classes(g, label)
        edge_labels = g.get_edge_attributes(label)

        values = {}
        for ((u, v), labels) in edge_labels:
            edge = g.node_to_idx[u], g.node_to_idx[v]
            for label in labels:
                if label != NO_LABEL:
                    values[(edge[0], edge[1], label)] = 1/len(labels)
                    values[(edge[1], edge[0], label)] = 1/len(labels)

        y = DOK((g.n_nodes, g.n_nodes, len(classes)), values, dtype=np.float32)
        return y.to_coo()