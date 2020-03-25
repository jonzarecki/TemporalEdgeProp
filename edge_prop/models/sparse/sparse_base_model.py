
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

    def __init__(self, max_iter: int = 50, tol: float = 1e-3, alpha: float = 1, tb_exp_name:str=None):
        super(SparseBaseModel, self).__init__(max_iter, tol, alpha, tb_exp_name)
        self.sparse = True

    def _create_y(self, g, label):
        classes = self._get_classes(g, label)
        self.classes = classes
        lbl2idx = {l:i for i, l in enumerate(self.classes)}
        edge_labels = g.get_edge_attributes_ordered(label)

        values = {}
        for ((u, v), labels) in edge_labels:
            edge = g.node_to_idx[u], g.node_to_idx[v]
            for label in labels:
                if label != NO_LABEL:
                    lbl_idx = lbl2idx[label]
                    values[(edge[0], edge[1], lbl_idx)] = 1/len(labels)
                    values[(edge[1], edge[0], lbl_idx)] = 1/len(labels)

        y = DOK((g.n_nodes, g.n_nodes, len(classes)), values, dtype=np.float16)  # reduced to save mem
        return y.to_coo()