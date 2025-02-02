import gc
import os
import warnings
import numpy as np
from scipy import sparse


np.set_printoptions(precision=3)
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import safe_sparse_dot
from tqdm.autonotebook import tqdm

from edge_prop.models import BaseModel

EDGEPROP_BASE_DIR = os.path.dirname(__file__) + "/"


class DenseEdgeProp(BaseModel):
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
    def _perform_edge_prop_on_graph(self, adj_mat: np.ndarray, y: np.ndarray, max_iter=100,
                                    tol=1e-1) -> np.ndarray:
        """
        Performs the EdgeProp algorithm on the given graph.
        returns the label distribution (|N|, |N|) matrix with scores between -1, 1 stating the calculated label distribution.
        TODO: inaccurate
        """

        label_distributions = y.copy()
        l_previous = None
        D = np.sum(adj_mat, axis=0)
        D[D == 0] = 1
        edge_exists = y.sum(axis=-1) > 0

        # _, adj_mat_sparse, _, _ = initialize_distributions(self.graph)
        adj_mat_sparse = sparse.csr_matrix(adj_mat)


        with tqdm(range(max_iter), desc='Fitting model', unit='iter') as pbar:
            for n_iter in pbar:
                dif = np.inf if l_previous is None else np.abs(label_distributions - l_previous).sum()
                pbar.set_postfix({'dif': dif})
                if n_iter != 0 and dif < tol:  # did not change
                    break  # end the loop, finished
                l_previous = label_distributions.copy()
                # B = np.sum(np.dot(adj_mat, label_distributions), axis=1)  # TODO: attention, was axis=0
                B = np.sum(safe_sparse_dot(adj_mat_sparse, label_distributions), axis=1)  # TODO: attention, was axis=0

                mat = (B[:, np.newaxis, :] + B[np.newaxis, :, :]) / (D[:, np.newaxis] + D[np.newaxis, :])[:, :, np.newaxis]

                mat[adj_mat == 0] = 0
                mat_sum = np.sum(mat[adj_mat != 0], axis=-1, keepdims=True)
                mat_sum[mat_sum == 0] = 1
                mat[adj_mat != 0] = mat[adj_mat != 0] / mat_sum

                # save original labels
                mat[edge_exists] = y[edge_exists] * self.alpha + mat[edge_exists] * (1 - self.alpha)
                label_distributions = mat
            else:
                warnings.warn("max_iter was reached without convergence", category=ConvergenceWarning)
                n_iter += 1

        return label_distributions