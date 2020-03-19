import os

import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from tqdm import tqdm

from edge_prop.models.base_model import BaseModel
from edge_prop.models.edge_prop_utils import initialize_distributions

EDGEPROP_BASE_DIR = os.path.dirname(__file__) + "/"


class DenseBasline(BaseModel):
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
    def _perform_edge_prop_on_graph(self, adj_mat: np.ndarray, y: np.ndarray, max_iter=50,
                                    tol=1e-3) -> np.ndarray:
        """
        Performs the EdgeProp algorithm on the given graph.
        returns the label distribution (|N|, |N|) matrix with scores between -1, 1 stating the calculated label distribution.
        """
        A = adj_mat
        Y = y
        _, adj_mat_sparse, _, _ = initialize_distributions(self.graph)
        l_previous = None
        last_Y = np.sum(safe_sparse_dot(adj_mat_sparse, Y), axis=0)
        mat_sum = np.sum(last_Y, axis=-1)[:, np.newaxis]
        mat_sum[mat_sum == 0] = 1
        last_Y = last_Y / mat_sum
        y_nodes = last_Y.copy()
        nodes_exists = y_nodes.sum(axis=-1) > 0

        # last_Y[last_Y != last_Y] = 0

        D = np.sum(A, axis=0)
        D[D == 0] = 1

        with tqdm(range(max_iter), desc='Fitting model', unit='iter') as pbar:

            for n_iter in pbar:
                dif = np.inf if l_previous is None else np.abs(last_Y - l_previous).sum()
                pbar.set_postfix({'dif': dif})
                if n_iter != 0 and dif < tol:  # did not change
                    break  # end the loop, finished
                l_previous = last_Y.copy()

                # B = np.dot(A, last_Y)
                B = safe_sparse_dot(adj_mat_sparse, last_Y)
                last_Y = B / D[:, np.newaxis]
                # save original labels
                last_Y[nodes_exists] = y_nodes[nodes_exists] * self.alpha + last_Y[nodes_exists] * (1 - self.alpha)

        #     last_Y=last_Y[:,np.newaxis,:]*last_Y[np.newaxis,:,:]
        last_Y = last_Y[:, np.newaxis, :] + last_Y[np.newaxis, :, :]
        last_Y[A == 0] = 0
        mat_sum = np.sum(last_Y[A != 0], axis=-1)[:, np.newaxis]
        mat_sum[mat_sum == 0] = 1
        last_Y[A != 0] = last_Y[A != 0] / mat_sum
        # last_Y[last_Y != last_Y] = 0

        # save original labels
        edge_exists = y.sum(axis=-1) > 0
        last_Y[edge_exists] = y[edge_exists] * self.alpha + last_Y[edge_exists] * (1 - self.alpha)

        #
        # label_distributions = y.copy()
        # l_previous = None
        # for n_iter in tqdm(range(max_iter), desc='Fitting model', unit='iter'):
        #     if n_iter != 0 and np.abs(label_distributions - l_previous).sum() < tol:  # did not change
        #         break  # end the loop, finished
        #     l_previous = label_distributions.copy()
        #     B = np.sum(np.dot(adj_mat, label_distributions), axis=0)
        #     D = np.sum(adj_mat, axis=0)
        #
        #     mat = (B[:, np.newaxis, :] + B[np.newaxis, :, :]) / (D[:, np.newaxis] + D[np.newaxis, :])[:, :, np.newaxis]
        #
        #     mat[adj_mat == 0] = 0
        #     mat[adj_mat != 0] = mat[adj_mat != 0] / np.sum(mat[adj_mat != 0], axis=-1, keepdims=True)
        #     mat[mat != mat] = 0
        #
        #     edge_exists = y.sum(axis=-1) > 0
        #     mat[edge_exists] = y[edge_exists] * self.alpha + mat[edge_exists] * (1 - self.alpha)
        #     label_distributions = mat
        # else:
        #     warnings.warn("max_iter was reached without convergence", category=ConvergenceWarning)
        #     n_iter += 1

        return last_Y