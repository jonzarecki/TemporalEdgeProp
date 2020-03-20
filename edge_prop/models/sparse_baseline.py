import os

import numpy as np
from scipy import sparse
from sklearn.utils.extmath import safe_sparse_dot
from tqdm import tqdm

from edge_prop.models.base_model import BaseModel
from edge_prop.models.edge_prop_utils import initialize_distributions
from edge_prop.models.sparse_base_model import SparseBaseModel

EDGEPROP_BASE_DIR = os.path.dirname(__file__) + "/"


class SparseBasline(SparseBaseModel):
    def _perform_edge_prop_on_graph(self, adj_mat: np.ndarray, y: np.ndarray, max_iter=50,
                                    tol=1e-3) -> np.ndarray:
        """
        Performs the EdgeProp algorithm on the given graph.
        returns the label distribution (|N|, |N|) matrix with scores between -1, 1 stating the calculated label distribution.
        """
        # A = adj_mat.todense()
        Y = y

        l_previous = None

        last_Y = adj_mat.dot(Y).sum(axis=1).todense()  # TODO: axis 0 or 1 ?
        mat_sum = np.sum(last_Y, axis=-1)[:, np.newaxis]
        mat_sum[mat_sum == 0] = 1
        last_Y = last_Y / mat_sum

        # D = np.sum(adj_mat, axis=0).todense()
        # D[D == 0] = 1
        # mD = D[:, np.newaxis]

        D = np.sum(adj_mat, axis=0).todense()
        D[D == 0] = 1
        mD = D[:, np.newaxis]

        with tqdm(range(max_iter), desc='Fitting model', unit='iter') as pbar:

            for n_iter in pbar:
                dif = np.inf if l_previous is None else np.abs(last_Y - l_previous).sum()
                pbar.set_postfix({'dif': dif})
                if n_iter != 0 and dif < tol:  # did not change
                    break  # end the loop, finished
                l_previous = last_Y.copy()

                B = adj_mat.dot(last_Y)
                last_Y = B / mD

        from sparse import COO
        last_Y = COO(last_Y)
        last_Y_edges = np.multiply(adj_mat[:, :, np.newaxis], last_Y[:, np.newaxis, :]) + \
                       np.multiply(adj_mat[:, :, np.newaxis], last_Y[np.newaxis, :, :]).transpose([1, 0, 2])

        mat_sum = np.sum(last_Y_edges, axis=-1, keepdims=True)
        mat_sum.fill_value = np.float64(1.0)
        last_Y_edges = last_Y_edges / mat_sum

        # last_Y = last_Y[:, np.newaxis, :] + last_Y[np.newaxis, :, :]
        # last_Y[A == 0] = 0
        # mat_sum = np.sum(last_Y[A != 0], axis=-1)[:, np.newaxis]
        # mat_sum[mat_sum == 0] = 1
        # last_Y[A != 0] = last_Y[A != 0] / mat_sum
        # last_Y[last_Y != last_Y] = 0

        # # save original labels
        # edge_exists = y.sum(axis=-1) > 0
        # last_Y[edge_exists] = y[edge_exists] * self.alpha + last_Y[edge_exists] * (1 - self.alpha)
        #


        return last_Y_edges
