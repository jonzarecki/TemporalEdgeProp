import os
import warnings

import numpy as np
from sparse import COO

from edge_prop.models.sparse_base_model import SparseBaseModel

np.set_printoptions(precision=3)
from sklearn.exceptions import ConvergenceWarning
from tqdm.autonotebook import tqdm

EDGEPROP_BASE_DIR = os.path.dirname(__file__) + "/"


class SparseEdgeProp(SparseBaseModel):
    def _perform_edge_prop_on_graph(self, adj_mat: COO, y: COO, max_iter=100,
                                    tol=1e-1) -> np.ndarray:
        """
        Performs the EdgeProp algorithm on the given graph.
        returns the label distribution (|N|, |N|) matrix with scores between -1, 1 stating the calculated label distribution.
        TODO: inaccurate
        """

        label_distributions = y.copy()
        l_previous = None
        D = adj_mat.sum(axis=0).todense()
        assert D.astype(np.uint16).max() == D.max()
        D = D.astype(np.uint16)
        D[D == 0] = 1
        # mD = (D[:, np.newaxis] + D[np.newaxis, :])[:, :, np.newaxis]

        with tqdm(range(max_iter), desc='Fitting model', unit='iter') as pbar:
            for n_iter in pbar:
                dif = np.inf if l_previous is None else np.abs(label_distributions - l_previous).sum()
                pbar.set_postfix({'dif': dif})
                if n_iter != 0 and dif < tol:  # did not change
                    break  # end the loop, finished
                l_previous = label_distributions.copy()
                # B = adj_mat.dot(label_distributions).sum(axis=1)  # TODO: attention, was axis=0
                B = adj_mat.dot(label_distributions.sum(axis=0))  # same-effect, much faster
                B /= D[:, np.newaxis]  # new, need to check equality
                mat = np.multiply(adj_mat[:, :, np.newaxis], B) + np.multiply(adj_mat[:, np.newaxis, :], B.T).transpose([0, 2, 1])
                # mat /= mD

                mat_sum = np.sum(mat, axis=-1, keepdims=True)
                mat_sum.fill_value = np.float64(1.0)
                mat = mat / mat_sum

                # save original labels
                mat = y * self.alpha + mat * (1 - self.alpha)
                label_distributions = mat
            else:
                warnings.warn("max_iter was reached without convergence", category=ConvergenceWarning)
                n_iter += 1

        return label_distributions