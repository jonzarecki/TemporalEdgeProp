import logging
import os
import warnings

import numpy as np
from sparse import COO

from edge_prop.models import SparseBaseModel

np.set_printoptions(precision=3)
from sklearn.exceptions import ConvergenceWarning
from tqdm.autonotebook import tqdm

EDGEPROP_BASE_DIR = os.path.dirname(__file__) + "/"


class SparseEdgeProp(SparseBaseModel):
    def _perform_edge_prop_on_graph(self, adj_mat: COO, y: COO, max_iter=100,
                                    tol=1e-3) -> np.ndarray:
        """
        Performs the EdgeProp algorithm on the given graph.
        returns the label distribution (|N|, |N|) matrix with scores between -1, 1 stating the calculated label distribution.
        TODO: inaccurate
        """
        logging.info("init calcs")
        label_distributions = y.copy()
        l_previous = None
        not_original_edge = y.sum(axis=-1, keepdims=True) == 0
        D = adj_mat.sum(axis=0).todense()
        assert D.astype(np.uint16).max() == D.max()
        D = D.astype(np.uint16)
        D[D == 0] = 1
        # mD = (D[:, np.newaxis] + D[np.newaxis, :])[:, :, np.newaxis]
        A = adj_mat.tocsr().astype(dtype=np.bool)
        adj_3d = adj_mat[:, :, np.newaxis]
        with tqdm(range(max_iter), desc='Fitting model', unit='iter') as pbar:
            for n_iter in pbar:
                logging.info("check cond")
                dif = np.inf if l_previous is None else np.abs((label_distributions - l_previous).data).sum()
                pbar.set_postfix({'dif': dif})
                if n_iter != 0 and dif < tol:  # did not change
                    break  # end the loop, finished
                l_previous = label_distributions
                # B = adj_mat.dot(label_distributions).sum(axis=1)  # TODO: attention, was axis=0
                logging.info("dot prod")
                B = A.dot(label_distributions.sum(axis=1).todense())  # same-effect, much faster
                B /= D[:, np.newaxis]  # new, need to check equality
                B = COO(B)
                logging.info("expand to 3d")
                mat = adj_3d * B[:, np.newaxis, :] + \
                      adj_3d * B[np.newaxis, :, :]  # each mul takes 10s in aminar_s
                # mat /= mD

                logging.info("norm mat")
                mat_sum = np.sum(mat, axis=-1, keepdims=True)
                mat_sum.fill_value = np.float64(1.0)
                mat = mat / mat_sum  # 20s

                logging.info("calc alpha changes")
                # save original labels
                original_edges_labels = y * self.alpha + (1 - not_original_edge) * mat * (1 - self.alpha)  # 30s
                label_distributions = original_edges_labels + not_original_edge * mat
            else:
                warnings.warn("max_iter was reached without convergence", category=ConvergenceWarning)
                n_iter += 1

        return label_distributions