import logging
import os
import warnings

import numpy as np
from sparse import COO

from edge_prop.constants import TENSORBOARD_DIR
from edge_prop.models import SparseBaseModel
from torch.utils.tensorboard import SummaryWriter

from edge_prop.visualization.adj_mat_to_image import graph2image

np.set_printoptions(precision=3)
from sklearn.exceptions import ConvergenceWarning
from tqdm.autonotebook import tqdm

EDGEPROP_BASE_DIR = os.path.dirname(__file__) + "/"


class SparseEdgeProp(SparseBaseModel):
    def _perform_edge_prop_on_graph(self, adj_mat: COO, y: COO, max_iter=100,
                                    tol=1e-3,tb_exp_name:str=None) -> np.ndarray:
        """
        Performs the EdgeProp algorithm on the given graph.
        returns the label distribution (|N|, |N|) matrix with scores between -1, 1 stating the calculated label distribution.
        TODO: inaccurate
        """
        # create tensorboard
        if tb_exp_name is not None:
            # create the tensorboard
            path = os.path.join(TENSORBOARD_DIR, tb_exp_name)  # , str(datetime.datetime.now()))
            writer = SummaryWriter(path)
            global_step = 0
        logging.info("init calcs")

        label_distributions = y.copy()
        l_previous = None
        original_edge = y.sum(axis=-1, keepdims=True) == 1
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
                if tb_exp_name is not None:
                    if y.shape[-1] > 2:
                        logging.warning("Graph visualization of multi class not supported ATM!")
                    else:
                        graph_image = graph2image(label_distributions[:,:,-1], adj_mat)
                        writer.add_image("Graph", graph_image, global_step=global_step)
                    self.edge_distributions = label_distributions
                    self.write_evaluation_to_tensorboard(writer, global_step)
                    writer.flush()
                    global_step += 1
                l_previous = label_distributions.copy()
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
                original_edges_labels = y * self.alpha + (original_edge) * mat * (1 - self.alpha)
                label_distributions = original_edges_labels + (1 - original_edge) * mat
            else:
                warnings.warn("max_iter was reached without convergence", category=ConvergenceWarning)
                n_iter += 1

        return label_distributions