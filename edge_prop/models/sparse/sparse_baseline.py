import logging
import os
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sparse import COO

from edge_prop.constants import TENSORBOARD_DIR
from edge_prop.models import SparseBaseModel
from edge_prop.visualization.adj_mat_to_image import graph2image


EDGEPROP_BASE_DIR = os.path.dirname(__file__) + "/"


class SparseBaseline(SparseBaseModel):
    def _perform_edge_prop_on_graph(self, adj_mat: np.ndarray, y: np.ndarray, max_iter=50,
                                    tol=1e-3, tb_exp_name=None) -> np.ndarray:
        """
        Performs the EdgeProp algorithm on the given graph.
        returns the label distribution (|N|, |N|) matrix with scores between -1, 1 stating the calculated label distribution.
        """
        # A = adj_mat.todense()
        if tb_exp_name is not None:
            # create the tensorboard
            logging.warning("tensorboard Not implemented yet for baseline")
            self.tb_exp_name = None
            # path = os.path.join(TENSORBOARD_DIR, tb_exp_name)  # , str(datetime.datetime.now()))
            # writer = SummaryWriter(path)
            # global_step = 0
        Y = y

        l_previous = None

        D = np.sum(adj_mat, axis=0)
        D = D.todense()
        D[D == 0] = 1
        mD = D[:, np.newaxis]

        # A = adj_mat.tocsr()
        # s = adj_mat.sum(axis=0)
        # s.fill_value = np.int64(1)
        A = (adj_mat / mD).tocsr()  # norm rows
        print("starting init setup")
        last_Y = Y.sum(axis=0).todense()  # same as before, might be a bug
        # last_Y = adj_mat.dot(Y).sum(axis=1).todense()  # TODO: axis 0 or 1 ?
        # last_Y = last_Y.todense()
        mat_sum = np.sum(last_Y, axis=-1)[:, np.newaxis]
        mat_sum[mat_sum == 0] = 1
        last_Y = last_Y / mat_sum
        # D = np.sum(adj_mat, axis=0).todense()
        # D[D == 0] = 1
        # mD = D[:, np.newaxis]

        with tqdm(range(max_iter), desc='Fitting model', unit='iter') as pbar:

            for n_iter in pbar:
                dif = np.inf if l_previous is None else np.abs(last_Y - l_previous).sum()
                pbar.set_postfix({'dif': dif})
                if n_iter != 0 and dif < tol:  # did not change
                    break  # end the loop, finished
                l_previous = last_Y.copy()

                # B = adj_mat.dot(last_Y)
                B = A.dot(last_Y)
                # as in paper
                last_Y = B

                continue
                # below not specified in the paper
                last_Y = B / mD
                last_Y = last_Y / last_Y.sum(axis=1)[:, np.newaxis]

        st = time.time()
        print("starting expand to edge")
        last_Y = COO(last_Y)
        last_Y_edges = np.multiply(adj_mat[:, :, np.newaxis], last_Y[:, np.newaxis, :]) + \
                       np.multiply(adj_mat[:, :, np.newaxis], last_Y[np.newaxis, :, :])
        print("starting norm")
        mat_sum = np.sum(last_Y_edges, axis=-1, keepdims=True)
        mat_sum.fill_value = np.float64(1.0)
        last_Y_edges = last_Y_edges / mat_sum  # sums all elems in dim 2 to 1
        print(f"expand to edge + norm took {int(time.time() - st)}s")
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
