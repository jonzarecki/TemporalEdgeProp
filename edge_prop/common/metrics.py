import numpy as np


def mean_rank(y_test, y_pred):
    ranks = []
    for cur_y_test, cur_y_pred in zip(y_test, y_pred):
        ranks += list(np.where(np.isin(cur_y_pred, cur_y_test))[0])
    return np.mean(ranks)


def hit_at_k(y_test, y_pred, k=1):
    hits = 0
    for cur_y_test, cur_y_pred in zip(y_test, y_pred):
        if len(np.where(np.isin(cur_y_pred[:k], cur_y_test))[0]) > 0:
            hits += 1
    return hits/len(y_test)
