import numpy as np


def mean_rank(y_test, y_pred):
    ranks = []
    for cur_y_test, cur_y_pred in zip(y_test, y_pred):
        cur_y_test_index = np.where(cur_y_test != 0)[0]
        cur_y_pred_index = np.argsort(cur_y_pred)[::-1]
        ranks += list(np.where(np.isin(cur_y_pred_index, cur_y_test_index))[0]+1)
    return np.mean(ranks)


def hit_at_k(y_test, y_pred, k=1):
    hits = 0
    for cur_y_test, cur_y_pred in zip(y_test, y_pred):
        cur_y_test_index = np.where(cur_y_test != 0)[0]
        cur_y_pred_index = np.argsort(cur_y_pred)[::-1][:k]
        if len(np.where(np.isin(cur_y_pred_index, cur_y_test_index))[0]) > 0:
            hits += 1
    return hits/len(y_test)*100
