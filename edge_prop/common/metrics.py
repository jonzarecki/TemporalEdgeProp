import numpy as np
from sklearn.metrics import accuracy_score
from scipy.stats import entropy

def mean_rank(y_test, y_pred):
    ranks = []
    for cur_y_test, cur_y_pred in zip(y_test, y_pred):
        cur_y_test_index = np.where(cur_y_test != 0)[0]
        cur_y_pred_index = np.argsort(cur_y_pred)[::-1]
        ranks += list(np.where(np.isin(cur_y_pred_index, cur_y_test_index))[0] + 1)
    return np.mean(ranks)


def hit_at_k(y_test, y_pred, k=1):
    hits = 0
    for cur_y_test, cur_y_pred in zip(y_test, y_pred):
        cur_y_test_index = np.where(cur_y_test != 0)[0]
        cur_y_pred_index = np.argsort(cur_y_pred)[::-1][:k]
        k_hits = np.isin(cur_y_pred_index, cur_y_test_index).sum()
        hits += k_hits
    max_possible_hist = np.minimum(k, y_test.sum(axis=1)).sum()
    return hits / max_possible_hist * 100


def get_all_metrics(y_pred, y_test):
    metrics = {f'hit_at_{k}': round(hit_at_k(y_test, y_pred, k=k), 3) for k in [1, 5, 10]}
    metrics['mean_rank'] = round(mean_rank(y_test, y_pred), 3)
    metrics['accuracy'] = round(accuracy_score(y_test.argmax(axis=-1), y_pred.argmax(axis=-1)), 3)

    pred_classes = y_pred.argmax(axis=-1)
    test_classes = y_test.argmax(axis=-1)
    num_classes = y_pred.shape[-1]
    metrics['KL_divergence'] = entropy(np.histogram(pred_classes, bins=num_classes)[0],
                                                 np.histogram(test_classes, bins=num_classes)[0])


    return metrics
