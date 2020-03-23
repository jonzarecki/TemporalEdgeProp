import time
from itertools import product

from edge_prop.common.metrics import mean_rank, hit_at_k
from edge_prop.common.multiproc_utils import parmap
from edge_prop.common import multiproc_utils
from edge_prop.constants import DATASET2PATH
from edge_prop.data_loader import DataLoader
from edge_prop.models import SparseBaseline, SparseEdgeProp
from edge_prop.constants import LABEL_GT, LABEL_TRAIN

data_name = 'aminer_s'#'epinions'#'aminer_s'


def get_expr_name(alpha, test_size, alg_cls):
    return f"data_name={data_name}, test_size={test_size}, alpha={alpha}, model={alg_cls.__name__}"


def run_alg_on_data(alpha, test_size, alg_cls):

    if alg_cls == SparseBaseline:  # no alpha
        if alpha != 0.:
            return (get_expr_name(alpha, test_size, alg_cls), {})
    if 'aminer' in data_name:
        if test_size != 0.75:
            return (get_expr_name(alpha, test_size, alg_cls), {})
        test_size = 1.

    expr_name = get_expr_name(alpha, test_size, alg_cls)


    print(expr_name)
    # create dataset
    path = DATASET2PATH[data_name]
    graph, true_labels, test_indices = DataLoader(path, test_size=test_size).load_data(10000)  # node number doesn't work on aminer
    y_test = true_labels[test_indices]

    print(f"Calculating {alg_cls.__name__}:")
    st = time.time()
    model = alg_cls(max_iter=100, alpha=alpha)
    model.fit(graph, LABEL_TRAIN)
    y_pred = model.predict_proba(test_indices)
    metrics = {f'hit_at_{k}': round(hit_at_k(y_test, y_pred, k=k), 3) for k in [1, 5, 10]}
    metrics.update({'mean_rank': round(mean_rank(y_test, y_pred), 3)})
    # our_metrics.update({'accuracy': round(accuracy_score(y_test, y_pred), 3)})
    print(f"took {(time.time() - st) / 60}. {metrics}")

    return expr_name, metrics


if __name__ == '__main__':
    alphas = [0, 0.5, 1]  # [0, 0.5, 0.8, 1]
    test_sizes = [0.25, 0.5, 0.75]
    compared_algs = [SparseBaseline]  #SparseEdgeProp,

    results_tpls = parmap(lambda args: run_alg_on_data(*args), list(product(alphas, test_sizes, compared_algs)), nprocs=1)
    results = dict(results_tpls)

    print(results)

    for (alpha, test_size) in product(alphas, test_sizes):
        baseline_metrics = results[get_expr_name(alpha, test_size, SparseBaseline)]
        our_metrics = results[get_expr_name(alpha, test_size, SparseEdgeProp)]
        print(f"alpha={alpha}, test_size={test_size}, \t Baseline: {baseline_metrics} \t New Model: {our_metrics}")
