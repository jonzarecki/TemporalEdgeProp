import time
from itertools import product

from sklearn.metrics import accuracy_score
from tqdm import tqdm

from edge_prop.common.metrics import mean_rank, hit_at_k
from edge_prop.constants import DATASET2PATH
from edge_prop.data_loader import DataLoader
from edge_prop.models.dense_baseline import DenseBasline
from edge_prop.models.dense_edge_propagation import DenseEdgeProp
from edge_prop.models.sparse_baseline import SparseBasline
from edge_prop.models.sparse_edgeprop import SparseEdgeProp

alphas = [0, 0.5, 1]  # [0, 0.5, 0.8, 1]
test_sizes = [0.25, 0.5, 0.75]

path = DATASET2PATH['epinions']
dtype_tuples = [('label', int), ('time', str)]
results = {}

for alpha, test_size in product(alphas, test_sizes):
    print(f"test_size={test_size}, alpha={alpha}")
    # create dataset
    graph, y_true, test_indices = DataLoader(path, test_size=test_size).load_data()
    y_test = y_true[test_indices]

    print("Calculating edgeprop:")
    st = time.time()
    edge_prop = SparseEdgeProp(graph.y_attr, max_iter=100, alpha=alpha)
    edge_prop.fit(graph)
    y_pred = edge_prop.predict()[test_indices]
    our_metrics = {f'hit_at_{k}': round(hit_at_k(y_test, y_pred, k=k), 3) for k in [1, 5, 10]}
    our_metrics.update({'mean_rank': round(mean_rank(y_test, y_pred), 3)})
    our_metrics.update({'accuracy': round(accuracy_score(y_test, y_pred), 3)})
    print(f"took {(time.time() - st) / 60}. {our_metrics}")

    print("Calculating baseline:")
    st = time.time()
    baseline = SparseBasline(graph.y_attr, max_iter=100, alpha=alpha)
    baseline.fit(graph)
    y_pred = baseline.predict()[test_indices]
    baseline_metrics = {f'hit_at_{k}': round(hit_at_k(y_test, y_pred, k=k), 3) for k in [1, 5, 10]}
    baseline_metrics.update({'mean_rank': round(mean_rank(y_test, y_pred), 3)})
    baseline_metrics.update({'accuracy': round(accuracy_score(y_test, y_pred), 3)})
    print(f"took {(time.time() - st) / 60}. {baseline_metrics}")

    results[(alpha, test_size)] = (our_metrics, baseline_metrics)

for (alpha, test_size), (our_metrics, baseline_metrics) in results.items():
    print(
        f"alpha={alpha}, test_size={test_size}, \t Baseline: {baseline_metrics} \t New Model: {our_metrics}")
