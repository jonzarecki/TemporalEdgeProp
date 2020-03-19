from itertools import product

from sklearn.metrics import accuracy_score
from tqdm import tqdm

from edge_prop.constants import DATASET2PATH
from edge_prop.data_loader import DataLoader
from edge_prop.models.dense_baseline import DenseBasline
from edge_prop.models.dense_edge_propagation import DenseEdgeProp

alphas = [0, 0.5, 0.8, 1]
test_sizes = [0.25, 0.5, 0.75]

path = DATASET2PATH['slashdot']
dtype_tuples = [('label', int), ('time', str)]
results = {}

for alpha, test_size in tqdm(product(alphas, test_sizes), desc="prod"):
    # create dataset
    graph, y_true, test_indices = DataLoader(path, test_size=test_size).load_data(10_000)
    y_test = y_true[test_indices]

    edge_prop = DenseEdgeProp(graph.y_attr, max_iter=100, alpha=alpha)
    edge_prop.fit(graph)
    y_pred = edge_prop.predict()[test_indices]
    our_accuracy = accuracy_score(y_test, y_pred)

    baseline = DenseBasline(graph.y_attr, max_iter=100, alpha=alpha)
    baseline.fit(graph)
    y_pred = baseline.predict()[test_indices]
    baseline_accuracy = accuracy_score(y_test, y_pred)
    results[(alpha, test_size)] = (our_accuracy, baseline_accuracy)

for (alpha, test_size), (new_acc, baseline_acc) in results.items():
    print(
        f"alpha={alpha}, test_size={test_size}, \t Baseline: {round(baseline_acc, 3)}\t New Model:{round(new_acc, 3)}")
