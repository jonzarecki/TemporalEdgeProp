import numpy as np

from edge_prop.data_loader import DataLoader
from edge_prop.constants import DATASET2PATH
from edge_prop.models import GraphEdgePropagation, TemporalGraphEdgePropagation


def run(datasets, models):
    for dataset in datasets:
        graph, true_labels = DataLoader(DATASET2PATH[dataset]).load_data()

        for model in models:
            model.fit(graph)
            y_pred = model.predict()

            print(model.__str__(), 'acc:', np.sum(true_labels == y_pred))


if __name__ == '__main__':
    datasets = DATASET2PATH.keys()
    models = [GraphEdgePropagation(), TemporalGraphEdgePropagation()]
    run(datasets, models)
