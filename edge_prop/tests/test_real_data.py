from unittest import TestCase

from sklearn.metrics import accuracy_score
import numpy as np
import random

from edge_prop.constants import DATASET2PATH
from edge_prop.data_loader import DataLoader
from edge_prop.models.dense_baseline import DenseBasline
from edge_prop.models.dense_edge_propagation import DenseEdgeProp


class TestDenseEdgeProp(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(42)
        random.seed(18)  # 18 = חי
        path = DATASET2PATH['slashdot']
        cls.graph, cls.y_true, cls.test_indices = DataLoader(path, test_size=0.2).load_data(1_000)
        cls.y_test = cls.y_true[cls.test_indices]

    def test_baseline(self):
        baseline = DenseBasline(self.graph.y_attr, max_iter=1000, alpha=0.8, tol=1e-3)
        baseline.fit(self.graph)
        y_pred = baseline.predict()[self.test_indices]
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred)}")

    def test_edgeprop(self):
        edge_prop = DenseEdgeProp(self.graph.y_attr, max_iter=1000, alpha=0.8, tol=1e-3)
        edge_prop.fit(self.graph)

        y_pred = edge_prop.predict()[self.test_indices]
        print(f"Accuracy: {accuracy_score(self.y_test, y_pred)}")


