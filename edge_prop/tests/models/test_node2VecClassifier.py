from unittest import TestCase

from edge_prop.constants import DATASET2PATH
from edge_prop.data_loader import DataLoader
from edge_prop.models.node2vec_classifier import Node2VecClassifier


class TestNode2VecClassifier(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        path = DATASET2PATH['epinions']
        cls.graph, true_labels, cls.test_indices = DataLoader(path).load_data(100)
        cls.model = Node2VecClassifier()

    def test_fit(self):
        self.model.fit(self.graph)

    def test_predict_proba(self):
        self.model.predict_proba(self.test_indices)
