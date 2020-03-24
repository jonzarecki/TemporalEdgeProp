import os

from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression
import numpy as np
from gensim.models import KeyedVectors
from edge_prop.constants import LABEL_TRAIN, NO_LABEL, NODE2VEC_CACHE
from edge_prop.graph_wrappers import BaseGraph


class Node2VecClassifier:

    def __init__(self, n2v_kwargs={}, n2v_fit_kwargs={}, cache_name="default"):
        self.n2v_kwargs = n2v_kwargs
        self.n2v_fit_kwargs = n2v_fit_kwargs
        self.clf = LogisticRegression(multi_class='ovr')
        self.save_path = os.path.join(NODE2VEC_CACHE, cache_name + '.emb')

    def fit(self, g: BaseGraph, label=LABEL_TRAIN):
        graph = g.graph_nx

        # fit node2vec
        if os.path.exists(self.save_path):
            node_vectors_dict = KeyedVectors.load(self.save_path)
        else:
            self.node2vec = Node2Vec(graph, workers = 1, **self.n2v_kwargs)
            self.node2vec = self.node2vec.fit(**self.n2v_fit_kwargs)
            node_vectors_dict = self.node2vec.wv

            # save the embedding
            if not os.path.exists(NODE2VEC_CACHE):
                os.mkdir(NODE2VEC_CACHE)
            node_vectors_dict.save(self.save_path)
        edge_vectors = [np.concatenate([node_vectors_dict[str(u)], node_vectors_dict.wv[str(v)]]) for u, v in g.edge_order]
        self.edge_vectors = np.stack(edge_vectors)

        # extract the train set
        edge_labels = np.array([label[0] for edge, label in g.get_edge_attributes_ordered(label)])
        train_mask = edge_labels != NO_LABEL
        x_train = self.edge_vectors[train_mask == True]
        y_train = edge_labels[train_mask]

        # fit the logistic regression
        self.clf.fit(x_train, y_train)

    def predict_proba(self, indices):
        x_test = self.edge_vectors[indices]
        y_proba = self.clf.predict_proba(x_test)
        return y_proba


