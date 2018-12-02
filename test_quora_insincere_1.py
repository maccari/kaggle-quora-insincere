import unittest
from quora_insincere_1 import (
    FeedForwardNN, train, load_embeddings, get_data_dir)
import numpy as np
import torch.nn as nn
import os
import torch.optim as optim


def generate_dummy_dataset(dataset_size, vocab_size, max_seq_len):
    middle_idx_vocab = int(vocab_size / 2)
    X_train = np.empty((dataset_size, max_seq_len), dtype=int)
    y_train = np.empty((dataset_size), dtype=int)
    for x_idx in range(dataset_size):
        if x_idx % 2 == 0:
            tokens_idx = np.random.choice(
                np.arange(0, middle_idx_vocab), max_seq_len)
        else:
            tokens_idx = np.random.choice(
                np.arange(middle_idx_vocab, vocab_size), max_seq_len)
        X_train[x_idx, :] = tokens_idx
        y_train[x_idx] = x_idx % 2
    p = np.random.permutation(len(X_train))
    return X_train[p], y_train[p]


class TestFeedForwardNetwork(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data_dir = get_data_dir()
        embed_dir = os.path.join(data_dir, 'embeddings/')
        model_path = 'glove.840B.300d/glove.840B.300d.txt'
        cls.weights, cls.vocab = load_embeddings(
            embed_dir, model=model_path, top_n=1000)
        cls.emb_size = cls.weights.shape[1]
        cls.X_train, cls.y_train = generate_dummy_dataset(
            dataset_size=10000, vocab_size=len(cls.vocab), max_seq_len=50)
        cls.num_classes = len(set(cls.y_train))

    def _train(self):
        scores = train(
            self.X_train, self.X_train, self.y_train, self.y_train, self.model,
            self.criterion, self.optimizer, num_epochs=50, batch_size=100,
            patience=10, min_improvement=0.01)
        return scores

    def test_logreg_overfit_training_data(self):
        """ Test logreg can learn """
        self.model = FeedForwardNN(
            self.emb_size, self.num_classes, self.weights)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=.001)
        scores = self._train()
        self.assertTrue(max(scores, default=0.) > .8)

    def test_mlp_overfit_training_data(self):
        """ Test mlp can learn """
        self.model = FeedForwardNN(
            self.emb_size, self.num_classes, self.weights, hidden1=100)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=.001)
        scores = self._train()
        self.assertTrue(max(scores, default=0.) > .8)


if __name__ == '__name__':
    unittest.main()
