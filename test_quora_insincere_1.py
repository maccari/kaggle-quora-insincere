import unittest
from quora_insincere_1 import (
    FeedForwardNN, train, load_embeddings, get_data_dir)
import numpy as np
import torch.nn as nn
import os
import torch.optim as optim


def generate_dummy_dataset(dataset_size, vocab, max_seq_len, padding_idx=0):
    """ generate a dummy dataset perfectly separable, using first half of the
        vocab to build samples of the first class and second half for the
        second class
    """
    vocab.discard(padding_idx)
    vocab = list(vocab)
    middle_idx_vocab = int(len(vocab) / 2)
    class_vocab = {0: vocab[:middle_idx_vocab], 1: vocab[middle_idx_vocab:]}
    X_train = np.empty((dataset_size, max_seq_len), dtype=int)
    y_train = np.empty((dataset_size), dtype=int)
    for x_idx in range(dataset_size):
        class_ = x_idx % 2
        seq_len = np.random.choice(range(1, max_seq_len + 1))
        tokens_idx = np.full(max_seq_len, padding_idx, dtype=int)
        seq = np.random.choice(class_vocab[class_], seq_len)
        tokens_idx[:seq_len] = seq
        X_train[x_idx, :] = tokens_idx
        y_train[x_idx] = class_
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
            dataset_size=10000, vocab=set(cls.vocab.values()), max_seq_len=50)
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
