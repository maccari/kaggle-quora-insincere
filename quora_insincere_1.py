import pandas as pd
import numpy as np
import os
import logging
from collections import Counter
import spacy
from spacy.tokenizer import Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from functools import partial
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Dict, Any, Iterator, Tuple, Iterable
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(funcName)s:%(lineno)d [%(levelname)s] %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler('out.log')])
logger = logging.getLogger(__name__)


def handle_exception(exc_type, exc_value, exc_traceback):
    exc_info = (exc_type, exc_value, exc_traceback)
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(*exc_info)
        return
    logger.error("Uncaught exception", exc_info=exc_info)


sys.excepthook = handle_exception


def set_seeds(seed):
    """ set seed for numpy and pytorch """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(seed)


def _get_candidate_dirs():
    """ if file candidate_directories.csv exists, read each line as a path
        return the set of paths read
    """
    candidate_dirs = set(['.', '../input'])
    if 'candidate_directories.csv' in os.listdir('.'):
        df = pd.read_csv('candidate_directories.csv', names=['directory'])
        candidate_dirs |= set(df['directory'].tolist())
    return candidate_dirs


def get_data_dir():
    """ helper to find directory containing input directory depending
        on machine being used
    """
    candidate_dirs = _get_candidate_dirs()
    data_files = set(['train.csv', 'test.csv'])
    for candidate_dir in candidate_dirs:
        candidate_dir = os.path.expanduser(candidate_dir)
        if not os.path.exists(candidate_dir):
            logger.debug(f"could not find directory {candidate_dir}")
            continue
        dir_files = set(os.listdir(candidate_dir))
        if data_files < dir_files:
            return candidate_dir
    raise FileNotFoundError(f"{data_files} not found in {candidate_dirs}")


def get_log_dir():
    """ return log dir, create it if does not exist """
    log_dir = './logs/'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    return log_dir


def load_data(data_dir, use_saved=False, shuffle=True):
    """ load training dataset and test dataset """
    saved_files = set(['train.pkl', 'test.pkl'])
    if use_saved and saved_files < set(os.listdir(data_dir)):
        train_data = pd.read_pickle(os.path.join(data_dir, 'train.pkl'))
        test_data = pd.read_pickle(os.path.join(data_dir, 'test.pkl'))
    else:
        train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    if shuffle:
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        test_data = test_data.sample(frac=1).reset_index(drop=True)

    return train_data, test_data


def load_embeddings(
        data_dir, model, top_n=0, vocab_filter=None, oov_vocab=None,
        padding_token='_PAD_', oov_token='_OOV_', lower=False):
    """ load embedding model in memory
        if top_n > 0, load the first top_n embeddings in file
        if vocab_filter is not None, ignore top_n
        if oov_vocab, add tokens in it that is not in vocab, init them using
            mean weight
        padding_token has idx 0
        oov_token has idx 1
        if lower, if a word is non-lower and has no lower version, use it as
            lower version
    """
    logger.info("load embeddings")
    vocab = {padding_token: 0, oov_token: 1}
    weights = [None, None]
    with open(os.path.join(data_dir, model)) as ifs:
        for idx, line in enumerate(ifs):
            if idx % 10000 == 0:
                logger.debug(idx)
            if not vocab_filter and top_n and idx >= top_n:
                break
            line = line.rstrip('\n').split(' ')
            word, vector = line[0], list(map(float, line[1:]))
            original_word = word
            if lower:
                word = word.lower()
            if vocab_filter and word not in vocab_filter:
                continue
            if word in vocab:
                if lower and original_word.islower():
                    # previous encounter was not lower, replace with current
                    weights[vocab[word]] = vector
            else:
                vocab[word] = len(vocab)
                weights.append(vector)
    if len(weights) <= 2:
        raise ValueError("No weight loaded")
    if vocab_filter:
        logger.info(f"Found {len(vocab)}/{len(vocab_filter)} tokens in model")
    emb_size = len(weights[2])
    weights[vocab[padding_token]] = [0.] * emb_size
    weights[vocab[oov_token]] = [0.] * emb_size
    mean_weight = np.mean(weights, axis=0)
    weights[vocab[oov_token]] = mean_weight
    if oov_vocab:
        oov_to_add = oov_vocab - set(vocab)
        logger.info(f"Add {len(oov_to_add)} oov tokens")
        for token in oov_to_add:
            vocab[token] = len(vocab)
            weights.append(mean_weight)
    weights = np.array(weights)
    return weights, vocab


def get_top_terms(data):
    classes = set(data['target'])
    top_terms = {class_: Counter() for class_ in classes}
    for index, row in data.iterrows():
        tokens = row['question_text'].split()
        top_terms[row['target']].update(tokens)
    return top_terms


def preprocess_data(data, tokenizer, lower=False):
    """ preprocess dataset: tokenize text, optionally lowercase
    """
    logger.info(f"preprocess data (lower={lower})")
    tokenize = partial(_tokenize, tokenizer=tokenizer, lower=lower)
    tokenized = []
    for index, row in data.iterrows():
        if index % 100000 == 0:
            logger.debug(index)
        tokenized.append(tokenize(row['question_text']))
    data['tokenized'] = tokenized


def _tokenize(string, tokenizer, lower=False):
    """ tokenize, optionally lowercase """
    tokenized = tokenizer(string)
    tokens = [token.text for token in tokenized]
    if lower:
        tokens = [token.lower() for token in tokens]
    return tokens


def run_eda(train_data, test_data):
    train_data_counts = train_data.groupby('target')['qid'].count()
    analysis = {
        'train data size': len(train_data),
        'test data size': len(test_data),
        'train data classes counts': train_data_counts,
        'top terms': get_top_terms(train_data),
    }
    return analysis


def save_data(data_dir, train_data, test_data):
    """ if not saved, save training data and test_data as pickle """
    saved_files = set(['train.pkl', 'test.pkl'])
    if not saved_files < set(os.listdir(data_dir)):
        train_data.to_pickle(os.path.join(data_dir, 'train.pkl'))
        test_data.to_pickle(os.path.join(data_dir, 'test.pkl'))
        logger.info("data saved")


def map_to_input_space(
        data, vocab, max_seq_len, pad_token='_PAD_', oov_token='_OOV_'):
    """ map token to token idx
    """
    logger.info("map to input space")
    X = np.full((len(data), max_seq_len), vocab[pad_token], dtype=int)
    for index, row in data.iterrows():
        if index % 100000 == 0:
            logger.debug(index)
        for ti, token in enumerate(row['tokenized']):
            if ti >= max_seq_len:
                break
            X[index][ti] = vocab.get(token) or vocab[oov_token]
    return X


class FeedForwardNN(nn.Module):
    """ Feed-forward neural network model
    """

    def __init__(
            self, input_size, num_classes, weights, trainable_emb=False,
            hidden1=None, padding_idx=0, emb_agg='mean'):
        """ weights: weights of pretrained embeddings
            if hidden1 is None, does not add hidden layer
            emb_agg: embedding aggregation method, 'sum' or 'mean'
        """
        super().__init__()
        self.weights = weights
        self.trainable_emb = trainable_emb
        self.padding_idx = padding_idx
        self.emb_agg = emb_agg
        self._init_embeddings()
        if hidden1:
            self.input1 = nn.Linear(input_size, hidden1)
            self.hidden1 = nn.Linear(hidden1, num_classes)
        else:
            self.input1 = nn.Linear(input_size, num_classes)
            self.hidden1 = None
        self.activation = F.log_softmax

    def _init_embeddings(self):
        num_emb, emb_size = self.weights.shape
        self.embed1 = nn.Embedding(
            num_emb, emb_size, self.padding_idx,
            _weight=torch.from_numpy(self.weights))
        self.embed1.weight.requires_grad = self.trainable_emb

    def forward(self, inputs, inputs_lengths=None):
        """ forward pass """
        embed1 = self.embed1(inputs).to(torch.float)
        agg_embed1 = embed1.sum(dim=1)
        if self.emb_agg == 'mean':
            if inputs_lengths is None:
                inputs_lengths = (inputs != self.padding_idx).sum(dim=1)
            inputs_lengths = inputs_lengths.to(torch.float).view(-1, 1)
            agg_embed1 /= inputs_lengths
        out = self.activation(self.input1(agg_embed1), dim=1)
        if self.hidden1:
            out = self.activation(self.hidden1(out), dim=1)
        return out

    def predict(self, inputs):
        """ predict output class """
        _, predictions = torch.max(self.predict_proba(inputs), 1)
        return predictions

    def predict_proba(self, inputs):
        """ predict probability of each output class """
        return self.forward(inputs)

    def reset_weights(self):
        """ reset model weights """
        self._init_embeddings()
        self.apply(weight_reset)


class RecurrentNN(nn.Module):
    """ Recurrent neural network model
    """

    def __init__(
            self, input_size, num_classes, weights, trainable_emb=False,
            hidden_dim_rnn=50, num_layers_rnn=1, unit_type='LSTM', dropout=0.,
        """ unit_type: 'LSTM' or 'GRU' """
            padding_idx=0, bidirectional=True, maxpooling=True,
            hidden_linear1=None):
        super().__init__()
        self.input_size = input_size
        self.weights = weights
        self.trainable_emb = trainable_emb
        self.hidden_dim_rnn = hidden_dim_rnn
        self.num_layers_rnn = num_layers_rnn
        self.bidirectional = bidirectional
        self.maxpooling = maxpooling
        self.unit_type = unit_type
        self.padding_idx = padding_idx
        self._init_embeddings()
        if unit_type == 'LSTM':
            self.rnn = nn.LSTM(
                self.input_size, self.hidden_dim_rnn, self.num_layers_rnn,
                bidirectional=self.bidirectional, batch_first=True)
        elif unit_type == 'GRU':
            self.rnn = nn.GRU(
                input_size, self.hidden_dim_rnn, self.num_layers_rnn,
                bidirectional=self.bidirectional, batch_first=True)
        else:
            raise ValueError(f"Unknown unit_type {unit_type}")
        self.linear1 = nn.Linear(
            self.hidden_dim_rnn * (1+self.bidirectional), num_classes)
        self.dropout = nn.Dropout(p=dropout)
        if self.maxpooling:
            # as opposed to MaxPool1D, AdaptiveMaxPool1d does not need kernel
            # size (== max_seq_len batch) which is different for since we have
            # variable length inputs
            self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.activation = F.log_softmax

    def _init_embeddings(self):
        num_emb, emb_size = self.weights.shape
        self.embed1 = nn.Embedding(
            num_emb, emb_size, self.padding_idx,
            _weight=torch.from_numpy(self.weights))
        self.embed1.weight.requires_grad = self.trainable_emb

    def forward(self, inputs, inputs_lengths=None):
        """ forward pass """
        embed1 = self.embed1(inputs).to(torch.float)
        if inputs_lengths is None:
            inputs_lengths = (inputs != self.padding_idx).sum(dim=1)
        # reset state at beginning of each batch
        self.hidden_rnn = self._init_hidden(len(embed1))
        # sort by length for pack_padded_sequence
        inputs_lengths, sort_idx = inputs_lengths.sort(0, descending=True)
        embed1 = embed1[sort_idx]
        # pack sequences to feed to rnn
        packed_embed1 = pack_padded_sequence(
            embed1, inputs_lengths, batch_first=True)
        rnn_out, self.hidden_rnn = self.rnn(packed_embed1, self.hidden_rnn)
        # unpack sequences out of rnn
        unpacked_rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        # put back in original order (necessary to compare to targets)
        _, unsort_idx = sort_idx.sort(0)
        if self.maxpooling:
            unpacked_rnn_out = unpacked_rnn_out[unsort_idx]
            out = self.max_pool(unpacked_rnn_out.permute(0, 2, 1)).squeeze(2)
        else:
            # get the last timestep of each element
            # seems to be different than self.hidden[0]
            # idx = (in_len_batch - 1).view(-1, 1).expand(
            #   len(in_len_batch), unpacked_rnn_out.shape[2]).unsqueeze(1)
            # last_step_rnn_out = unpacked_rnn_out.gather(1, idx).squeeze(1)
            # not as good as for variable length many gradients is zero
            # last_step_rnn_out = unpacked_rnn_out[:, -1, :]
            # [batch_size, max_seq_len (of batch), (1+is_bidir)*hidden_dim_rnn]
            last_step_rnn_out = torch.cat(tuple(self.hidden_rnn[0]), dim=1)
            out = last_step_rnn_out[unsort_idx]
        # note that dropout argument of RNN layer applies dropout on all but
        # the last layer, so it is not applied if num_layers_rnn = 1
        out = self.dropout(out)
        out = self.activation(self.linear1(out), dim=1)
        return out

    def predict_proba(self, inputs):
        """ predict probability of each output class """
        return self.forward(inputs)

    def predict(self, inputs):
        """ predict output class """
        _, predictions = torch.max(self.predict_proba(inputs), 1)
        return predictions

    def _init_hidden(self, batch_size):
        """ init hidden state
            if unit_type == 'LSTM', returns (h0, c0) elif 'GRU' returns h0
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h0 = Variable(torch.zeros(
            self.num_layers_rnn * (1 + self.bidirectional), batch_size,
            self.hidden_dim_rnn))
        if self.unit_type == 'LSTM':
            c0 = Variable(torch.zeros(
                self.num_layers_rnn * (1 + self.bidirectional), batch_size,
                self.hidden_dim_rnn))
            return h0.to(device), c0.to(device)
        elif self.unit_type == 'GRU':
            return h0.to(device)

    def reset_weights(self):
        """ reset model weights """
        self._init_embeddings()
        self.apply(weight_reset)


def batchify(l, batch_size):
    """ return generator of batches of batch_size """
    for offset in range(0, len(l), batch_size):
        yield l[offset:offset+batch_size]


def generate_random_params(
        params_space: Dict[str, Any], num_samples: int
        ) -> Iterator[Dict[str, Any]]:
    """ yield a generator of parameters, drawing a value for each param
        :param params_space: maps param -> values
            if values is a list, draw a random element from it,
            else return values
        :param num_samples: number of combinations to generate
    """
    for _ in range(num_samples):
        params = {}
        for param, values in params_space.items():
            if type(values) == list:
                value = np.random.choice(values).item()
            else:
                value = values
            params[param] = value
        yield params


def _dict2sortedtuple(d):
    return tuple(sorted(d.items()))


def run_random_search(
        X, y, weights, params_space, params_score, num_samples,
        train_ratio=0.8, num_folds=None):
    """ perform random search
        if num_folds is not None, train_ratio is ignored
        fill params_score dictionary with params -> scores
        if num_folds is not None, scores is a list of list of scores per fold
        per iteration, else scores is a list of scores per iteration
    """
    param_samples = generate_random_params(params_space, num_samples)
    for params in param_samples:
        tuple_params = _dict2sortedtuple(params)
        if tuple_params in params_score:
            continue
        logger.info(f"drawn params: {params}")
        model, scores = train_for_params(
            X, y, weights, params, train_ratio, num_folds)
        params_score[tuple_params] = scores
        score = get_params_score(scores, params['num_folds'] is not None)
        logger.info(f"Max score for params: {score}")
        best_params, best_score = get_best_params(params_score)
        logger.info(f"Best params so far {best_score} using {best_params}")


def get_best_params(
        params_score: Dict[Tuple[Any], Iterable]
        ) -> Tuple[Dict[Tuple, Any], float]:
    """ return (best_params, score) given dictionary of
        params -> score per iteration
        :param params_score: params argument is a dict mapping a tuple of pairs
         (param, value) to scores
        return: params returned is a dict
    """
    best_params, best_score = {}, 0.
    for params, scores in params_score.items():
        params_dict = dict(params)
        score = get_params_score(scores, params_dict['num_folds'] is not None)
        if score > best_score:
            best_params, best_score = params_dict, score
    return best_params, best_score


def get_params_score(scores: list, cv: bool = False) -> float:
    """ return max score of all epochs
        if cv=False, scores should be a list of score per epoch, else scores
        should be a list of list of score per epoch per fold of
        cross-validation
    """
    if cv:
        folds_score = [max(fold_scores, default=0.) for fold_scores in scores]
        score = np.mean(folds_score) if folds_score else 0.
    else:
        score = max(scores, default=0.)
    return score


def train_for_params(X, y, weights, params, train_ratio=0.8, num_folds=None):
    """ train classifier for given parameters
    """
    model = build_model_from_params(params, len(set(y)), weights)
    if params['criterion'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown criterion {params['criterion']}")
    if params['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            model.parameters(), lr=params['learning_rate'],
            momentum=params['momentum'], weight_decay=params['weight_decay'])
    else:
        raise ValueError(f"Unknown optimizer {params['optimizer']}")
    if num_folds:
        scores = train_cv(
            X, y, model, criterion, optimizer, params['num_epochs'],
            params['batch_size'], patience=params['patience'],
            min_improvement=params['min_improvement'])
    elif train_ratio:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio)
        train_index, test_index = next(sss.split(X, y))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        scores = train(
            X_train, X_test, y_train, y_test, model, criterion, optimizer,
            num_epochs=params['num_epochs'], batch_size=params['batch_size'],
            patience=params['patience'],
            min_improvement=params['min_improvement'])
    else:
        scores = train(
            X, X, y, y, model, criterion, optimizer,
            num_epochs=params['num_epochs'], batch_size=params['batch_size'],
            patience=params['patience'],
            min_improvement=params['min_improvement'])
    return model, scores


def build_model_from_params(params, num_classes, weights):
    if params['clf_model'] == 'FeedForwardNN':
        model = FeedForwardNN(
            input_size=weights.shape[1], num_classes=num_classes,
            weights=weights, trainable_emb=params['trainable_emb'],
            hidden1=params['hidden_size_1'])
    elif params['clf_model'] == 'RecurrentNN':
        model = RecurrentNN(
            input_size=weights.shape[1], num_classes=num_classes,
            weights=weights, trainable_emb=params['trainable_emb'],
            hidden_dim_rnn=params['hidden_dim_rnn'], dropout=params['dropout'])
    else:
        raise ValueError(f"Unknown model {params['clf_model']}")
    logger.info(f"Built model:\n{model}")
    return model


def train_cv(
        X, y, model, criterion, optimizer, num_epochs, batch_size,
        num_folds=5, metric='f1_score', patience=10, min_improvement=0.):
    """ train using cross-validation
        return score per iteration per fold
    """
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
    cv_scores = []
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        logger.info(f"fold index: {fold_idx}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        fold_scores = train(
            X_train, X_test, y_train, y_test, model, criterion, optimizer,
            num_epochs, batch_size, metric, patience)
        cv_scores.append(fold_scores)
    return cv_scores


def train(
        X_train, X_test, y_train, y_test, model, criterion, optimizer,
        num_epochs, batch_size, metric='f1_score', patience=10,
        min_improvement=0.):
    """ train model
        return score per iteration
    """
    model_checkpoint_path = 'best_model.pt'
    X_train, X_test = torch.from_numpy(X_train), torch.from_numpy(X_test)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)
    scores = []
    iter = 0
    model.reset_weights()
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Start training...")
    for epoch in range(num_epochs):
        X_train, y_train = unison_shuffled_copies(X_train, y_train)
        for offset in range(0, len(X_train), batch_size):
            inputs = X_train[offset: offset+batch_size].to(device)
            targets = y_train[offset: offset+batch_size].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            iter += batch_size
            if iter % 100000 == 0:
                logger.debug(
                    f"iter: {iter}, loss: {loss.item():.3f}")
        logger.debug(f"evaluate {metric} on: {Counter(y_test.tolist())}")
        score = evaluate(X_test, y_test, model, metric)
        if score >= max(scores, default=0.):
            torch.save(model.state_dict(), model_checkpoint_path)
        scores.append(score)
        logger.info(f"EPOCH {epoch}: {metric} {score:.3f}")
        if early_stopping(scores, patience, min_improvement):
            logger.info(
                f"Early stopping triggered (patience {patience}, "
                f"min_improvement {min_improvement})")
            break
    if os.path.exists(model_checkpoint_path):
        model.load_state_dict(torch.load(model_checkpoint_path))
    model.eval()
    return scores


def evaluate(inputs, targets, model, metric, batch_size=1000):
    """ compute predictions for inputs and evaluate w.r.t to targets """
    predictions = predict(inputs, model, batch_size)
    if metric == 'f1_score':
        return f1_score(targets, predictions)
    else:
        raise Exception(f"unknown metric {metric}")


def predict(inputs, model, batch_size=1000):
    """ predict classes given inputs and model, used to process by batch if
        large number of inputs
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictions = []
    for input_batch in batchify(inputs, batch_size):
        input_batch = input_batch.to(device)
        predictions.extend(model.predict(input_batch).cpu().numpy())
    return predictions


def early_stopping(scores, patience, min_improvement=0.):
    """ return True if scores did not improvement within the last <patience>
        iterations, wih a minimun improvement of <min_improvement>
    """
    if len(scores) <= patience:
        return False
    best_score_outside_patience = max(scores[:-patience])
    best_score_in_patience = max(scores[-patience:])
    improvement = best_score_in_patience - best_score_outside_patience
    return improvement < min_improvement


def weight_reset(m):
    """ reset weights of model, used between runs
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def downsample(data, downsample_ratio=None, max_imbalance_ratio=None):
    """ downsample data, either keeping the same proportion between classes
        using <downsample_ratio> or by enforcing a maximum ratio of imbalance
        with <max_imbalance_ratio>.
        max_imbalance_ratio=2. means that the most common class cannot have
        more than twice the number of datapoints than the least common class
    """
    if max_imbalance_ratio:
        counts_per_class = dict(data.groupby('target')['qid'].count())
        downsampled_dfs = {}
        min_class_count = min(counts_per_class.values(), default=0.)
        min_class_count = int(min_class_count * max_imbalance_ratio)
        logger.info(f"Downsampling current counts {counts_per_class}, "
                    f"keeping a max ratio of {max_imbalance_ratio} between "
                    f"classes")
        for class_, count in counts_per_class.items():
            downsampled_dfs[class_] = data.loc[data['target'] == class_] \
                .sample(min(count, min_class_count))
        data = pd.concat(downsampled_dfs.values()).reset_index(drop=True)
    if downsample_ratio and downsample_ratio < 1.:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=downsample_ratio)
        data_index, _ = next(sss.split(data, data['target']))
        data = data.iloc[data_index]
    # shuffle
    data = data.sample(frac=1).reset_index(drop=True)
    return data


def unison_shuffled_copies(a, b):
    """ shuffle two numpy array keeping the alignment between them """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def build_vocab(data):
    """ return set containing all distinct tokens appearing in the data
    """
    vocab = set()
    for tokenized in data['tokenized'].values:
        vocab.update(tokenized)
    return vocab


def get_saved_best_params():
    best_params = {
        'batch_size': 256,
        'clf_model': 'RecurrentNN',
        'criterion': 'CrossEntropyLoss',
        'downsample': 1.,
        'dropout': 0.5,
        'embedding_model': 'glove.840B.300d/glove.840B.300d.txt',
        'hidden_dim_rnn': 150,
        'learning_rate': 0.004,
        'lower': True,
        'max_imbalance_ratio': 3.0,
        'max_seq_len': 50,
        'min_improvement': 0.01,
        'momentum': 0.6000000000000001,
        'num_epochs': 200,
        'num_folds': None,
        'optimizer': 'SGD',
        'patience': 10,
        'seed': 21757971,
        'spacy_model': 'en_core_web_sm',
        'train_ratio': 0.8,
        'trainable_emb': True,
        'vocab_size': 1000000.0,
        'weight_decay': 1e-06
    }
    return best_params


def get_params_space():
    PARAMS_SPACE = {
        'seed': np.random.randint(1E9),
        'lower': True,
        'downsample': 1.,  # None, 0 or 1 to ignore
        'max_imbalance_ratio': 3.,
        'max_seq_len': 50,
        'embedding_model': 'glove.840B.300d/glove.840B.300d.txt',
        'vocab_size': 1E6,
        'spacy_model': 'en_core_web_sm',
        'batch_size': [2**i for i in range(8, 12)],
        'weight_decay': [10**i for i in range(-6, -4)],
        'momentum': np.arange(0., 0.91, 0.1).tolist(),
        'num_epochs': 200,
        'patience': 10,
        'min_improvement': 1E-2,
        'criterion': "CrossEntropyLoss",
        'optimizer': "SGD",
        'trainable_emb': True,
        'train_ratio': 0.8,
        'num_folds': None,
        'clf_model': 'RecurrentNN',
    }
    if PARAMS_SPACE['clf_model'] == 'FeedForwardNN':
        PARAMS_SPACE.update({
            'hidden_size_1': list(range(0, 201, 50)),
            'emb_agg': ['mean'],
            'learning_rate': [i*1E-4 for i in [1, 2, 5, 10]],
        })
    elif PARAMS_SPACE['clf_model'] == 'RecurrentNN':
        PARAMS_SPACE.update({
            'hidden_dim_rnn': list(range(50, 201, 50)),
            'learning_rate': [i*1E-3 for i in [.8, 1., 2, 4]],
            'dropout': np.arange(0., 0.51, 0.1).tolist(),
        })
    else:
        raise ValueError(
            f"Unknown classifier model {PARAMS_SPACE['clf_model']}")
    return PARAMS_SPACE


if __name__ == '__main__':
    SUBMIT = True
    PARAMS_SPACE = get_params_space()
    best_params = get_saved_best_params()
    if SUBMIT:
        PARAMS_SPACE = best_params
    set_seeds(PARAMS_SPACE['seed'])
    logger.info(f"SEED: {PARAMS_SPACE['seed']}")
    logger.info(f"torch.initial_seed(): {torch.initial_seed()}")
    NUM_SAMPLES = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Use device {device}")
    data_dir = get_data_dir()
    embed_dir = os.path.join(data_dir, 'embeddings/')
    log_dir = get_log_dir()
    train_data, test_data = load_data(data_dir)
    train_data = downsample(
        train_data, PARAMS_SPACE['downsample'],
        PARAMS_SPACE['max_imbalance_ratio'])
    nlp = spacy.load(PARAMS_SPACE['spacy_model'])
    tokenizer = Tokenizer(nlp.vocab)
    preprocess_data(train_data, tokenizer, PARAMS_SPACE['lower'])
    preprocess_data(test_data, tokenizer, PARAMS_SPACE['lower'])
    train_vocab = build_vocab(train_data)
    weights, vocab = load_embeddings(
        embed_dir, model=PARAMS_SPACE['embedding_model'],
        top_n=PARAMS_SPACE['vocab_size'], vocab_filter=train_vocab,
        oov_vocab=train_vocab, lower=PARAMS_SPACE['lower'])
    logger.info(f"Vocab size: {len(vocab)}")
    emb_size = weights.shape[1]
    num_classes = len(set(train_data['target']))
    X_train = map_to_input_space(
        train_data, vocab, PARAMS_SPACE['max_seq_len'])
    y_train = train_data['target'].values
    if not SUBMIT:
        params_score = {}
        run_random_search(
            X_train, y_train, weights, PARAMS_SPACE, params_score,
            NUM_SAMPLES, PARAMS_SPACE['train_ratio'],
            PARAMS_SPACE['num_folds'])
        best_params, score = get_best_params(params_score)
        model, scores = train_for_params(
            X_train, y_train, weights, best_params, best_params['train_ratio'],
            best_params['num_folds'])
        # analysis = run_eda(train_data, test_data)
    else:
        logger.info("final training")
        model, scores = train_for_params(
            X_train, y_train, weights, best_params, best_params['train_ratio'],
            best_params['num_folds'])
        logger.info("preprocess and predict target on test set")
        X_test = map_to_input_space(
            test_data, vocab, PARAMS_SPACE['max_seq_len'])
        X_test = torch.from_numpy(X_test)
        test_data['prediction'] = predict(X_test, model)
        test_data[['qid', 'prediction']].to_csv('submission.csv', index=False)
