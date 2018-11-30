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
from functools import partial
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import f1_score
from itertools import zip_longest


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(funcName)s:%(lineno)d [%(levelname)s] %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler('out.log')])
logger = logging.getLogger(__name__)


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_candidate_dirs():
    candidate_dirs = set(['.', '../input'])
    if 'candidate_directories.csv' in os.listdir('.'):
        df = pd.read_csv('candidate_directories.csv', names=['directory'])
        candidate_dirs |= set(df['directory'].tolist())
    return candidate_dirs


def get_data_dir():
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
    log_dir = './logs/'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    return log_dir


def load_data(data_dir, use_saved=False, shuffle=True):
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


def load_embeddings(data_dir, model, top_n, vocab_filter=None):
    """ if vocab is not None, ignore top_n
    """
    logger.info("load embeddings")
    vocab = {}
    weights = []
    with open(os.path.join(data_dir, model)) as ifs:
        for idx, line in enumerate(ifs):
            if idx % 10000 == 0:
                logger.debug(idx)
            if not vocab and idx >= top_n:
                break
            line = line.rstrip('\n').split(' ')
            word, vector = line[0], list(map(float, line[1:]))
            if vocab_filter and word not in vocab_filter:
                continue
            vocab[word] = len(vocab)
            weights.append(vector)
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
    logger.info(f"preprocess data (lower={lower})")
    tokenize = partial(_tokenize, tokenizer=tokenizer, lower=lower)
    tokenized = []
    for index, row in data.iterrows():
        if index % 100000 == 0:
            logger.debug(index)
        tokenized.append(tokenize(row['question_text']))
    data['tokenized'] = tokenized


def _tokenize(string, tokenizer, lower=False):
    """ tokenize """
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
    saved_files = set(['train.pkl', 'test.pkl'])
    if not saved_files < set(os.listdir(data_dir)):
        train_data.to_pickle(os.path.join(data_dir, 'train.pkl'))
        test_data.to_pickle(os.path.join(data_dir, 'test.pkl'))
        logger.info("data saved")


def embed_data(data, embeddings):
    embedded = []
    for index, row in data.iterrows():
        if index % 10000 == 0:
            logger.debug(index)
        embedding = _embed(data['tokenized'], embeddings)
        embedded.append(embedding)
    data['embeddings'] = embedded


def _embed(tokens, embeddings):
    raise Exception()


def map_to_input_space(data, vocab, max_seq_len):
    logger.info("map to input space")
    X = np.zeros((len(data), max_seq_len), dtype=int)
    for index, row in data.iterrows():
        if index % 100000 == 0:
            logger.debug(index)
        # TODO: better handle OOV
        for ti, token in enumerate(row['tokenized']):
            if ti >= max_seq_len:
                break
            X[index][ti] = vocab.get(token, 0)
    return X


class LogReg(nn.Module):

    def __init__(self, input_size, num_classes, weights):
        super(LogReg, self).__init__()
        self.embed1 = nn.Embedding.from_pretrained(torch.from_numpy(weights))
        self.linear1 = nn.Linear(input_size, num_classes)
        self.nonlinear1 = F.log_softmax

    def forward(self, inputs):
        embed1 = self.embed1(inputs).to(torch.float)
        sum_embed1 = embed1.sum(dim=1)
        return self.nonlinear1(self.linear1(sum_embed1), dim=1)

    def predict(self, inputs):
        _, predictions = torch.max(self.predict_proba(inputs), 1)
        return predictions

    def predict_proba(self, inputs):
        return self.forward(inputs)


def batchify(l, batch_size):
    for offset in range(0, len(l), batch_size):
        yield l[offset:offset+batch_size]


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def train_cv(
        X, y, model, criterion, optimizer, num_epochs, batch_size,
        num_folds=5, metric='f1_score', patience=10, min_improvement=0.):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
    cv_scores = []
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        logger.info(f"fold index: {fold_idx}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        score = train(
            X_train, X_test, y_train, y_test, model, criterion, optimizer,
            num_epochs, batch_size, metric, patience)
        cv_scores.append(score)
    return cv_scores


def train(
        X_train, X_test, y_train, y_test, model, criterion, optimizer,
        num_epochs, batch_size, metric='f1_score', patience=10,
        min_improvement=0.):
    model_checkpoint_path = 'best_model.pt'
    X_train, X_test = torch.from_numpy(X_train), torch.from_numpy(X_test)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)
    scores = []
    iter = 0
    model.apply(weight_reset)
    model.train()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(num_epochs):
        logger.info(f"EPOCH: {epoch}")
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
    predictions = predict(inputs, model, batch_size)
    if metric == 'f1_score':
        return f1_score(targets, predictions)
    else:
        raise Exception(f"unknown metric {metric}")


def predict(inputs, model, batch_size=1000):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictions = []
    for input_batch in batchify(inputs, batch_size):
        input_batch = input_batch.to(device)
        predictions.extend(model.predict(input_batch).cpu().numpy())
    return predictions


def early_stopping(scores, patience, min_improvement=0.):
    if len(scores) <= patience:
        return False
    best_score_outside_patience = max(scores[:-patience])
    best_score_in_patience = max(scores[-patience:])
    improvement = best_score_in_patience - best_score_outside_patience
    return improvement < min_improvement


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def downsample(data, downsample_ratio=None, max_imbalance_ratio=None):
    if max_imbalance_ratio:
        counts_per_class = dict(data.groupby('target')['qid'].count())
        downsampled_dfs = {}
        min_class_count = min(counts_per_class.values(), default=0.)
        min_class_count = int(min_class_count * max_imbalance_ratio)
        logger.info(f"Downsampling current counts {counts_per_class}, "
                    f"keeping a max ratio of {max_imbalance_ratio} between "
                    f"classes")
        for class_, count in counts_per_class.items():
            class_df = data.loc[data['target'] == class_]
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
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def build_vocab(train_data, test_data):
    """ we also add test_data vocab as they may be useful at inference time
    """
    vocab = set()
    for tokenized in train_data['tokenized'].values:
        vocab.update(tokenized)
    for tokenized in test_data['tokenized'].values:
        vocab.update(tokenized)
    return vocab


if __name__ == '__main__':
    SUBMIT = True
    FULL_SIZE_FOR_SUBMIT = False
    SEED = 42
    LOWER = True
    DOWNSAMPLE = 0.  # None, 0 or 1 to ignore
    MAX_IMBALANCE_RATIO = 3.
    MAX_SEQ_LEN = 50
    BATCH_SIZE = 1000
    LEARNING_RATE = 2E-4
    WEIGHT_DECAY = 1E-6
    MOMENTUM = 0.5
    VOCAB_SIZE = 1E6
    EMBEDDING_MODEL = 'glove.840B.300d/glove.840B.300d.txt'
    SPACY_MODEL = 'en_core_web_sm'
    NUM_EPOCHS = 200
    PATIENCE = 40
    MIN_IMPROVEMENT = 1E-3
    set_seeds(SEED)
    logger.info(f"torch.initial_seed(): {torch.initial_seed()}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Use device {device}")
    data_dir = get_data_dir()
    embed_dir = os.path.join(data_dir, 'embeddings/')
    log_dir = get_log_dir()
    train_data, test_data = load_data(data_dir)
    train_data = downsample(train_data, DOWNSAMPLE, MAX_IMBALANCE_RATIO)
    nlp = spacy.load(SPACY_MODEL)
    tokenizer = Tokenizer(nlp.vocab)
    preprocess_data(train_data, tokenizer, LOWER)
    preprocess_data(test_data, tokenizer, LOWER)
    vocab_filter = build_vocab(train_data, test_data)
    # if not DOWNSAMPLE:
    #     save_data(data_dir, train_data, test_data)
    weights, vocab = load_embeddings(
        embed_dir, model=EMBEDDING_MODEL, top_n=VOCAB_SIZE,
        vocab_filter=vocab_filter)
    emb_size = weights.shape[1]
    num_classes = len(set(train_data['target']))
    train_X = map_to_input_space(train_data, vocab, MAX_SEQ_LEN)
    train_y = train_data['target'].values
    model = LogReg(
        input_size=emb_size, num_classes=num_classes, weights=weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY)
    # TODO: use Dataloader
    # TODO: use vocab from train and test set to select embeddings to load
    if not SUBMIT:
        cv_scores = train_cv(
            train_X, train_y, model, criterion, optimizer, NUM_EPOCHS,
            BATCH_SIZE, patience=PATIENCE, min_improvement=MIN_IMPROVEMENT)
        # analysis = run_eda(train_data, test_data)
    else:
        if DOWNSAMPLE and FULL_SIZE_FOR_SUBMIT:  # recompute train data
            train_data, test_data = load_data(data_dir)
            preprocess_data(train_data, tokenizer, LOWER)
            train_X = map_to_input_space(train_data, vocab, MAX_SEQ_LEN)
            train_y = train_data['target'].values
        logger.info("final training")
        scores = train(
            train_X, train_X, train_y, train_y, model, criterion, optimizer,
            NUM_EPOCHS, BATCH_SIZE, patience=PATIENCE,
            min_improvement=MIN_IMPROVEMENT)
        logger.info("preprocess and predict target on test set")
        test_X = map_to_input_space(test_data, vocab, MAX_SEQ_LEN)
        test_X = torch.from_numpy(test_X)
        test_data['prediction'] = predict(test_X, model)
        test_data[['qid', 'prediction']].to_csv('submission.csv', index=False)