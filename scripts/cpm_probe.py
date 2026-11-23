"""Train-label-only GNB adaptation of Luan et al. (NeurIPS 2023) CPM.

The published aggregation/preprocessing is made explicit in the protocol.
The API accepts labels ONLY for the supplied training-node feature rows.
Repeated overlapping holdouts yield a descriptive score, not independent tests.
"""
from __future__ import annotations
import hashlib
import warnings
import numpy as np
from scipy import sparse, stats
from sklearn.naive_bayes import GaussianNB


def seed_for(dataset, split_id):
    return int.from_bytes(hashlib.sha256((dataset + ':' + split_id).encode()).digest()[:8], 'little')


def aggregation_operator(edge_index, n):
    edges = np.asarray(edge_index)
    if edges.ndim != 2 or edges.shape[0] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError('integer 2-by-E edge index required')
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError('edge outside graph')
    a = sparse.csr_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(n, n))
    a = a.maximum(a.T)
    a.setdiag(1)
    a.data[:] = 1
    return sparse.diags(1 / np.asarray(a.sum(axis=1)).ravel()) @ a


def directional_score(raw_accuracy, graph_accuracy):
    x, g = np.asarray(raw_accuracy, float), np.asarray(graph_accuracy, float)
    if x.shape != g.shape or x.ndim != 1 or len(x) < 2 or not np.isfinite([x, g]).all():
        raise ValueError('paired finite replicate arrays required')
    wins = float(np.mean(g > x))
    if np.ptp(x) == 0 and np.ptp(g) == 0:
        return 0.5 if x[0] == g[0] else float(wins > 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        p = float(stats.ttest_ind(x, g, equal_var=False).pvalue)
    if not np.isfinite(p):
        raise ValueError('undefined nondegenerate Welch score')
    return p / 2 if wins <= 0.5 else 1 - p / 2


def probe_training_rows(raw_train, aggregated_train, train_labels, *, seed, repeats=100, cap=500):
    """No full labels, validation labels, test labels or candidate outcomes accepted."""
    x, g, y = np.asarray(raw_train), np.asarray(aggregated_train), np.asarray(train_labels)
    if x.ndim != 2 or x.shape != g.shape or y.shape != (len(x),):
        raise ValueError('only aligned outer-training rows/labels accepted')
    if not np.isfinite(x).all() or not np.isfinite(g).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite probe input')
    classes = np.unique(y)
    if len(classes) < 2 or repeats < 2 or cap < len(classes):
        raise ValueError('insufficient classes, repeats or sample cap')
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in classes]
    xs, gs, sizes, partitions = [], [], [], hashlib.sha256()
    for _ in range(repeats):
        if len(y) <= cap:
            sample = np.arange(len(y))
        else:
            quota = int(round(cap / len(classes)))
            sample = np.sort(np.concatenate([rng.permutation(ids)[:quota] for ids in groups]))
        ys = y[sample]
        quota = int(round(0.6 * len(sample) / len(classes)))
        fit_parts, hold_parts = [], []
        for c in classes:
            indices = rng.permutation(np.flatnonzero(ys == c))
            fit_parts.append(indices[:quota])
            hold_parts.append(indices[quota:])
        fit, hold = np.concatenate(fit_parts), np.concatenate(hold_parts)
        if not len(fit) or not len(hold) or len(np.unique(ys[fit])) != len(classes):
            raise ValueError('empty inner partition or missing fit class')
        partitions.update(np.asarray([len(sample), len(fit), len(hold)], '<i8').tobytes())
        for indices in (sample, fit, hold):
            partitions.update(np.asarray(indices, '<i8').tobytes())
        accuracies = []
        for features in (x, g):
            a, b = features[sample[fit]].astype(np.float64), features[sample[hold]].astype(np.float64)
            # GaussianNB cannot define an all-constant zero-variance likelihood.
            # Equal likelihoods reduce to empirical class-prior prediction.
            if float(np.var(a, axis=0).max()) == 0:
                cls, counts = np.unique(ys[fit], return_counts=True)
                predictions = np.full(len(hold), cls[np.argmax(counts)])
            else:
                model = GaussianNB(var_smoothing=1e-9).fit(a, ys[fit])
                predictions = model.predict(b)
            accuracies.append(float(np.mean(predictions == ys[hold])))
        xs.append(accuracies[0])
        gs.append(accuracies[1])
        sizes.append([len(sample), len(fit), len(hold)])
    return {'cpm_gnb': directional_score(xs, gs), 'probe_gap': float(np.mean(np.asarray(gs) - xs)),
            'raw_accuracy': xs, 'graph_accuracy': gs, 'partition_sizes': sizes,
            'partition_sha256': partitions.hexdigest(), 'resampling_seed': seed, 'outer_train_count': len(y),
            'strict_graph_win_fraction': float(np.mean(np.asarray(gs) > xs))}
