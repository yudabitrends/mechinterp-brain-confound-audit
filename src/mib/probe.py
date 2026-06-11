"""Phase 2: probe SAE features for scanner(confound)-vs-disease information.

We apply a trained SAE (trained on all tokens) to the decision-relevant CLS activations,
then ask -- at the *feature* level -- how well the dictionary linearly separates diagnosis
vs scanner/site. Reproducing the behavioral scanner>>disease gap here, and showing the
two information types live in (near-)disjoint feature sets, is the mechanistic analog of
the behavioral Jaccard~0.04 finding and the entry point to Phase-3 activation patching.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, label_binarize


@torch.no_grad()
def encode_features(sae, X) -> np.ndarray:
    """Map (N, d_in) activations -> (N, d_sae) SAE feature activations."""
    return sae.encode(torch.as_tensor(np.asarray(X)).float()).cpu().numpy()


def linear_probe_auc(X, y, multiclass: bool = False, cv: int = 5, seed: int = 0,
                     C: float = 1.0) -> float:
    """Cross-validated linear-probe AUC. Binary -> ROC-AUC; multiclass -> macro OVR AUC.
    Features are standardized; LR is the standard linear readout for 'is this information
    linearly decodable from the representation?'."""
    y = np.asarray(y)
    Xs = StandardScaler().fit_transform(np.asarray(X))
    skf = StratifiedKFold(cv, shuffle=True, random_state=seed)
    if multiclass:
        clf = LogisticRegression(max_iter=2000, C=C)
        proba = cross_val_predict(clf, Xs, y, cv=skf, method="predict_proba")
        classes = np.unique(y)
        Y = label_binarize(y, classes=classes)
        return float(roc_auc_score(Y, proba, average="macro", multi_class="ovr"))
    clf = LogisticRegression(max_iter=2000, C=C)
    proba = cross_val_predict(clf, Xs, y, cv=skf, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def per_feature_auc(X, y_bin) -> np.ndarray:
    """Univariate ROC-AUC of each feature predicting a binary label (0.5 = uninformative).
    Constant/dead features -> 0.5. Used to *label* features as scanner- vs disease-tuned."""
    X = np.asarray(X)
    y_bin = np.asarray(y_bin)
    aucs = np.full(X.shape[1], 0.5)
    for j in range(X.shape[1]):
        col = X[:, j]
        if col.std() < 1e-12:
            continue
        try:
            aucs[j] = roc_auc_score(y_bin, col)
        except ValueError:
            pass
    return aucs


def top_feature_jaccard(auc_a, auc_b, top_frac: float = 0.05) -> dict:
    """Overlap of the most-informative feature sets for two binary targets.
    Informativeness = |AUC - 0.5| (direction-agnostic). Low Jaccard => disease and
    scanner information occupy (near-)disjoint dictionary features -- the mechanistic
    analog of the behavioral feature-disjointness finding."""
    auc_a, auc_b = np.asarray(auc_a), np.asarray(auc_b)
    n = max(1, int(len(auc_a) * top_frac))
    A = set(np.argsort(-np.abs(auc_a - 0.5))[:n].tolist())
    B = set(np.argsort(-np.abs(auc_b - 0.5))[:n].tolist())
    inter = len(A & B)
    return {"jaccard": inter / len(A | B), "n_top": n, "overlap": inter}
