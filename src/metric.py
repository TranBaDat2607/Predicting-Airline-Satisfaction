"""Competition metric: ROC AUC on the predicted probability of `satisfaction`."""

import numpy as np
from sklearn.metrics import roc_auc_score


def score(y_true, y_pred) -> float:
    return float(roc_auc_score(np.asarray(y_true).astype(int), np.asarray(y_pred, dtype=float)))
