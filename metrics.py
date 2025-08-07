import numpy as np
from typing import Iterable

def recall_at_k(
    y_true: Iterable[int], 
    y_pred_ranked: Iterable[int], 
    k: int
    ) -> float:
    """
    y_pred_ranked: list of predicted items in ranked order (to play with different k)
    y_true: list/set of actual items in basket
    k: cutoff
    """
    pred_k = set(y_pred_ranked[:k])
    actual = set(y_true)
    return len(pred_k & actual) / len(actual) if len(actual) > 0 else 0