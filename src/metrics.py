"""Shared ranking calculations; each algorithm supplies its own population."""
from math import log2
from numbers import Integral

import pandas as pd


def evaluate_predictions(predictions: dict[str, list[str]], targets: pd.DataFrame, k: int = 10) -> dict:
    """Return per-user Recall/NDCG@k and macro means for users with targets.

    Missing/empty predictions count as zero. Empty target populations have no
    score. Never drop an eligible user because recommendations are exhausted.
    """
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive integer")
    rows = []
    for user_id, group in targets.groupby("user_id", sort=True):
        relevant = set(group.parent_asin)
        recommended = list(predictions.get(user_id, []))
        if len(set(recommended)) != len(recommended):
            raise ValueError("Recommended product IDs must be unique")
        top = recommended[:k]
        hits = [rank for rank, product in enumerate(top, start=1) if product in relevant]
        ideal = sum(1 / log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
        rows.append({"user_id": user_id, "targets": len(relevant), "returned": len(top),
                     "recall": len(hits) / len(relevant),
                     "ndcg": sum(1 / log2(rank + 1) for rank in hits) / ideal})
    per_user = pd.DataFrame(rows, columns=["user_id", "targets", "returned", "recall", "ndcg"])
    return {"summary": {"k": int(k), "users": len(rows),
                        "recall": float(per_user.recall.mean()) if rows else None,
                        "ndcg": float(per_user.ndcg.mean()) if rows else None},
            "per_user": per_user}
