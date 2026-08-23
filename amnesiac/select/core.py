"""Pure vector mathematics for document selection."""

from collections.abc import Mapping, Sequence

import numpy as np

from amnesiac.exceptions import ConfigurationError


def _dedup(
    subset_indices: list[int],
    X_full: np.ndarray,
    texts: Sequence[str],
    threshold: float,
) -> list[int]:
    if not subset_indices:
        return []

    X_sub = X_full[subset_indices]
    sim = X_sub @ X_sub.T
    np.fill_diagonal(sim, 0.0)

    dropped: set[int] = set()
    for i in range(len(subset_indices)):
        if i in dropped:
            continue
        for j in range(i + 1, len(subset_indices)):
            if j in dropped:
                continue
            if sim[i, j] < threshold:
                continue

            idx_i = subset_indices[i]
            idx_j = subset_indices[j]
            len_i = len(texts[idx_i])
            len_j = len(texts[idx_j])

            if len_i <= len_j:
                dropped.add(j)
            else:
                dropped.add(i)
                break

    return [subset_indices[i] for i in range(len(subset_indices)) if i not in dropped]


def select_by_axis(
    X: np.ndarray,
    queries: Mapping[str, np.ndarray],
    *,
    texts: Sequence[str],
    top_k: int,
    dedup_threshold: float,
    order_by: Sequence[float] | None = None,
) -> dict[str, list[int]]:
    """Select document-row indices independently for each query axis."""
    n = len(X)
    if len(texts) != n:
        raise ConfigurationError(f"len(texts)={len(texts)} does not match len(X)={n}")
    if order_by is not None and len(order_by) != n:
        raise ConfigurationError(f"len(order_by)={len(order_by)} does not match len(X)={n}")

    if n == 0:
        return {axis: [] for axis in queries}

    X_norm = X.copy()
    for vec in X_norm:
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

    result: dict[str, list[int]] = {}
    for axis, query in queries.items():
        q_mat = np.asarray(query)
        if q_mat.ndim == 1:
            q_vec = q_mat.copy()
        elif q_mat.ndim == 2:
            q_vec = q_mat.mean(axis=0)
        else:
            raise ConfigurationError(
                f"query for axis {axis!r} must have 1 or 2 dimensions, got {q_mat.ndim}"
            )

        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        sims = X_norm @ q_vec

        k = min(top_k, n)
        if k == n:
            top_idx = np.argsort(-sims)
        else:
            part = np.argpartition(sims, -k)[-k:]
            top_idx = part[np.argsort(-sims[part])]

        kept_idx = _dedup(top_idx.tolist(), X_norm, texts, dedup_threshold)
        if order_by is not None:
            kept_idx.sort(key=lambda idx: order_by[idx])
        result[axis] = kept_idx

    return result
