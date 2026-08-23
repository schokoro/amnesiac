"""Synthetic tests for pure vector document selection."""

import numpy as np
import pytest

from amnesiac import ConfigurationError
from amnesiac.select import select_by_axis


def test_top_k_returns_hand_computed_similarity_order_without_modifying_x() -> None:
    X = np.array([[3.0, 0.0], [1.0, 1.0], [0.0, 2.0], [-2.0, 0.0]])
    original = X.copy()

    result = select_by_axis(
        X,
        {"axis": np.array([1.0, 0.0])},
        texts=["a", "b", "c", "d"],
        top_k=3,
        dedup_threshold=1.1,
    )

    assert result == {"axis": [0, 1, 2]}
    assert np.array_equal(X, original)


def test_pre_normalized_and_unnormalized_x_give_same_result() -> None:
    X = np.array([[3.0, 0.0], [1.0, 1.0], [0.0, 2.0], [-2.0, 0.0]])
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    kwargs = {"texts": ["a", "b", "c", "d"], "top_k": 3, "dedup_threshold": 1.1}

    unnormalized = select_by_axis(X, {"axis": np.array([1.0, 0.0])}, **kwargs)
    normalized = select_by_axis(X_normalized, {"axis": np.array([1.0, 0.0])}, **kwargs)

    assert unnormalized == normalized == {"axis": [0, 1, 2]}


def test_zero_document_row_does_not_produce_nan_or_raise() -> None:
    with np.errstate(invalid="raise", divide="raise"):
        result = select_by_axis(
            np.array([[0.0, 0.0], [2.0, 0.0]]),
            {"axis": np.array([1.0, 0.0])},
            texts=["zero", "positive"],
            top_k=2,
            dedup_threshold=1.1,
        )

    assert result == {"axis": [1, 0]}


def test_one_dimensional_and_single_row_queries_are_equivalent() -> None:
    X = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    queries = {
        "one_dimensional": np.array([1.0, 0.0]),
        "single_row": np.array([[1.0, 0.0]]),
    }

    result = select_by_axis(
        X,
        queries,
        texts=["a", "b", "c"],
        top_k=2,
        dedup_threshold=1.1,
    )

    assert result["one_dimensional"] == result["single_row"] == [0, 1]


def test_multirow_query_matches_hand_computed_normalized_mean() -> None:
    X = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    multirow_query = np.array([[2.0, 0.0], [0.0, 2.0]])
    normalized_mean = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])
    kwargs = {"texts": ["a", "b", "c", "d"], "top_k": 1, "dedup_threshold": 1.1}

    from_multirow = select_by_axis(X, {"axis": multirow_query}, **kwargs)
    from_hand_computed_mean = select_by_axis(X, {"axis": normalized_mean}, **kwargs)

    assert from_multirow == from_hand_computed_mean == {"axis": [0]}


def test_zero_mean_query_does_not_produce_nan_or_raise() -> None:
    with np.errstate(invalid="raise", divide="raise"):
        result = select_by_axis(
            np.array([[1.0, 0.0]]),
            {"axis": np.array([[1.0, 0.0], [-1.0, 0.0]])},
            texts=["only"],
            top_k=1,
            dedup_threshold=1.1,
        )

    assert result == {"axis": [0]}


def test_top_k_greater_than_n_returns_every_row() -> None:
    result = select_by_axis(
        np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]),
        {"axis": np.array([1.0, 0.0])},
        texts=["a", "b", "c"],
        top_k=10,
        dedup_threshold=1.1,
    )

    assert result == {"axis": [0, 1, 2]}


def test_partition_top_k_is_prefix_of_full_sort_without_ties_or_dedup() -> None:
    X = np.array([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8], [0.0, 1.0]])
    query = {"axis": np.array([1.0, 0.0])}
    kwargs = {"texts": ["a", "b", "c", "d"], "dedup_threshold": 1.1}

    full = select_by_axis(X, query, top_k=4, **kwargs)
    partial = select_by_axis(X, query, top_k=2, **kwargs)

    assert full == {"axis": [0, 1, 2, 3]}
    assert partial == {"axis": full["axis"][:2]}


def test_dedup_keeps_shorter_text_even_when_it_is_less_similar() -> None:
    result = select_by_axis(
        np.array([[1.0, 0.0], [0.8, 0.6]]),
        {"axis": np.array([1.0, 0.0])},
        texts=["the much longer document", "short"],
        top_k=2,
        dedup_threshold=0.75,
    )

    assert result == {"axis": [1]}
    assert len(result["axis"]) < 2


def test_dedup_three_mutually_similar_documents_follows_asymmetric_loop() -> None:
    X = np.array(
        [
            [1.0, 0.0],
            [0.984807753, 0.173648178],
            [0.939692621, 0.342020143],
        ]
    )

    result = select_by_axis(
        X,
        {"axis": np.array([1.0, 0.0])},
        texts=["tenletters", "short", "twenty characters...."],
        top_k=3,
        dedup_threshold=0.9,
    )

    assert result == {"axis": [1]}


def test_dedup_break_preserves_document_beyond_dropped_bridge() -> None:
    angles = np.deg2rad([0.0, -30.0, 32.0])
    X = np.column_stack((np.cos(angles), np.sin(angles)))

    result = select_by_axis(
        X,
        {"axis": np.array([1.0, 0.0])},
        texts=["0123456789", "01234", "01234567890123456789"],
        top_k=3,
        dedup_threshold=0.82,
    )

    assert result == {"axis": [1, 2]}


def test_dedup_threshold_is_inclusive() -> None:
    result = select_by_axis(
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        {"axis": np.array([1.0, 0.0])},
        texts=["a", "longer"],
        top_k=2,
        dedup_threshold=1.0,
    )

    assert result == {"axis": [0]}


def test_order_by_is_stable_over_descending_similarity_order() -> None:
    result = select_by_axis(
        np.array([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8]]),
        {"axis": np.array([1.0, 0.0])},
        texts=["a", "b", "c"],
        top_k=3,
        dedup_threshold=1.1,
        order_by=[1.0, 1.0, 0.0],
    )

    assert result == {"axis": [2, 0, 1]}


def test_order_by_none_preserves_descending_similarity_after_dedup() -> None:
    result = select_by_axis(
        np.array([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8]]),
        {"axis": np.array([1.0, 0.0])},
        texts=["a", "b", "c"],
        top_k=3,
        dedup_threshold=1.1,
        order_by=None,
    )

    assert result == {"axis": [0, 1, 2]}


def test_empty_x_returns_all_axis_keys_in_query_order() -> None:
    result = select_by_axis(
        np.empty((0, 2)),
        {"zeta": np.array([1.0, 0.0]), "alpha": np.array([0.0, 1.0])},
        texts=[],
        top_k=3,
        dedup_threshold=0.9,
    )

    assert result == {"zeta": [], "alpha": []}
    assert list(result) == ["zeta", "alpha"]


@pytest.mark.parametrize(
    ("texts", "order_by", "message"),
    [
        (["one"], None, "len(texts)=1 does not match len(X)=2"),
        (["one", "two"], [0.0], "len(order_by)=1 does not match len(X)=2"),
    ],
)
def test_parallel_sequence_length_mismatch_raises_configuration_error_first(
    texts: list[str], order_by: list[float] | None, message: str
) -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        select_by_axis(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            {"invalid_if_reached": np.zeros((1, 1, 1))},
            texts=texts,
            top_k=2,
            dedup_threshold=1.1,
            order_by=order_by,
        )
    assert str(exc_info.value) == message


def test_query_with_unsupported_dimensions_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="must have 1 or 2 dimensions, got 3"):
        select_by_axis(
            np.array([[1.0, 0.0]]),
            {"axis": np.zeros((1, 1, 2))},
            texts=["one"],
            top_k=1,
            dedup_threshold=1.1,
        )


def test_float64_input_runs_without_dtype_coercion() -> None:
    X = np.array([[1.0, 0.0], [0.8, 0.6]])

    result = select_by_axis(
        X,
        {"axis": np.array([1.0, 0.0])},
        texts=["a", "b"],
        top_k=2,
        dedup_threshold=1.1,
    )

    assert result == {"axis": [0, 1]}


def test_repeated_calls_are_deterministic() -> None:
    X = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    query = {"axis": np.array([1.0, 0.0])}
    kwargs = {"texts": ["a", "b", "c"], "top_k": 2, "dedup_threshold": 1.1}

    results = [select_by_axis(X, query, **kwargs) for _ in range(3)]

    assert results[0] == results[1] == results[2] == {"axis": [0, 1]}
