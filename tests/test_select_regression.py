"""Fixture-based regression tests for document selection."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from amnesiac.select import select_by_axis

FIXTURE_DIR = Path(__file__).parent / "fixtures"
NPZ_PATH = FIXTURE_DIR / "slice_ru_macro_v1.npz"
JSON_PATH = FIXTURE_DIR / "slice_ru_macro_v1.json"


def load_fixture() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with JSON_PATH.open(encoding="utf-8") as fixture_file:
        metadata = json.load(fixture_file)
    with np.load(NPZ_PATH) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return metadata, arrays


def test_npz_sha256_matches_json_provenance() -> None:
    metadata, _ = load_fixture()

    with NPZ_PATH.open("rb") as fixture_file:
        actual = hashlib.sha256(fixture_file.read()).hexdigest()

    assert actual == metadata["origin"]["npz_sha256"]


@pytest.mark.parametrize("parameter_set", ["k_lt_n", "k_eq_n"])
def test_selection_matches_ordered_fixture_doc_ids(parameter_set: str) -> None:
    metadata, arrays = load_fixture()
    axes = metadata["params"]["axes"]
    queries = {axis: arrays[f"q_{axis}"] for axis in axes}
    expected_case = metadata["expected"][parameter_set]

    result = select_by_axis(
        arrays["X"],
        queries,
        texts=metadata["texts"],
        top_k=expected_case["top_k"],
        dedup_threshold=metadata["params"]["dedup_threshold"],
        order_by=arrays["order_by"],
    )

    assert list(result) == axes
    for axis in axes:
        actual = [metadata["doc_ids"][row_index] for row_index in result[axis]]
        expected = expected_case["selected"][axis]
        assert actual == expected


def test_float64_fixture_input_runs_without_dtype_coercion() -> None:
    metadata, arrays = load_fixture()
    axes = metadata["params"]["axes"]
    queries = {axis: arrays[f"q_{axis}"].astype(np.float64) for axis in axes}

    result = select_by_axis(
        arrays["X"].astype(np.float64),
        queries,
        texts=metadata["texts"],
        top_k=metadata["expected"]["k_lt_n"]["top_k"],
        dedup_threshold=metadata["params"]["dedup_threshold"],
        order_by=arrays["order_by"],
    )

    assert list(result) == axes
    assert all(isinstance(result[axis], list) for axis in axes)


def test_fixture_selection_is_deterministic_across_three_calls() -> None:
    metadata, arrays = load_fixture()
    axes = metadata["params"]["axes"]
    queries = {axis: arrays[f"q_{axis}"] for axis in axes}
    expected_case = metadata["expected"]["k_lt_n"]
    kwargs = {
        "texts": metadata["texts"],
        "top_k": expected_case["top_k"],
        "dedup_threshold": metadata["params"]["dedup_threshold"],
        "order_by": arrays["order_by"],
    }

    results = [select_by_axis(arrays["X"], queries, **kwargs) for _ in range(3)]

    assert results[0] == results[1] == results[2]
