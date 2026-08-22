"""Tests for shared data models."""

import pytest
from pydantic import ValidationError

from amnesiac.types import Doc, Usage, _add_usage


def test_doc_is_frozen() -> None:
    doc = Doc(text="text", channel="channel", day_number=1)

    with pytest.raises(ValidationError):
        doc.text = "changed"


def test_doc_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Doc(text="text", channel="channel", day_number=1, unknown="value")


def test_doc_rejects_wrong_type() -> None:
    with pytest.raises(ValidationError):
        Doc(text="text", channel="channel", day_number="abc")


@pytest.mark.parametrize(
    ("doc_id", "expected_type"),
    [(1, int), ("1", str), (None, type(None))],
)
def test_doc_id_preserves_supported_types(
    doc_id: str | int | None, expected_type: type[object]
) -> None:
    doc = Doc(text="text", channel="channel", day_number=1, doc_id=doc_id)

    assert doc.doc_id == doc_id
    assert type(doc.doc_id) is expected_type


@pytest.mark.parametrize("day_number", [0, -1])
def test_day_number_has_no_range_constraint(day_number: int) -> None:
    doc = Doc(text="text", channel="channel", day_number=day_number)

    assert doc.day_number == day_number


def test_usage_fields_default_to_zero() -> None:
    usage = Usage()

    assert usage.model_dump() == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }


def test_add_usage_mutates_target_and_sums_every_field() -> None:
    target = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3, calls=4)
    other = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30, calls=40)
    original_target = target

    result = _add_usage(target, other)

    assert result is None
    assert target is original_target
    assert target == Usage(prompt_tokens=11, completion_tokens=22, total_tokens=33, calls=44)
