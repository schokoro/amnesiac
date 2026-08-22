from typing import Any

import pytest
from pydantic import ValidationError

from amnesiac.exceptions import PromptRenderError
from amnesiac.summarize.prompts import PromptPack
from amnesiac.types import Doc


def make_pack(**kwargs: Any) -> PromptPack:
    values = {
        "name": "test",
        "axis_system": "axis={axis}; shared={shared}",
        "axis_user": "axis={axis}; docs={docs}; shared={shared}",
        "meta_system": "shared={shared}",
        "meta_user": "blocks={axis_blocks}; shared={shared}",
        "doc_template": "{day_number}:{channel}:{text}:{shared}",
        "axis_block_template": "{axis}:{summary}:{shared}",
    }
    values.update(kwargs)
    return PromptPack(**values)


def test_bind_returns_new_pack_without_mutating_receiver() -> None:
    pack = make_pack(params={"earlier": "value"})

    bound = pack.bind(shared="bound")

    assert bound is not pack
    assert pack.params == {"earlier": "value"}
    assert bound.params == {"earlier": "value", "shared": "bound"}


def test_bind_layers_parameters() -> None:
    first = make_pack().bind(first=1, shared="old")
    second = first.bind(second=2, shared="new")

    assert second.params == {"first": 1, "shared": "new", "second": 2}
    assert first.params == {"first": 1, "shared": "old"}


def test_bound_key_is_visible_to_every_template() -> None:
    pack = make_pack().bind(shared="yes")
    docs = pack._render_docs([Doc(day_number=1, channel="channel", text="text")])
    blocks = pack._render_axis_blocks({"axis": "summary"})

    assert pack._render_axis_system(axis="axis") == "axis=axis; shared=yes"
    assert pack._render_axis_user(axis="axis", docs=docs).endswith("shared=yes")
    assert pack._render_meta_system() == "shared=yes"
    assert pack._render_meta_user(axis_blocks=blocks).endswith("shared=yes")
    assert docs.endswith(":yes")
    assert blocks.endswith(":yes")


def test_runtime_keys_take_precedence_over_bound_parameters() -> None:
    pack = make_pack().bind(axis="bound", shared="yes")

    assert pack._render_axis_system(axis="runtime") == "axis=runtime; shared=yes"


def test_missing_key_raises_prompt_render_error_with_context() -> None:
    pack = make_pack()

    with pytest.raises(PromptRenderError, match=r"meta_system.*shared") as exc_info:
        pack._render_meta_system()

    assert not isinstance(exc_info.value, KeyError)


def test_prompt_pack_is_frozen() -> None:
    pack = make_pack()

    with pytest.raises(ValidationError):
        pack.name = "changed"


def test_prompt_pack_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        make_pack(unknown=True)


def test_render_docs_preserves_input_order() -> None:
    pack = make_pack(doc_template="{day_number}:{text}", params={"shared": "unused"})
    docs = [
        Doc(day_number=9, channel="later", text="first"),
        Doc(day_number=1, channel="earlier", text="second"),
    ]

    assert pack._render_docs(docs) == "9:first\n1:second"
