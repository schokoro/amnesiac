import json
from pathlib import Path
from typing import Any

from amnesiac.summarize.prompts import RU_MACRO_V1
from amnesiac.types import Doc


def load_fixture() -> dict[str, Any]:
    fixture_path = Path(__file__).parent / "fixtures" / "prompts_ru_macro_v1.json"
    with fixture_path.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def test_ru_macro_v1_rendering_regression() -> None:
    fixture = load_fixture()
    inputs = fixture["input"]
    expected = fixture["expected"]
    prompts = RU_MACRO_V1.bind(horizon_days=inputs["horizon_days"])

    assert inputs["axis_summaries"][-1]["summary"] == inputs["failed_axis_placeholder"]

    for axis_input in inputs["axes"]:
        axis = axis_input["axis"]
        docs = [Doc(**doc) for doc in axis_input["docs"]]
        rendered_docs = prompts._render_docs(docs)

        assert prompts._render_axis_system(axis=axis) == expected["axis_system"][axis]
        assert (
            prompts._render_axis_user(axis=axis, docs=rendered_docs) == expected["axis_user"][axis]
        )

    assert prompts._render_meta_system() == expected["meta_system"]

    axis_summaries = {item["axis"]: item["summary"] for item in inputs["axis_summaries"]}
    axis_blocks = prompts._render_axis_blocks(axis_summaries)
    assert prompts._render_meta_user(axis_blocks=axis_blocks) == expected["meta_user"]
