import pytest
from pydantic import ValidationError

from amnesiac.summarize.config import SummarizeConfig


def test_defaults() -> None:
    config = SummarizeConfig()

    assert config.temperature == 0.3
    assert config.concurrency == 5
    assert config.max_attempts == 3
    assert config.retry_delays == (5.0, 15.0, 45.0)
    assert config.max_failed_axes == 2
    assert config.failed_axis_placeholder == "(нет данных по оси из-за ошибки провайдера)"


def test_config_is_frozen() -> None:
    config = SummarizeConfig()

    with pytest.raises(ValidationError):
        config.temperature = 0.5


def test_config_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        SummarizeConfig(unknown=True)


@pytest.mark.parametrize("kwargs", [{"retry_delays": ()}, {"max_attempts": 0}])
def test_invalid_retry_configuration_fails_at_construction(kwargs: object) -> None:
    with pytest.raises(ValidationError):
        SummarizeConfig(**kwargs)


def test_model_fields_set_tracks_explicit_default_value() -> None:
    implicit = SummarizeConfig()
    explicit = SummarizeConfig(concurrency=5)

    assert implicit.concurrency == explicit.concurrency
    assert "concurrency" not in implicit.model_fields_set
    assert "concurrency" in explicit.model_fields_set
