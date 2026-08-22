"""Configuration for two-stage summarization."""

from pydantic import BaseModel, ConfigDict, Field


class SummarizeConfig(BaseModel):
    """Runtime settings for the summarization pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float = 0.3
    concurrency: int = 5
    max_attempts: int = Field(3, ge=1)
    retry_delays: tuple[float, ...] = Field((5.0, 15.0, 45.0), min_length=1)
    max_failed_axes: int = 2
    failed_axis_placeholder: str = "(нет данных по оси из-за ошибки провайдера)"
