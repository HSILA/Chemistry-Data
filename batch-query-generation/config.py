from pydantic import BaseModel, Field, field_validator
import os
from typing import Optional


class BaseQueryGenerationConfig(BaseModel):
    data_path: str = Field(
        ..., description="path to a csv file or a huggingface dataset"
    )
    hf_config_name: Optional[str] = Field(
        default=None, description="Huggingface dataset config name."
    )
    root_dir: str = Field(
        ...,
        description="Root directory for the job. Generated requests, batch job details and retrieved responses are saved here.",
    )
    model: str = Field(..., description="The model name, e.g. 'gpt-4o-mini'.")
    text_column: str = Field(
        ..., description="The name of the column containing text for query generation."
    )
    id_columns: list[str] = Field(
        ..., description="List of column names to identify a row in the dataset."
    )
    prompt_template: str = Field(
        ..., description="Template query string for generating requests."
    )


    @field_validator("data_path")
    def validate_data_path(cls, v):
        if os.path.exists(v) or v.lower().endswith(".csv"):
            return v
        if len(v.split("/")) == 2:
            return v
        raise ValueError(
            "data_path must be either a local CSV file path (or URL ending with '.csv') or a Huggingface dataset identifier in the format 'user/dataset'."
        )


class BatchQueryGenerationConfig(BaseQueryGenerationConfig):
    params: dict = Field(
        ...,
        description="Dictionary of OpenAI API call parameters (e.g. `temperature`, `reasoning_effort`).",
    )
    shard_size: int = Field(
        50000,
        description="Number of records per shard for generating batch requests. Default is 50,000 (max for OpenAI).",
    )