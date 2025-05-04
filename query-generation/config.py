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


class BedrockQueryGenerationConfig(BaseQueryGenerationConfig):
    max_completion_tokens: int = Field(
        1024,
        description="Maximum number of tokens allowed in the model's completion output.",
    )
    temperature: float = Field(
        0.0,
        description="Sampling temperature for the Bedrock model. Controls randomness of outputs.",
    )
    num_workers: int = Field(
        4,
        description="Number of concurrent workers to use for batch/concurrent processing.",
    )
    cooldown: float = Field(
        0.5,
        description="Cooldown time in seconds between Bedrock API calls to avoid throttling.",
    )
    is_reasoning: bool = Field(
        False, description="Indicating whether the model is reasoning or not."
    )
    sample_size: Optional[int] = Field(
        None, description="Exact number of records to randomly sample (seeded)."
    )
    sample_frac: Optional[float] = Field(
        None, description="Fraction of the dataset to sample (seeded)."
    )
    random_seed: Optional[int] = Field(
        42, description="Seed for reproducible sampling."
    )
    sample_ids_file: str = Field(
        "sampled_ids.json",
        description="Filename (in root_dir) to save/load sampled IDs.",
    )
