import tiktoken
from typing import Optional
import argparse
from tooldantic import OpenAiResponseFormatBaseModel
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, field_validator
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import os
from typing import Union
import yaml
import tqdm
import json

load_dotenv()


class Config(BaseModel):
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
    params: dict = Field(
        ...,
        description="Dictionary of OpenAI API call parameters (e.g. `temperature`, `reasoning_effort`).",
    )
    shard_size: int = Field(
        50000,
        description="Number of records per shard for generating requests. Default is 50,000 (max for OpenAI).",
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


class QueryGeneration(OpenAiResponseFormatBaseModel):
    question: Optional[str]


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OPENAI_API_KEY found in the environment variables.")
    return OpenAI(api_key=api_key)


def generate_requests(
    data: pd.DataFrame,
    output_path: str,
    config: Config,
    response_format: Union[BaseModel, dict],
):
    """
    Generates request shards for the given data using the provided configuration.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        output_path (str): The file path where the generated request JSONL files will be saved.
        config (Config): A configuration instance containing settings like text_column, template, model, params, shard_size, and root_dir.
        response_format (BaseModel or str): Response format model or string for query generation.
    """
    input_tokens = 0
    SHARD_SIZE = 50_000
    for shard_start in tqdm.tqdm(range(0, len(data), SHARD_SIZE)):
        dataset_slice = data.iloc[shard_start : shard_start + SHARD_SIZE]
        shard_num = shard_start // SHARD_SIZE
        for i, row in dataset_slice.iterrows():
            prompt = config.prompt_template.format(text=row[config.text_column])
            with open(
                os.path.join(output_path, f"shard-{shard_num:03d}.jsonl"), "a"
            ) as f:
                task = {
                    "custom_id": row["custom_id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": config.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": response_format
                        if isinstance(response_format, dict)
                        else response_format.model_json_schema(),
                        **config.params,
                    },
                }
                if config.model in ["o3", "o3-mini", "o1", "o1-mini"]:
                    task["body"].pop("temperature")
                f.write(json.dumps(task) + "\n")
                input_tokens += count_tokens(prompt)

    print(f"Total input tokens: {input_tokens}")
    print(f"Total shards: {shard_num + 1}")
    return input_tokens


def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in a given text.

    Args:
        text (str): Text to count tokens.

    Returns:
        int: Number of tokens in the text.
    """
    enc = tiktoken.get_encoding("o200k_base")
    tokens = enc.encode(text)
    return len(tokens)


def merge_jsonl_files(file_paths: Union[str, list], output_file=None) -> list:
    """
    Merges multiple json files into a single json file.

    Args:
        file_paths (Union[str, list]): List of file paths to merge or directory path.
        output_file (str): Output file path. If None, returns the merged json as a list.

    Returns:
        list: Merged jsons as a single list.
    """
    if isinstance(file_paths, str):
        file_paths = [
            os.path.join(file_paths, f)
            for f in os.listdir(file_paths)
            if f.endswith(".jsonl")
        ]

    merged_lines = []
    for path in file_paths:
        with open(path, "r") as infile:
            merged_lines.extend(infile.readlines())

    if output_file:
        with open(output_file, "w") as outfile:
            outfile.writelines(merged_lines)
    return merged_lines


def generate_batch_job(jsonl_path: str) -> str:
    """
    Generates a batch job for the given JSONL file.

    Args:
        jsonl_path (str): Path to the JSONL file.

    Returns:
        str: Batch job ID.
    """
    client = get_client()
    batch_file = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return batch_job.id


def generate_batch_jobs(input_path: str, json_output_path: str):
    """
    Generates batch jobs for the given input path.

    Args:
        input_path (str): Path to the input files.
        json_output_path (str): Path to save the batch job details.
    """
    shards = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if f.endswith(".jsonl")
    ]
    file_to_batch_id = {}
    for shard in shards:
        batch_id = generate_batch_job(shard)
        file_to_batch_id[shard] = batch_id

    with open(os.path.join(json_output_path), "w") as f:
        json.dump(file_to_batch_id, f)


def extract_responses_to_df(responses_path: str, id_columns: list[str]) -> pd.DataFrame:
    """
    Extract responses from jsonl files, parse them along with their id columns

    Args:
        responses_path (str): Path to the jsonl response file(s) or directory.
        id_columns (list[str]): List of column names (as defined in the config) that make up the custom_id.

    Returns:
        pd.DataFrame: DataFrame containing individual id column values and 'generated_query'
    """

    merged_lines = merge_jsonl_files(responses_path)

    rows = []
    total_completion_tokens = 0

    for line in merged_lines:
        data = json.loads(line)
        custom_id = data.get("custom_id", "")
        response_body = data.get("response", {}).get("body", {})
        choices = response_body.get("choices", [])
        if not choices:
            continue
        message = choices[0].get("message", {})
        response_text = message.get("content", "")
        usage = response_body.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        total_completion_tokens += completion_tokens

        if len(id_columns) > 1:
            id_values = custom_id.split("__")
        else:
            id_values = [custom_id]

        row = {}
        # Map the split id values to their corresponding column names (order defined in id_columns).
        for col, val in zip(id_columns, id_values):
            row[col] = val
        row["generated_query"] = JsonOutputParser().invoke(response_text)["question"]

        rows.append(row)

    print(f"Total completion tokens: {total_completion_tokens}")
    return pd.DataFrame(rows)


def extract_responses(jsonl_list: list, response_format: BaseModel) -> pd.DataFrame:
    """
    Extracts responses from the given list of JSONL files.

    Args:
        jsonl_list (list): List of JSONL files.
        response_format (BaseModel): Response format model.

    Returns:
        pd.DataFrame: Extracted responses as a DataFrame.
    """
    fields = response_format.model_fields.keys()

    responses = []
    for jsonl in jsonl_list:
        loaded = json.loads(jsonl)
        if loaded["error"]:
            continue
        custom_id = loaded["custom_id"]
        response = loaded["response"]["body"]["choices"][0]["message"]["content"]
        response = json.loads(response)
        responses.append(
            {**{field: response[field] for field in fields}, "custom_id": custom_id}
        )

    responses_df = pd.DataFrame(responses)
    return responses_df


def all_completed(batch_job_ids: str) -> bool:
    """
    Checks if all batch jobs are completed.

    Args:
        batch_job_details (str): Path to the JSON file containing the batch job details.

    Returns:
        bool: True if all batch jobs are completed, False otherwise.
    """
    with open(batch_job_ids, "r") as f:
        file_to_batch_id = json.load(f)

    client = get_client()
    for file, batch_id in file_to_batch_id.items():
        batch = client.batches.retrieve(batch_id)
        if batch.status != "completed":
            return False
    return True


def download_outputs(batch_job_details: str, output_dir: str):
    """
    Downloads the batch outputs to a given directory.

    Args:
        batch_job_ids (str): Path to the JSON file containing the batch job details.
        output_dir (str): Path to save the responses.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(batch_job_details, "r") as f:
        file_to_batch_id = json.load(f)

    client = get_client()

    for file, batch_id in tqdm.tqdm(file_to_batch_id.items()):
        batch = client.batches.retrieve(batch_id)
        output_file_id = batch.output_file_id
        response_bytes = client.files.content(output_file_id).content

        base_name = os.path.basename(file)
        base_name, ext = os.path.splitext(base_name)
        response_name = f"{base_name}-response{ext}"
        output_file = os.path.join(output_dir, response_name)

        with open(output_file, "wb") as f:
            f.write(response_bytes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=".\configs\query_generation_v1.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--stage",
        default="dl",
        choices=["submit", "dl"],
        help="Which stage of the pipeline to run.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config_data = yaml.safe_load(file)
    config = Config(**config_data)

    requests_path = os.path.join(config.root_dir, "requests")
    responses_path = os.path.join(config.root_dir, "responses")
    batch_details_path = os.path.join(config.root_dir, "batch_details.json")

    for path in [requests_path, responses_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    if os.path.exists(config.data_path) or config.data_path.lower().endswith(".csv"):
        data_df = pd.read_csv(config.data_path)
    else:
        hf_dataset = load_dataset(config.data_path, name=config.hf_config_name)
        data_df = hf_dataset["train"].to_pandas()

    if args.stage == "submit":

        def generate_custom_id(row, columns):
            if isinstance(columns, str):
                return columns
            if len(columns) == 1:
                return columns[0]
            return "__".join([str(row[col]) for col in columns])

        data_df["custom_id"] = data_df.apply(
            lambda row: generate_custom_id(row, config.id_columns), axis=1
        )
        print("Generating requests...")
        generate_requests(
            data=data_df,
            output_path=requests_path,
            config=config,
            response_format=QueryGeneration,
        )
        print("Submitting requests...")
        generate_batch_jobs(requests_path, batch_details_path)

    elif args.stage == "dl":
        if not all_completed(batch_details_path):
            print("Some batch jobs are not completed yet, try again later.")
            exit()
        print("Downloading responses...")
        download_outputs(batch_details_path, responses_path)

        print("Parsing responses...")
        response_df = extract_responses_to_df(responses_path, config.id_columns)

        for col in config.id_columns:
            if col in response_df.columns and col in data_df.columns:
                response_df[col] = response_df[col].astype(data_df[col].dtype)

        merged_df = pd.merge(data_df, response_df, on=config.id_columns, how="inner")
        output_csv = os.path.join(
            os.path.join(config.root_dir, f"{os.path.basename(config.root_dir)}.csv")
        )
        merged_df.to_csv(output_csv, index=False)
