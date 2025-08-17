# Query Generation (OpenAI Batch and AWS Bedrock)

This module generates questions from input text using two execution paths:

- OpenAI Batch API for large-scale asynchronous processing
- AWS Bedrock (Claude/other models) for parallel, online generation

Outputs are saved as JSONL (intermediate) and CSV (final), joined back to your input identifiers.

## Structure

- `config.py`: Pydantic config models shared by both flows.
- `batch_query_generator.py`: OpenAI Batch pipeline (request sharding, batch submission, response download, parsing, merge to CSV).
- `bedrock_query_generator.py`: AWS Bedrock pipeline (concurrent workers, optional reasoning, sampling, incremental cache + final CSV).
- `configs/`: Example YAML configuration files for both flows.

## Install

```
pip install -r requirements.txt
```

Environment variables:

- OpenAI flow: set `OPENAI_API_KEY`.
- Bedrock flow: configure AWS credentials/region (e.g., via `aws configure` or env vars `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`).

## Configure

Copy and adjust a YAML from `query-generation/configs/`.

Shared fields (from `BaseQueryGenerationConfig`):

- `data_path`: Local CSV path or Hugging Face dataset id (`user/dataset`).
- `hf_config_name` (optional): HF dataset config name when needed.
- `root_dir`: Working directory for this job. Requests, responses, cache, and results are saved here.
- `model`: Model name (e.g., `gpt-4o-mini`, a Bedrock model id).
- `text_column`: Column containing the text to prompt on.
- `id_columns`: List of columns that uniquely identify each row; used to construct and later rejoin results.
- `prompt_template`: Template string used to build prompts (must include `{text}`).

OpenAI Batch fields (`BatchQueryGenerationConfig`):

- `params`: Dictionary of OpenAI chat parameters (e.g., `temperature`, `reasoning_effort`).
- `shard_size` (default 50000): Rows per JSONL shard for submission.

Bedrock fields (`BedrockQueryGenerationConfig`):

- `max_completion_tokens` (default 1024), `temperature` (default 0.0).
- `num_workers` (default 4), `cooldown` (seconds between calls; default 0.5).
- `is_reasoning` (bool): Enables reasoning mode where applicable.
- Sampling: `sample_size`, `sample_frac`, `random_seed`, `sample_ids_file`.
- `output_schema`: `QueryGeneration` or `TwoHopQueryGeneration` (controls output columns).

## Usage

### OpenAI Batch pipeline

Submit requests (creates shards and submits batch jobs):

```
python query-generation/batch_query_generator.py --config query-generation/configs/batch_generation_v1.yaml --stage submit
```

After jobs complete, download, parse, and merge to CSV:

```
python query-generation/batch_query_generator.py --config query-generation/configs/batch_generation_v1.yaml --stage dl
```

What happens:

- Requests written to: `{root_dir}/requests/shard-XXX.jsonl`
- Batch job ids saved to: `{root_dir}/batch_details.json`
- Responses downloaded to: `{root_dir}/responses/*-response.jsonl`
- Parsed and joined CSV saved to: `{root_dir}/{basename(root_dir)}.csv`

Notes:

- If `model` is one of `o3`, `o3-mini`, `o1`, `o1-mini`, `temperature` is removed automatically.
- Input may be a CSV or a Hugging Face dataset (train split is used by default).

### AWS Bedrock pipeline

Run the generator (parallel, incremental cache, final CSV):

```
python query-generation/bedrock_query_generator.py --config query-generation/configs/bedrock_generation_v1.yaml
```

What happens:

- Incremental JSONL cache: `{root_dir}/cache_result.jsonl` (append-only)
- Final CSV with input ids, text, and generated fields: `{root_dir}/results.csv`
- If sampling is enabled, sampled ids are persisted to `{root_dir}/{sample_ids_file}` to keep runs reproducible.

Reasoning notes:

- When `is_reasoning` is true, temperature and request parameters are adjusted for supported models (e.g., Claude 3.7). Raw reasoning content (if available) is captured alongside parsed output.

## Input/Output

Input requirements:

- CSV or HF dataset must include `text_column` and all `id_columns` defined in the config.

Outputs:

- OpenAI Batch: Final CSV contains original columns and `generated_query` parsed from model output. Intermediate shards and responses are retained under `{root_dir}`.
- Bedrock: Final CSV contains `id_columns`, the original `text_column`, and fields from the selected `output_schema` (e.g., `question`; for two-hop also `chunk1`, `chunk2`).

## Examples

Minimal OpenAI Batch YAML (illustrative):

```yaml
data_path: data/sample.csv
root_dir: runs/openai_batch_sample
model: gpt-4o-mini
text_column: paragraph
id_columns: [doc_id]
prompt_template: |
  Read the following text and generate a single question.
  Text: {text}
params:
  temperature: 0.2
  response_format: {"type": "json_object"}
```

Minimal Bedrock YAML (illustrative):

```yaml
data_path: data/sample.csv
root_dir: runs/bedrock_sample
model: anthropic.claude-3-5-sonnet-20240620-v1:0
text_column: paragraph
id_columns: [doc_id]
prompt_template: |
  Read the following text and generate a single question.
  Text: {text}
max_completion_tokens: 1024
temperature: 0.2
num_workers: 4
cooldown: 0.5
is_reasoning: false
output_schema: QueryGeneration
```

