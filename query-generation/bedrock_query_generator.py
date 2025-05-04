import time
import random
import json
import tqdm
import math
import yaml
import os
import re

import boto3
from botocore.config import Config
from config import BedrockQueryGenerationConfig
import argparse
from datasets import load_dataset

import pandas as pd
from typing import Any, Union, Tuple, Callable, Optional
from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd
import concurrent.futures
from threading import Lock
from datetime import datetime


class QueryGeneration(BaseModel):
    question: Optional[str] = Field(
        ..., description="A question generated from a paragraph."
    )


class StructuredLLM:
    def __init__(
        self,
        config: BedrockQueryGenerationConfig,
    ):
        self.model_id = config.model
        self.temperature = config.temperature
        self.max_completion_tokens = config.max_completion_tokens
        self.is_reasoning = config.is_reasoning
        self.thinking_params = None

        if self.is_reasoning:
            if "claude-3-7" in self.model_id:
                self.temperature = 1.0
                self.thinking_params = {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": self.max_completion_tokens - 64,
                    }
                }
            else:
                self.temperature = 0.6

        self.client = self._initialize_client()
        self.bedrock_llm = self._get_bedrock_llm()

    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        session = boto3.session.Session()
        configured_region = session.region_name
        return boto3.client(
            "bedrock-runtime",
            region_name=configured_region,
            config=Config(
                connect_timeout=300,
                read_timeout=1000,
                retries={"max_attempts": 3},
            ),
        )

    def _parse_json_from_text(self, text_to_parse: str) -> BaseModel:
        """Extract and parse JSON from text string."""
        try:
            parsed_json = JsonOutputParser().invoke(text_to_parse)
            parsed_output = QueryGeneration.model_validate(parsed_json)
        except Exception as e:
            regex_match = re.search(r"(\{.*\})", text_to_parse, re.DOTALL)
            if regex_match:
                cleaned_text = regex_match.group(1)
                try:
                    parsed_json = JsonOutputParser().invoke(cleaned_text)
                    parsed_output = QueryGeneration.model_validate(parsed_json)
                except Exception as e:
                    print(f"Error parsing JSON: {e}")
                    parsed_output = self._generate_empty_output()
            else:
                parsed_output = self._generate_empty_output()

        return parsed_output

    def _extract_from_content(
        self, content: Union[str, list[AIMessage]]
    ) -> Tuple[str, str]:
        """Extract raw_response and reasoning from model response content."""
        raw_response, reason = None, None
        try:
            if isinstance(content, str):
                raw_response = content
            elif isinstance(content, list):
                # We only have reasoning content if `content` is a list
                reason = next(
                    (
                        item["reasoning_content"]["text"]
                        for item in content
                        if item.get("type") == "reasoning_content"
                    ),
                    None,
                )
                # We don't use tool call for these models, just raw json output
                if self.model_id in [
                    "us.deepseek.r1-v1:0",
                    "mistral.mistral-large-2402-v1:0",
                ]:
                    raw_response = next(
                        (
                            item["text"]
                            for item in content
                            if item.get("type") == "text"
                        ),
                        None,
                    )
                else:
                    raw_response = next(
                        (
                            item["input"]
                            for item in content
                            if item.get("type") == "tool_use"
                        ),
                        None,
                    )
        except Exception as e:
            raw_response = None

        return raw_response, reason

    @staticmethod
    def _parse_raw_reasoning_output(raw_output: str) -> Tuple[str, str]:
        """Parse the raw reasoning output from the AI model."""
        pattern = r"<think>\s*(.*?)\s*</think>\s*(.*)"
        match = re.search(pattern, raw_output, re.DOTALL)
        if match:
            reasoning_tokens = match.group(1).strip()
            final_output = match.group(2).strip()
        else:
            reasoning_tokens = None
            final_output = raw_output

        return reasoning_tokens, final_output

    def _get_bedrock_llm(self):
        """Get the Bedrock LLM model based on the model name and temperature."""
        llm = ChatBedrockConverse(
            client=self.client,
            model_id=self.model_id,
            max_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            additional_model_request_fields=self.thinking_params,
        )
        llm = (
            llm.with_structured_output(QueryGeneration, include_raw=True)
            if self.model_id
            not in ["us.deepseek.r1-v1:0", "mistral.mistral-large-2402-v1:0"]
            else llm
        )
        return llm

    def _call_bedrock(self, messages: list[dict]) -> dict[str, Any]:
        """Call the Bedrock LLM model with the given message."""
        try:
            reason = None
            response = self.bedrock_llm.invoke(messages)
            if self.model_id in [
                "us.deepseek.r1-v1:0",
                "mistral.mistral-large-2402-v1:0",
            ]:
                content = response.content
                raw_response, reason = self._extract_from_content(content)
                parsed_output = self._parse_json_from_text(raw_response)
            else:
                parsed_output = response["parsed"]
                response = response["raw"]
                raw_response, reason = self._extract_from_content(response.content)

            usage_metadata = response.usage_metadata
            latency = response.response_metadata["metrics"]["latencyMs"][0]

            output = {
                "raw_response": raw_response,
                "parsed_output": parsed_output,
                "date": datetime.now(),
                "latency": latency,
                "input_tokens": usage_metadata["input_tokens"],
                "output_tokens": usage_metadata["output_tokens"],
                "reasoning": reason,
            }
            return output
        except Exception as e:
            print(f"Error calling Bedrock LLM: {e}")
            return {
                "raw_response": None,
                "parsed_output": None,
                "date": datetime.now(),
                "latency": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning": None,
                "error": str(e),
            }

    def _generate_empty_output(self):
        """Create an empty instance of the output format."""
        field_types = QueryGeneration.__annotations__
        fields = {}
        for field_name, field_type in field_types.items():
            if field_type == str:
                fields[field_name] = ""
            elif field_type == bool:
                fields[field_name] = False
            elif field_type == int:
                fields[field_name] = 0
            elif field_type == float:
                fields[field_name] = 0.0
            else:
                fields[field_name] = None
        return QueryGeneration(**fields)

    def __call__(self, prompt: str) -> dict[str, Any]:
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        return self._call_bedrock(messages)


class Generator:
    def __init__(
        self,
        config: BedrockQueryGenerationConfig,
        data: list[dict],
    ):
        os.makedirs(config.root_dir, exist_ok=True)
        # Random sampling
        if config.random_seed is not None and (
            config.sample_size or config.sample_frac
        ):
            sample_path = os.path.join(config.root_dir, config.sample_ids_file)
            if os.path.exists(sample_path):
                with open(sample_path, "r") as f:
                    sampled_ids = {tuple(ids) for ids in json.load(f)}
            else:
                random.seed(config.random_seed)
                all_ids = [tuple(r[col] for col in config.id_columns) for r in data]
                if config.sample_size:
                    k = config.sample_size
                else:
                    k = max(1, int(len(all_ids) * config.sample_frac))
                sampled_ids = set(random.sample(all_ids, k))
                with open(sample_path, "w") as f:
                    json.dump([list(t) for t in sampled_ids], f)
            data = [
                r
                for r in data
                if tuple(r[col] for col in config.id_columns) in sampled_ids
            ]

        self.llm = StructuredLLM(config=config)
        self.config = config
        self.file_lock = Lock()
        self.total_records = len(data)
        self.root_dir = config.root_dir
        self.jsonl_cache_dir = os.path.join(config.root_dir, "cache_result.jsonl")
        self.num_workers = config.num_workers
        self.cooldown = config.cooldown

        self.llm_params = {"config": config}

        self.processed_ids = self._load_processed_ids()

        self.records = [
            r
            for r in data
            if tuple(r[col] for col in self.config.id_columns) not in self.processed_ids
        ]
        print(
            f"Skipped {len(self.processed_ids)} cached records; {len(self.records)} to process."
        )

        records_per_worker = len(data) / self.num_workers
        self.batch_size = max(1, math.ceil(records_per_worker))

    def _save_result_to_jsonl(self, result: dict):
        """Save a single result to the JSONL file incrementally"""
        with self.file_lock:
            with open(self.jsonl_cache_dir, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")

    def _load_processed_ids(self) -> set[tuple]:
        """Read the cache JSONL and return a set of already‐seen id‐tuples."""
        seen = set()
        if not os.path.exists(self.jsonl_cache_dir):
            return seen
        with open(self.jsonl_cache_dir, "r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    raw = entry.get("raw", {})
                    key = tuple(raw.get(c) for c in self.config.id_columns)
                    seen.add(key)
                except (json.JSONDecodeError, TypeError):
                    continue
        return seen

    def _create_worker_llm(self):
        """Create new LLM instance for workers to avoid thread safety issues"""
        qa_llm = StructuredLLM(**self.llm_params)
        return qa_llm

    def _generate_question(self, record, worker_llm=None):
        qa_llm = worker_llm

        prompt = config.prompt_template.format(text=record[config.text_column])
        response = qa_llm(prompt)
        question = (
            response["parsed_output"].question
            if response["parsed_output"] is not None
            else None
        )

        result = {
            "question": question,
            "raw": record,
            "raw_response": response.get("raw_response", None),
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
            "reasoning_tokens": response.get("reasoning_tokens", 0),
            "latency": round(response.get("latency", 0), 2),
            "date": response.get("date", datetime.now()).strftime("%Y-%m-%d %H:%M"),
            "reasoning": response.get("reasoning", None),
            "error": response.get("error", None),
        }
        self._save_result_to_jsonl(result)

        return result

    def _process_batch(self, batch, process_fn, progress_callback=None):
        """Process a batch of records with the given processing function"""
        # Create worker-specific LLM instances to avoid thread safety issues
        worker_llms = self._create_worker_llm()
        results = []

        for record in batch:
            result = process_fn(record, worker_llms)
            results.append(result)

            if progress_callback:
                progress_callback(1)
            time.sleep(self.cooldown)
        return results

    def _parallel_process(self, process_fn: Callable):
        """Process all records in parallel using the provided function"""
        results = []
        total_records = len(self.records)

        batches = [
            self.records[i : i + self.batch_size]
            for i in range(0, len(self.records), self.batch_size)
        ]

        with tqdm.tqdm(total=total_records, desc="Processing records") as pbar:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_workers
            ) as executor:
                futures = []

                def update_progress(n):
                    pbar.update(n)

                for batch in batches:
                    futures.append(
                        executor.submit(
                            self._process_batch,
                            batch=batch,
                            process_fn=process_fn,
                            progress_callback=update_progress,
                        )
                    )

                for future in concurrent.futures.as_completed(futures):
                    batch_results = future.result()
                    results.extend(batch_results)

        assert len(results) == len(self.records), "Some records were not processed"

        return results

    def __call__(self):
        if len(self.processed_ids) < self.total_records:
            self._parallel_process(self._generate_question)
        self._save_final_output()

    def _save_final_output(self):
        """Assemble a CSV of id‐columns, text_column, and generated question."""
        rows = []
        with open(self.jsonl_cache_dir, "r") as f:
            for line in f:
                entry = json.loads(line)
                raw = entry.get("raw", {})
                out = {c: raw.get(c) for c in self.config.id_columns}
                out[self.config.text_column] = raw.get(self.config.text_column)
                out["generated_query"] = entry.get("question", "")
                rows.append(out)
        df = pd.DataFrame(rows)
        out_path = os.path.join(self.root_dir, "results.csv")
        df.to_csv(out_path, index=False)
        print(f"Final CSV written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config_data = yaml.safe_load(file)
    config = BedrockQueryGenerationConfig(**config_data)

    if os.path.exists(config.data_path) or config.data_path.lower().endswith(".csv"):
        data_df = pd.read_csv(config.data_path)
    else:
        hf_dataset = load_dataset(config.data_path, name=config.hf_config_name)
        data_df = hf_dataset["train"].to_pandas()
    data = data_df.to_dict(orient="records")

    generator = Generator(config, data)
    generator()
