import json
import os
import pathlib
import random
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from openai import OpenAI

from finetune_data import build_finetune_conversations


@dataclass
class OpenAIFinetunePaths:
    output_dir: pathlib.Path
    training_file: pathlib.Path
    validation_file: Optional[pathlib.Path]


@dataclass
class OpenAIFinetuneStats:
    total_examples: int
    training_examples: int
    validation_examples: int


class OpenAIFinetuner:
    def __init__(
        self,
        prompt_name: Optional[str] = None,
        model: Optional[str] = None,
        output_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.prompt_name = prompt_name
        self.model = model
        self.output_dir = output_dir
        self.client = self._get_openai_client(api_key=api_key, base_url=base_url)

    def prepare_dataset(
        self,
        prompt_name: Optional[str] = None,
        validation_split: float = 0.1,
        seed: int = 3407,
    ) -> Tuple[OpenAIFinetunePaths, OpenAIFinetuneStats]:
        prompt_name = prompt_name or self.prompt_name
        if prompt_name is None:
            raise ValueError("prompt_name is required to prepare fine-tuning data.")

        conversations = build_finetune_conversations(prompt_name)

        rng = random.Random(seed)
        rng.shuffle(conversations)

        total_examples = len(conversations)
        validation_count = int(total_examples * validation_split) if validation_split > 0 else 0
        training_count = total_examples - validation_count

        training_conversations = conversations[:training_count]
        validation_conversations = conversations[training_count:] if validation_count else []

        output_dir = self.output_dir
        if output_dir is None:
            output_dir = f"output/openai_finetune/{prompt_name}"
        output_path = pathlib.Path(output_dir)

        training_path = output_path / "train.jsonl"
        self._write_openai_jsonl(training_conversations, training_path)

        validation_path = None
        if validation_conversations:
            validation_path = output_path / "validation.jsonl"
            self._write_openai_jsonl(validation_conversations, validation_path)

        paths = OpenAIFinetunePaths(
            output_dir=output_path,
            training_file=training_path,
            validation_file=validation_path,
        )
        stats = OpenAIFinetuneStats(
            total_examples=total_examples,
            training_examples=len(training_conversations),
            validation_examples=len(validation_conversations),
        )
        return paths, stats

    def upload_dataset(self, paths: OpenAIFinetunePaths) -> Tuple[str, Optional[str]]:
        training_file_id = self._upload_finetune_file(paths.training_file)
        validation_file_id = None
        if paths.validation_file is not None:
            validation_file_id = self._upload_finetune_file(paths.validation_file)
        return training_file_id, validation_file_id

    def create_job(
        self,
        training_file_id: str,
        validation_file_id: Optional[str] = None,
        model: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        model = model or self.model
        if model is None:
            raise ValueError("model is required to create a fine-tuning job.")

        params = {
            "training_file": training_file_id,
            "model": model,
        }
        if validation_file_id is not None:
            params["validation_file"] = validation_file_id
        if suffix is not None:
            params["suffix"] = suffix
        return self.client.fine_tuning.jobs.create(**params)

    def wait_for_job(
        self,
        job_id: str,
        poll_interval: int = 30,
        timeout_seconds: Optional[int] = None,
    ):
        start_time = time.time()
        while True:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            if job.status in {"succeeded", "failed", "cancelled"}:
                return job
            if timeout_seconds is not None and (time.time() - start_time) > timeout_seconds:
                return job
            time.sleep(poll_interval)

    def get_job(self, job_id: str):
        return self.client.fine_tuning.jobs.retrieve(job_id)

    def save_job_metadata(
        self,
        job,
        training_file_id: str,
        validation_file_id: Optional[str],
        paths: OpenAIFinetunePaths,
        prompt_name: Optional[str] = None,
        model: Optional[str] = None,
    ) -> pathlib.Path:
        prompt_name = prompt_name or self.prompt_name
        model = model or self.model
        if prompt_name is None or model is None:
            raise ValueError("prompt_name and model are required to save job metadata.")

        paths.output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "job_id": job.id,
            "status": job.status,
            "fine_tuned_model": getattr(job, "fine_tuned_model", None),
            "prompt_name": prompt_name,
            "base_model": model,
            "training_file_id": training_file_id,
            "validation_file_id": validation_file_id,
            "training_file_path": str(paths.training_file),
            "validation_file_path": str(paths.validation_file) if paths.validation_file else None,
        }
        metadata_path = paths.output_dir / f"{job.id}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        return metadata_path

    def _upload_finetune_file(self, file_path: pathlib.Path) -> str:
        with file_path.open("rb") as handle:
            uploaded = self.client.files.create(file=handle, purpose="fine-tune")
        return uploaded.id

    @staticmethod
    def _write_openai_jsonl(conversations: Iterable[list[dict]], path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for convo in conversations:
                record = {"messages": convo}
                handle.write(json.dumps(record, ensure_ascii=True))
                handle.write("\n")

    @staticmethod
    def _get_openai_client(
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> OpenAI:
        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI fine-tuning.")

        return OpenAI(base_url=base_url, api_key=api_key)
