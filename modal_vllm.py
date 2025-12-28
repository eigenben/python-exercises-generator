import modal
from typing import Any

APP_NAME = "python-exercises-generator"
OUTPUT_VOLUME_NAME = "python-exercises-generator-output"

output_volume = modal.Volume.from_name(
    OUTPUT_VOLUME_NAME,
    create_if_missing=True,
)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.13.0",
        "huggingface-hub==0.36.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster model transfers
)

app = modal.App(APP_NAME)

@app.function(
    gpu="H100",
    timeout=30 * 60,
    image=vllm_image,
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")],
    volumes={
        "/root/output": output_volume,
        "/root/.cache/huggingface": modal.Volume.from_name("hf-cache-volume", create_if_missing=True),
        "/root/.cache/vllm": modal.Volume.from_name("vllm-cache-volume", create_if_missing=True),
    },
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=8000, startup_timeout=10 * 60)
def serve_gpt_oss_20b() -> None:
    import sys
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "unsloth/gpt-oss-20b",
        "--enable-lora",
        "--lora-modules",
        "gpt-oss-20b-finetuned-python-exercises=/root/output/finetuned_models/gpt-oss-20b-finetuned-python-exercises",
        "--host",
        "0.0.0.0",
        "--port",
        "8000"
    ]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True, stdout=sys.stdout, stderr=sys.stdout)
