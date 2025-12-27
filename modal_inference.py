import modal

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
        "vllm==0.11.2",
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.5.2",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster model transfers
)

app = modal.App(APP_NAME)

FAST_BOOT = True

@app.function(
    gpu="A100-80GB",
    timeout=30 * 60,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={
        "/root/output": output_volume,
        "/root/.cache/huggingface": modal.Volume.from_name("hf-cache-volume", create_if_missing=True),
        "/root/.cache/vllm": modal.Volume.from_name("vllm-cache-volume", create_if_missing=True),
    },
    image=vllm_image,
)
@modal.concurrent(
    max_inputs=8
)
@modal.web_server(
    port=8000,
    startup_timeout=10 * 60
)
def serve() -> None:
    import sys
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        "Qwen/Qwen3-8B-FP8",
        "--served-model-name",
        "Qwen/Qwen3-8B-FP8",
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        "8000"
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True, stdout=sys.stdout, stderr=sys.stdout)
