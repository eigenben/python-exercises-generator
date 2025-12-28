import modal
import aiohttp
import json
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
        "vllm==0.11.2",
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.5.2",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster model transfers
)

app = modal.App(APP_NAME)

FAST_BOOT = True

@app.function(
    gpu="H200",
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
        "unsloth/Llama-3.3-70B-Instruct",
        "--served-model-name",
        "unsloth/Llama-3.3-70B-Instruct",
        "--quantization",
        "fp8",
        "--kv-cache-dtype",
        "fp8",
        "--enable-lora",
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

@app.local_entrypoint()
async def test(test_timeout=10 * 60, content=None, twice=True):
    url = serve.get_web_url()

    system_prompt = {
        "role": "system",
        "content": "You are an expert python programmer.",
    }
    if content is None:
        content = "Explain the singular value decomposition."

    messages = [  # OpenAI chat format
        system_prompt,
        {"role": "user", "content": content},
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        async with session.get("/health", timeout=test_timeout - 1 * 60) as resp:
            up = resp.status == 200
        assert up, f"Failed health check for server at {url}"
        print(f"Successful health check for server at {url}")

        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await _send_request(session, "llm", messages)
        if twice:
            messages[0]["content"] = "You are Jar Jar Binks."
            print(f"Sending messages to {url}:", *messages, sep="\n\t")
            await _send_request(session, "llm", messages)


async def _send_request(
    session: aiohttp.ClientSession, model: str, messages: list
) -> None:
    # `stream=True` tells an OpenAI-compatible backend to stream chunks
    payload: dict[str, Any] = {"messages": messages, "model": model, "stream": True}

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers, timeout=1 * 60
    ) as resp:
        async for raw in resp.content:
            resp.raise_for_status()
            # extract new content and stream it
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):  # SSE prefix
                line = line[len("data: ") :]

            chunk = json.loads(line)
            assert (
                chunk["object"] == "chat.completion.chunk"
            )  # or something went horribly wrong
            print(chunk["choices"][0]["delta"]["content"], end="")
    print()
