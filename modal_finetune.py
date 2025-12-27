import modal

APP_NAME = "python-exercises-generator"
OUTPUT_VOLUME_NAME = "python-exercises-generator-output"

output_volume = modal.Volume.from_name(
    OUTPUT_VOLUME_NAME,
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim()
    .uv_sync(extras=["finetune"])
    .add_local_python_source(
        "cli",
        "distillation",
        "exercises",
        "finetune",
        "generation",
        "helpers",
    )
    .add_local_dir("prompts", "/root/prompts")
    .add_local_dir("data", "/root/data")
)

with image.imports():
    from finetune import Finetuner

app = modal.App(APP_NAME)


@app.function(
    gpu="A100-80GB",
    timeout=2 * 60 * 60,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={"/root/output": output_volume},
    image=image,
)
def finetune_remote(model_name: str, prompt: str) -> None:
    finetuner = Finetuner(model_name, prompt=prompt)
    finetuner.train()

@app.local_entrypoint()
def main(model_name: str, prompt: str = "default") -> None:
    finetune_remote.remote(model_name, prompt)
