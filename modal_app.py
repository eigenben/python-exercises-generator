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
    from generation import BatchGenerator, DEFAULT_EXERCISES

app = modal.App(APP_NAME)
app_context = {
    "gpu": "H100",
    "timeout": 2 * 60 * 60,
    "secrets": [
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    "volumes": {"/root/output": output_volume},
    "image": image,
}


@app.function(**app_context)
def finetune(model: str, prompt: str) -> None:
    finetuner = Finetuner(model, prompt=prompt)
    finetuner.train()

@app.function(**app_context)
def inference(model: str, prompt: str) -> None:
    finetuner = Finetuner(model)
    print(finetuner.inference(prompt))

@app.function(**app_context)
def batch_generate(model: str) -> None:
    batch_generator = BatchGenerator(finetuned_model=model)
    results = batch_generator(DEFAULT_EXERCISES)
    print(results)
