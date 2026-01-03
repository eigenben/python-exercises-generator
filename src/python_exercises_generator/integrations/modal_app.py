import modal

from ..config import get_defaults, load_env

APP_NAME = "python-exercises-generator"

output_volume = modal.Volume.from_name("python-exercises-generator-output", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("python-exercises-generator-hf-cache", create_if_missing=True)

load_env()
defaults = get_defaults(require=True)

image = (
    modal.Image.debian_slim()
    .uv_sync(extras=["finetune"])
    .add_local_dir("src", "/root/src")
    .add_local_dir("prompts", "/root/prompts")
    .add_local_dir("data", "/root/data")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONPATH": "/root/src",
        "DEFAULT_GENERATION_EXAMPLES": ",".join(defaults.generation_examples),
        "DEFAULT_GENERATION_EXERCISES": ",".join(defaults.generation_exercises),
    })
)

with image.imports():
    from python_exercises_generator.config import get_defaults, load_env
    from python_exercises_generator.finetune import Finetuner
    from python_exercises_generator.generation import BatchGenerator
    load_env()
    _defaults = get_defaults(require=True)
    DEFAULT_EXERCISES = _defaults.generation_exercises

app = modal.App(APP_NAME)
app_context = {
    "gpu": "H100",
    "timeout": 2 * 60 * 60,
    "secrets": [
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    "volumes": {
        "/root/output": output_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
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
