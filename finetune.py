import builtins

original_print = builtins.print

import torch
from typing import Optional
from dataclasses import dataclass
from contextlib import contextmanager
from rich import print
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq

@dataclass
class FinetuneConfig:
    model_name: str
    max_seq_length: int
    dtype: Optional[torch.dtype]
    load_in_4bit: bool
    lora_r: int
    lora_alpha: int
    chat_template: Optional[str]
    chat_template_instruction_part: str
    chat_template_response_part: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    num_train_epochs: int

PRESET_CONFIGS = {
    "llama-3.3-70b-instruct": FinetuneConfig(
        model_name="unsloth/Llama-3.3-70B-Instruct",
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,
        lora_r=16,
        lora_alpha=16,
        chat_template="llama-3.1",
        chat_template_instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        chat_template_response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        max_steps=-1,
        num_train_epochs=1,
    ),
    "gpt-oss-20b": FinetuneConfig(
        model_name="unsloth/gpt-oss-20b",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        lora_r=16,
        lora_alpha=16,
        chat_template=None,
        chat_template_instruction_part="<|start|>user<|message|>",
        chat_template_response_part="<|start|>assistant<|message|>",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        max_steps=-1,
        num_train_epochs=1,
    ),
    "qwen3-coder-30b-a3b-instruct": FinetuneConfig(
        model_name="unsloth/Qwen3-Coder-30B-A3B-Instruct",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        lora_r=16,
        lora_alpha=16,
        chat_template=None,
        chat_template_instruction_part="<|im_start|>user\n",
        chat_template_response_part="<|im_start|>assistant\n",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        max_steps=-1,
        num_train_epochs=1,
    ),
    "ministral-3-14b-instruct-2512": FinetuneConfig(
        model_name="unsloth/Ministral-3-14B-Instruct-2512",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
        lora_r=16,
        lora_alpha=16,
        chat_template=None,
        chat_template_instruction_part="[INST]",
        chat_template_response_part="[/INST]",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        max_steps=-1,
        num_train_epochs=1,
    ),
}

class Finetuner:
    def __init__(self, model_name: str, prompt: str = "default"):
        self.model_name = model_name
        self.config = PRESET_CONFIGS[model_name]
        self.prompt = prompt
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.trainer = None
        self.finetuned_model_dir = None

    def load_model(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.config.model_name,
            max_seq_length = self.config.max_seq_length,
            dtype = self.config.dtype,
            load_in_4bit = self.config.load_in_4bit,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r = self.config.lora_r,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
            lora_alpha = self.config.lora_alpha,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def apply_chat_template_to_tokenizer(self):
        if self.config.chat_template is not None:
            self.tokenizer = get_chat_template(
                self.tokenizer,
                chat_template = self.config.chat_template,
            )
        return self.tokenizer

    def load_exercises_dataset(self):
        from datasets import Dataset
        import pathlib
        from exercises import load_all_exercises

        # Load the prompt template
        template_path = pathlib.Path(f"prompts/finetune/{self.prompt}.md")
        template = template_path.read_text()

        # Load all exercises
        exercises = load_all_exercises()

        # Create conversations in ShareGPT format
        conversations = []
        for exercise in exercises:
            if exercise.problem_md:
                user_message = template.replace("{{ problem }}", exercise.problem_md)
                conversation = [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": exercise.solution_md}
                ]
                conversations.append(conversation)

        # Create dataset with conversations field
        dataset = Dataset.from_dict({"conversations": conversations})
        self.dataset = dataset
        return dataset

    def apply_chat_template_to_dataset(self):
        def formatting_prompts_func(examples):
            convos = examples["conversations"]
            texts = [self.tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
            return { "text" : texts, }

        self.dataset = standardize_sharegpt(self.dataset)
        self.dataset = self.dataset.map(formatting_prompts_func, batched=True)
        return self.dataset

    def setup_trainer(self):
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = self.dataset,
            dataset_text_field = "text",
            max_seq_length = self.config.max_seq_length,
            data_collator = DataCollatorForSeq2Seq(tokenizer = self.tokenizer),
            packing = False,
            args = SFTConfig(
                per_device_train_batch_size = self.config.per_device_train_batch_size,
                gradient_accumulation_steps = self.config.gradient_accumulation_steps,
                warmup_steps = 2,
                num_train_epochs = self.config.num_train_epochs,
                max_steps = self.config.max_steps,
                learning_rate = 2e-4,
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.001,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = f"output/finetuned_models/{self.model_name}",
                report_to = "wandb",
            ),
        )

        trainer = train_on_responses_only(
            trainer,
            instruction_part = self.config.chat_template_instruction_part,
            response_part = self.config.chat_template_response_part,
        )

        self.trainer = trainer
        return trainer

    def save_model(self, output_dir: Optional[str] = None):
        """Save the finetuned model and tokenizer."""
        finetuned_model_name = f"{self.model_name}-finetuned-python-exercises"
        if output_dir is None:
            output_dir = f"output/finetuned_models/{finetuned_model_name}"
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.model.push_to_hub(finetuned_model_name, private=True)
        self.tokenizer.push_to_hub(finetuned_model_name, private=True)
        self.finetuned_model_dir = output_dir
        return output_dir

    def load_finetuned_model(self, output_dir: Optional[str] = None):
        """Load the finetuned model and tokenizer from disk."""
        if output_dir is None:
            if self.finetuned_model_dir:
                output_dir = self.finetuned_model_dir
            else:
                finetuned_model_name = f"{self.model_name}-finetuned-python-exercises"
                output_dir = f"output/finetuned_models/{finetuned_model_name}"

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = output_dir,
            max_seq_length = self.config.max_seq_length,
            dtype = self.config.dtype,
            load_in_4bit = self.config.load_in_4bit,
        )

        self.model = model
        self.tokenizer = tokenizer
        self.apply_chat_template_to_tokenizer()
        self.model.eval()
        return model, tokenizer

    def load_finetuned_model_for_inference(self, output_dir: Optional[str] = None):
        model, tokenizer = self.load_finetuned_model(output_dir)
        FastLanguageModel.for_inference(model)
        return model, tokenizer

    def inference(self, user_message: str, max_new_tokens: int = 512) -> str:
        """Run inference on a single user message and return the response."""
        if self.model is None or self.tokenizer is None:
            self.load_finetuned_model_for_inference()

        FastLanguageModel.for_inference(self.model)

        messages = [{"role": "user", "content": user_message}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            return_tensors = "pt",
        )
        input_ids = input_ids.to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids = input_ids,
                max_new_tokens = max_new_tokens,
                do_sample = False,
            )

        response_ids = output_ids[0][input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response_text.strip()

    @contextmanager
    def track_gpu_memory(self):
        """Context manager to track GPU memory usage during training."""
        # Capture initial GPU stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        print(f"\n[bold blue]GPU Memory Stats[/bold blue]")
        print(f"[dim]  • GPU: {gpu_stats.name}[/dim]")
        print(f"[dim]  • Max memory: {max_memory} GB[/dim]")
        print(f"[dim]  • Reserved at start: {start_gpu_memory} GB[/dim]")

        # Yield a dict to store training stats
        stats_container = {}
        yield stats_container

        # Calculate final memory usage
        trainer_stats = stats_container.get('trainer_stats')
        if trainer_stats:
            used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
            used_percentage = round(used_memory / max_memory * 100, 3)
            lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

            print(f"\n[bold blue]Training Complete - Memory Summary[/bold blue]")
            print(f"[dim]  • Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds ({round(trainer_stats.metrics['train_runtime']/60, 2)} minutes)[/dim]")
            print(f"[dim]  • Peak reserved memory: {used_memory} GB ({used_percentage}% of max)[/dim]")
            print(f"[dim]  • Peak memory for training: {used_memory_for_lora} GB ({lora_percentage}% of max)[/dim]")

    def train(self):
        print(f"[bold blue]Starting finetuning with config:[/bold blue] [yellow]{self.config.model_name}[/yellow]")
        print(f"[dim]  • Max steps: {self.config.max_steps}[/dim]")
        print(f"[dim]  • Num epochs: {self.config.num_train_epochs}[/dim]")
        print(f"[dim]  • Batch size: {self.config.per_device_train_batch_size}[/dim]")
        print(f"[dim]  • Gradient accumulation: {self.config.gradient_accumulation_steps}[/dim]")

        # Load model and tokenizer
        print("\n[bold blue]Loading model and tokenizer...[/bold blue]")
        self.load_model()
        print("[green]✓ Model loaded successfully[/green]")

        # Apply chat template to tokenizer
        print("\n[bold blue]Applying chat template to tokenizer...[/bold blue]")
        self.apply_chat_template_to_tokenizer()
        print("[green]✓ Chat template applied[/green]")

        # Load and prepare dataset
        print("\n[bold blue]Loading exercises dataset...[/bold blue]")
        self.load_exercises_dataset()
        print(f"[green]✓ Loaded {len(self.dataset)} exercise examples[/green]")
        
        print("\n[bold blue]Applying chat template to dataset...[/bold blue]")
        self.apply_chat_template_to_dataset()
        print("[green]✓ Dataset prepared[/green]")

        print("\n[bold blue]Sample Templated Text...[/bold blue]")
        original_print(self.dataset[0]["text"])

        # Setup trainer
        print("\n[bold blue]Setting up trainer...[/bold blue]")
        self.setup_trainer()
        print("[green]✓ Trainer configured[/green]")

        # Start training with memory tracking
        print("\n[bold magenta]Starting training...[/bold magenta]")
        with self.track_gpu_memory() as stats:
            trainer_stats = self.trainer.train()
            stats['trainer_stats'] = trainer_stats

        # Save the finetuned model
        print("\n[bold blue]Saving finetuned model...[/bold blue]")
        output_dir = self.save_model()
        print(f"[green]✓ Model saved to {output_dir}[/green]")

        print("\n[bold green]✓ Training complete![/bold green]")
