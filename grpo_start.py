from modal import App, gpu, Image, Secret

# Initialize Modal app
app = App("grpo-training")

cuda_ver = "12.2.0"
flavor = "devel"
op_sys = "ubuntu22.04"
tag = f"{cuda_ver}-{flavor}-{op_sys}"

image = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install(  # required to build flash-attn
        "ninja",
        "packaging",
        "wheel",
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "trl",
        "bitsandbytes",
        "wandb",
        "vllm",
    )
    # .pip_install(  # add flash-attn - commented out for faster startup
    #     "flash-attn==2.7.4.post1", extra_options="--no-build-isolation"
    # )
)

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# Helper function to extract answer from XML tags
def extract_xml_answer(completion):
    """Extract raw text between <answer> tags"""
    import re
    match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_hash_answer(text: str) -> str | None:
    """Extract final answer from GSM8K format (after ####)"""
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")



# Reward function for strict formatting
def check_strict_formatting(completion):
    """Check for proper formatting with newlines"""
    import re
    # Check for proper multi-line format with newlines
    reasoning_pattern = r'<reasoning>\s*\n.*?\n\s*</reasoning>'
    answer_pattern = r'<answer>\s*\n.*?\n\s*</answer>'

    has_proper_reasoning = bool(re.search(reasoning_pattern, completion, re.DOTALL))
    has_proper_answer = bool(re.search(answer_pattern, completion, re.DOTALL))

    if has_proper_reasoning and has_proper_answer:
        return 1.0
    return 0.0

# Reward function for soft formatting
def check_soft_formatting(completion):
    """Check for basic XML tags presence using regex for better performance"""
    import re
    # Compile patterns once for better performance
    reasoning_pattern = re.compile(r'<reasoning>.*?</reasoning>', re.DOTALL)
    answer_pattern = re.compile(r'<answer>.*?</answer>', re.DOTALL)

    has_reasoning = bool(reasoning_pattern.search(completion))
    has_answer = bool(answer_pattern.search(completion))

    if has_reasoning and has_answer:
        return 1.0
    elif has_reasoning or has_answer:
        return 0.3  # Partial credit
    return -0.5  # No proper formatting

# Reward function for integer answers
def check_integer_reward(completion):
    """Reward integer answers - simplified version"""
    answer = extract_xml_answer(completion)
    if answer is None:
        return 0.0  # No extractable answer

    # Clean the answer (remove extra spaces, etc.)
    answer_clean = answer.strip()

    # Check if it's a valid integer (very simple)
    if answer_clean.lstrip('-').isdigit():
        return 0.5  # Good integer answer

    return 0.0  # Not an integer

# Reward function for correctness
def check_correctness(prompt, completion):
    """Compare extracted answer against correct answer encoded in prompt"""
    import re

    # Extract correct answer from prompt (encoded as [CORRECT:123])
    correct_match = re.search(r'\[CORRECT:([^\]]+)\]', prompt)
    if not correct_match:
        return 0.0  # No correct answer provided

    correct_answer = correct_match.group(1).strip()
    extracted_answer = extract_xml_answer(completion)

    if extracted_answer is None:
        return -0.5  # No answer extracted

    try:
        # Clean both answers for comparison (remove $ and commas)
        correct_clean = correct_answer.replace("$", "").replace(",", "").strip()
        extracted_clean = extracted_answer.replace("$", "").replace(",", "").strip()

        # Try to compare as numbers
        if correct_clean.replace(".", "").replace("-", "").isdigit() and extracted_clean.replace(".", "").replace("-", "").isdigit():
            if float(correct_clean) == float(extracted_clean):
                return 1.0  # Correct answer!
            else:
                return -0.5  # Wrong answer
        else:
            # Compare as strings if not numbers
            if correct_clean.lower() == extracted_clean.lower():
                return 1.0  # Correct answer!
            else:
                return -0.5  # Wrong answer
    except ValueError:
        return -0.3  # Couldn't parse for comparison



def get_dataset(split="train", max_items: int | None = None ):
    ds = load_dataset("openai/gsm8k", "main")[split]

    data = ds.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"])
        }
    )
    return data if not max_items else data.select(range(max_items))

@app.function(
    image=image,
    gpu="A100-40GB",  # Request A100 GPU with CUDA
    timeout=3600,
    secrets=[
        Secret.from_name("wandb-secret"),
        Secret.from_name("huggingface-secret")
    ]
)
def train_grpo_model(model_name: str, dataset_size: int | None = None):
    """
    GRPO training function that runs on Modal's GPU infrastructure
    """
    import torch
    from datasets import load_dataset

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")

    # Load GSM8K dataset directly for GRPO
    ds = load_dataset("openai/gsm8k", "main")
    train_subset = ds["train"].select(range(40))  # First 40 problems

    # Convert to simple prompt format for GRPO
    def format_gsm8k_for_grpo(example):
        question = example["question"]
        correct_answer = extract_hash_answer(example["answer"])

        # Simple prompt with correct answer encoded for reward function
        prompt = f"Solve this math problem step by step:\n{question}\n\nPlease format your response with <reasoning> and <answer> tags. [CORRECT:{correct_answer}]"
        return {"prompt": prompt}

    grpo_dataset = train_subset.map(format_gsm8k_for_grpo)
    print(f"Prepared {len(grpo_dataset)} GSM8K problems for GRPO training")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure all special tokens are properly set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        # attn_implementation="flash_attention_2",  # Commented out for faster startup
        device_map="balanced", #Evenly distribute across devices
    )

    # Setup GRPO training configuration
    from trl import GRPOConfig, GRPOTrainer

    # GRPO training configuration
    grpo_config = GRPOConfig(
        output_dir=f"grpo-{model_name}-{dataset_size}",
        run_name=f"grpo-{model_name}-{dataset_size}",
        num_train_epochs=1,
        per_device_train_batch_size=2,  # Must be divisible by num_generations
        gradient_accumulation_steps=1,
        learning_rate=5e-6,  # Lower learning rate for stability
        max_grad_norm=1.0,  # Gradient clipping to prevent instability
        logging_steps=5,
        save_steps=50,
        eval_strategy="no",
        warmup_steps=10,
        remove_unused_columns=False,
        report_to="wandb",  # Disable wandb logging
        # GRPO-specific parameters
        max_prompt_length=256,
        max_completion_length=256,
        num_generations=2,  # Number of responses per prompt (matches our dataset)
        beta=0.1,  # KL coefficient
        epsilon=0.2,  # PPO-style clipping parameter
        temperature=0.8,  # Slightly higher to avoid numerical issues
        loss_type="grpo",
        num_iterations=1,
        log_completions=True,  # Log sample completions for debugging
        # Add generation safety parameters
        generation_kwargs={
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
        },
    )

    # Use the mapped dataset directly
    train_dataset = grpo_dataset

    # Safe wrapper for reward functions to prevent NaN/Inf
    def safe_reward_wrapper(reward_func):
        def wrapper(*args, **kwargs):
            try:
                rewards = reward_func(*args, **kwargs)
                # Ensure no NaN or Inf values
                safe_rewards = []
                for r in rewards:
                    if not isinstance(r, (int, float)) or r != r or abs(r) == float('inf'):  # Check for NaN/Inf
                        safe_rewards.append(0.0)  # Default safe value
                    else:
                        safe_rewards.append(float(r))  # Ensure float type
                return safe_rewards
            except Exception as e:
                print(f"Reward function error: {e}")
                return [0.0] * len(args[1])  # Return zeros for all completions
        return wrapper

    # List of individual reward functions for GRPO
    reward_functions = [
        safe_reward_wrapper(lambda prompts, completions, **kwargs: [check_soft_formatting(c) for c in completions]),
        safe_reward_wrapper(lambda prompts, completions, **kwargs: [check_integer_reward(c) for c in completions]),
        safe_reward_wrapper(lambda prompts, completions, **kwargs: [check_correctness(p, c) for p, c in zip(prompts, completions)])
    ]

    # Initialize GRPO trainer with list of reward functions
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_functions,  # Use list of reward functions
        args=grpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model("./grpo-math-model")
    print("GRPO training completed and model saved!")

    return "GRPO training completed successfully"

@app.local_entrypoint()
def main():
    # Run the training function with a smaller, more stable model
    result = train_grpo_model.remote(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",  # More stable for GRPO
        dataset_size=10
    )
    print(result)
