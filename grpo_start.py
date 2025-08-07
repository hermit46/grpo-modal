from modal import App, gpu, Image

# Initialize Modal app
app = App("grpo-training")

cuda_ver = "12.2.0"
flavor = "devel"
op_sys = "ubuntu22.04"
tag = f"{cuda_ver}-{flavor}-{op_sys}"

image: Image = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install([
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "wandb",
        "numpy",
        "scipy",
        "trl"
    ])
)

# Define image with GRPO dependencies
grpo_image = (
    Image.debian_slim()
    .pip_install([
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "wandb",
        "numpy",
        "scipy",
        "trl"
    ])
    .copy_local_file("synthetic_data.py", "/root/synthetic_data.py")  # Copy local data generator
)

@app.function(
    image=grpo_image,
    gpu="A100-40GB",  # Request A100 GPU with CUDA
    timeout=3600  # 1 hour timeout
)
def train_grpo_model():
    """
    GRPO training function that runs on Modal's GPU infrastructure
    """
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")

    # Import and generate synthetic dataset
    import sys
    sys.path.append('/root')
    from synthetic_data import generate_synthetic_dataset, validate_dataset_diversity

    dataset = generate_synthetic_dataset()
    validate_dataset_diversity(dataset)
    print(f"Generated {len(dataset)} problem groups for GRPO training")

    # Load Qwen2.5-1.5B-Instruct model
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    print(f"Loaded model: {model_name}")
    print(f"Sample dataset entry: {dataset[0]}")

    return "GRPO training setup complete with synthetic dataset"

@app.local_entrypoint()
def main():
    # Run the training function
    result = train_grpo_model.remote()
    print(result)
