# grpo-modal


This repository contains code for training a Group Relative Policy Optimization (GRPO) model for math word problem solving on Modal's GPU infrastructure.

`grpo_start.py`: This script defines a Modal function that runs the GRPO training process on a GPU instance. The function imports the gsm8k dataset, loads a pre-trained language model (Qwen2.5-1.5B-Instruct), and sets up the training process.

The `grpo_start.py` script can be run locally using `modal run grpo_start.py` or deployed to a Modal cloud environment using `modal deploy grpo_start.py`.
