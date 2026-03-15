#!/usr/bin/env python
# coding: utf-8

# <!-- To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance! -->
# <div class="align-center">
# <a href="https://nvidia.com/"><img src="https://www.nvidia.com/content/dam/en-zz/Solutions/about-nvidia/logo-and-brand/01-nvidia-logo-horiz-500x200-2c50-d@2x.png" width="115"></a>
# 
#     
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# 
# <!-- <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div> -->
# 
# # Goal: Multi-environment GRPO using Unsloth and NeMo Gym
# 
# Modern reinforcement learning often involves training a model for more than one task. For example, we may want to train an agent to do deep research, software engineering, and puzzle solving simultaneously. Each environment can have different dependencies, tools, state, or other complex requirements. 
# 
# In this notebook, we demonstrate how to train a model with multiple NeMo Gym environments using Unsloth. Our goal is to teach Qwen-2.5-1.5b-Instruct to play sudoku AND follow instructions better using GRPO on a single GPU!
# 
# You will learn how to:
# - configure an Unsloth optimized model
# - start multiple NeMo Gym resources servers
# - train using Unsloth and multiple NeMo Gym environments
# - test and save the trained model
# 
# 
# This notebook was developed on 1 H100 GPU through NVIDIA Brev. 
# 
# If you are using a GPU with lower VRAM, you should adjust configuration parameters accordingly, such as max output length, quantization, or parameter efficient finetuning. Unsloth has a bunch of examples of low VRAM training that work with NeMo Gym training environments!

# ## Installation
# If you are using Google Colab, please visit [Unsloth installation docs](https://unsloth.ai/docs/get-started/install) rather than the pip install below. 

# In[ ]:


get_ipython().system('pip install unsloth unsloth_zoo omegaconf')

# If your jupyter kernel and pip python do not match, check where the jupyter kernel python is, and install there, for example:
# !source /home/ubuntu/.venv/bin/activate
# !python -m ensurepip --upgrade
# !python -m pip install -U pip
# !python -m pip install -U unsloth


# # Load the model
# 
# In this example, we will do full finetuning, but Unsloth supports optimized low precision (e.g. 4 or 8 bit) or parameter-efficient training methods (e.g. LoRA). Check out Unsloth's documentation if you are interested in these methods!

# In[ ]:


from unsloth import FastLanguageModel

model_name = "unsloth/Qwen2.5-1.5B-Instruct"
max_seq_length = (
    4096  # Can increase for longer outputs, or decrease if running into OOM
)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = False,  # set to True for low precision training to save VRAM
    full_finetuning = True,  # set to False for LoRA training
    offload_embedding = True,  # Reduces VRAM a little
)


# If you want to try out LoRA, uncomment the code below, and make sure that full_finetuning is set to False above. LoRA is a parameter-efficient training method that reduces computational cost by only training a small percentage of the full model parameters.

# In[ ]:


# lora_rank = 4 # Larger rank = smarter, but slower
# model = FastLanguageModel.get_peft_model(
#     model,
#     r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
#     target_modules = [
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj",
#     ],
#     lora_alpha = lora_rank*2, # *2 speeds up training
#     use_gradient_checkpointing = "unsloth", # Reduces memory usage
#     random_state = 42,
# )


# # NeMo Gym resources server setup
# 
# NeMo Gym resources servers provide tool implementations, logic to process actions, update state, provide observations, and calculate rewards for actions taken.
# 
# The reasoning gym resource server is an integration of [reasoning gym](https://github.com/open-thought/reasoning-gym), which is a library of procedural dataset generators and algorithmically verifiable reasoning environments for training reasoning models with reinforcement learning (RL). It includes more than 100 tasks over many domains with configurable difficulty, including but not limited to algebra, arithmetic, computation, cognition, geometry, graph theory, logic, and many common games. 
# 
# The instruction following resources server evaluates language model responses against instruction-following criteria using Open-Instruct and IFEval constraints.
# 
# If you are using Google Colab, add the flag `uv_pip_set_python=true` to `ng_run` command.
# 
# The cell below will automatically:
# 1. Clone [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) (requires Python 3.12+ and `uv` on the system)
# 2. Set up the virtual environment and install dependencies
# 3. Create the mini sudoku training dataset
# 4. Download the instruction following dataset
# 5. Ensure resources-only configs exist for both servers
# 6. Start both resources servers in the background
# 
# Google Colab is auto-detected and the `uv_pip_set_python=true` flag is added when needed.

# In[ ]:


import subprocess
import os
import time
import atexit
import requests

GYM_DIR = os.path.expanduser("~/Gym")

# Detect Google Colab
try:
    import google.colab
    _on_colab = True
except ImportError:
    _on_colab = False

# Step 1: Clone NeMo Gym
if not os.path.exists(GYM_DIR):
    print("Cloning NeMo Gym...")
    subprocess.run(
        ["git", "clone", "https://github.com/NVIDIA-NeMo/Gym.git", GYM_DIR],
        check = True,
    )

# Step 2: Create venv and install dependencies
if not os.path.exists(os.path.join(GYM_DIR, ".venv", "bin", "python")):
    print("Setting up NeMo Gym environment (this may take a few minutes)...")
    subprocess.run(["uv", "venv", "--python", "3.12"], cwd = GYM_DIR, check = True)
    subprocess.run(
        ["bash", "-c", "source .venv/bin/activate && uv sync"],
        cwd = GYM_DIR, check = True,
    )
    subprocess.run(
        ["bash", "-c", "source .venv/bin/activate && uv pip install reasoning-gym"],
        cwd = GYM_DIR, check = True,
    )
# Step 3: Create sudoku dataset
sudoku_path = os.path.join(
    GYM_DIR, "resources_servers/reasoning_gym/data/train_mini_sudoku.jsonl"
)
if not os.path.exists(sudoku_path):
    print("Creating mini_sudoku dataset (2000 examples)...")
    subprocess.run(
        [
            "bash", "-c",
            "source .venv/bin/activate && python "
            "resources_servers/reasoning_gym/scripts/create_dataset.py "
            "--task mini_sudoku --size 2000 --seed 42 "
            f"--output {sudoku_path}",
        ],
        cwd = GYM_DIR, check = True,
    )

# Step 4: Download instruction_following dataset
import shutil
from huggingface_hub import hf_hub_download

if_path = os.path.join(
    GYM_DIR,
    "resources_servers/instruction_following/data/instruction_following.jsonl",
)
if not os.path.exists(if_path):
    print("Downloading instruction_following dataset...")
    src = hf_hub_download(
        repo_id = "nvidia/Nemotron-RL-instruction_following",
        filename = "instruction_following.jsonl",
        repo_type = "dataset",
    )
    os.makedirs(os.path.dirname(if_path), exist_ok = True)
    shutil.copy(src, if_path)
# Step 5: Create resources_only.yaml for instruction_following if missing
_if_resources_only = os.path.join(
    GYM_DIR, "resources_servers/instruction_following/configs/resources_only.yaml"
)
if not os.path.exists(_if_resources_only):
    with open(_if_resources_only, "w") as _f:
        _f.write(
            "instruction_following:\\n"
            "  resources_servers:\\n"
            "    instruction_following:\\n"
            "      entrypoint: app.py\\n"
            "      domain: instruction_following\\n"
            "      verified: false\\n"
        )
# Start NeMo Gym servers if not already running
try:
    requests.get("http://127.0.0.1:11000/global_config_dict_yaml", timeout = 2)
    print("NeMo Gym servers already running on port 11000.")
except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
    _colab_flag = " +uv_pip_set_python=true"
    print("Starting NeMo Gym servers...")
    _ng_log = open(os.path.join(GYM_DIR, "ng_run.log"), "w")
    ng_process = subprocess.Popen(
        [
            "bash", "-c",
            "source .venv/bin/activate && ng_run "
            '"+config_paths=[resources_servers/reasoning_gym/configs/resources_only.yaml,resources_servers/instruction_following/configs/resources_only.yaml]"'
            + _colab_flag,
        ],
        cwd = GYM_DIR,
        stdout = _ng_log,
        stderr = subprocess.STDOUT,
    )

    def _cleanup_ng():
        if ng_process.poll() is None:
            ng_process.terminate()
            try:
                ng_process.wait(timeout = 10)
            except subprocess.TimeoutExpired:
                ng_process.kill()
        _ng_log.close()
    atexit.register(_cleanup_ng)

    print("Waiting for servers", end = "", flush = True)
    for _ in range(120):
        try:
            requests.get(
                "http://127.0.0.1:11000/global_config_dict_yaml", timeout = 2
            )
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            if ng_process.poll() is not None:
                raise RuntimeError(
                    "Server process exited unexpectedly. "
                    f"Check {GYM_DIR}/ng_run.log for details."
                )
            print(".", end = "", flush = True)
            time.sleep(3)
    else:
        raise RuntimeError(
            "NeMo Gym servers did not start within 6 minutes."
        )
    print("\nHead server ready!")


# NeMo Gym starts a head server on port 11000 by default, and the resources server port is selected at random from available ports, unless specified otherwise. We can automatically extract the resources servers ports using the head server:

# In[ ]:


import yaml
import requests
from omegaconf import OmegaConf


head_port = 11000
try:
    response = requests.get(
        f"http://127.0.0.1:{head_port}/global_config_dict_yaml", timeout = 5
    )
    response.raise_for_status()
except requests.exceptions.ConnectionError:
    raise RuntimeError(
        "Could not connect to NeMo Gym head server on port 11000. "
        "Make sure the setup cell above ran successfully."
    )
global_config_dict = OmegaConf.create(yaml.safe_load(response.text))

# Verify all required servers are present
_required_servers = {"reasoning_gym", "instruction_following"}
_missing = _required_servers - set(global_config_dict.keys())
if _missing:
    raise RuntimeError(
        f"Head server is missing required server configs: {_missing}. "
        "Please stop any existing NeMo Gym server and re-run the setup cell above."
    )

def get_verify_endpoint(server_name: str) -> str:
    config = global_config_dict[server_name].resources_servers[server_name]
    return f"http://{config.host}:{config.port}/verify"

# Wait for all resources servers to be fully ready
for _srv in global_config_dict:
    try:
        _cfg = global_config_dict[_srv].resources_servers[_srv]
    except (KeyError, AttributeError):
        print(f"Warning: skipping {_srv} (no resources_server config found)")
        continue
    print(f"Waiting for {_srv} at {_cfg.host}:{_cfg.port}", end = "", flush = True)
    for _i in range(90):
        try:
            requests.get(f"http://{_cfg.host}:{_cfg.port}/", timeout = 2)
            break
        except requests.exceptions.ConnectionError:
            print(".", end = "", flush = True)
            time.sleep(2)
    else:
        raise RuntimeError(
            f"{_srv} server at {_cfg.host}:{_cfg.port} did not start within 3 minutes."
        )
    print(f"\n{_srv} ready!")


# # Dataset prep
# 
# Next, let's create and load the dataset. We can generate a mini sudoku dataset using the create script in NeMo Gym, which uses the reasoning gym library.
# 
# Both datasets were created automatically by the setup cell above. Now load them! We will limit each dataset to 1000 samples to create an even task distribution.

# In[ ]:


import os
import json
import random
from datasets import Dataset

dataset_configs = [
    # (dataset path, server name)
    (
        "~/Gym/resources_servers/reasoning_gym/data/train_mini_sudoku.jsonl",
        "reasoning_gym",
    ),
    (
        "~/Gym/resources_servers/instruction_following/data/instruction_following.jsonl",
        "instruction_following",
    ),
]

train_data = []
examples_by_server = {}
max_length_seen = 0
max_per_dataset = 1000
for dataset_path, server_name in dataset_configs:
    expanded_path = os.path.expanduser(dataset_path)
    if not os.path.exists(expanded_path):
        raise FileNotFoundError(
            f"Dataset not found at {expanded_path}. "
            "Run the setup cell above first."
        )
    with open(expanded_path, "r") as f:
        lines = f.readlines()
    if len(lines) > max_per_dataset:
        lines = random.sample(lines, max_per_dataset)
    for line in lines:
        data = json.loads(line)
        task_prompt = data["responses_create_params"]["input"][0]["content"]

        example = {
            "prompt": [{"role": "user", "content": task_prompt}],
            "resources_server_ref": {"name": server_name},
            "verify_extra": data,
        }
        train_data.append(example)

        if server_name not in examples_by_server:
            examples_by_server[server_name] = example

        prompt_length = len(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": task_prompt}], add_generation_prompt = True
            )
        )
        max_length_seen = max(max_length_seen, prompt_length)

random.shuffle(train_data)

print(f"Loaded {len(train_data)} examples!\n")
for server_name, example in examples_by_server.items():
    print(f"Example data for {server_name}")
    print(f"{example['prompt'][0]['content']}\n")

train_dataset = Dataset.from_list(train_data)


# # Define reward function
# 
# Now lets create a reward function that uses NeMo Gym's verifiers, routing tasks to resources servers using the server names:

# In[ ]:


import numpy as np


def reward_fn(completions, prompts = None, **kwargs):
    resources_server_refs = kwargs["resources_server_ref"]
    verify_extras = kwargs["verify_extra"]
    scores = []
    for i, completion in enumerate(completions):
        completion_text = completion[0]["content"]
        task_prompt = prompts[i][0]["content"]
        verify_endpoint = get_verify_endpoint(resources_server_refs[i]["name"])

        verify_request = {k: v for k, v in verify_extras[i].items() if v is not None}
        verify_request["responses_create_params"] = {
            "input": [{"role": "user", "content": task_prompt}]
        }
        verify_request["response"] = {
            "id": "resp",
            "created_at": 0,
            "model": model_name,
            "object": "response",
            "output": [
                {
                    "id": "msg",
                    "role": "assistant",
                    "type": "message",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": completion_text,
                            "annotations": [],
                        }
                    ],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        try:
            resp = requests.post(verify_endpoint, json = verify_request, timeout = 30)
            reward = resp.json().get("reward", 0.0) if resp.status_code == 200 else 0.0
        except requests.exceptions.RequestException as e:
            print(f"Warning: verify request failed: {e}")
            reward = 0.0
        scores.append(reward)
    return np.array(scores)


# # Configure and launch GRPO
# 
# In this example, we will train the model using Group Relative Policy Optimization (GRPO), an efficient and effective reinforcement learning algorithm. Unsloth also supports GSPO, GAPO, Dr GRPO and more!
# 
# Below we set training hyperparameters. We will train for 100 steps and see significant improvements in the models performance at completing both mini sudoku and diverse instruction following tasks!

# In[ ]:


from trl import GRPOConfig, GRPOTrainer

max_prompt_length = max_length_seen + 1  # +1 just in case as in other unsloth examples
max_completion_length = max_seq_length - max_prompt_length

training_args = GRPOConfig(
    temperature = 1.0,
    learning_rate = 1e-5,
    weight_decay = 0.001,
    warmup_ratio = 0.0,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 64,
    num_generations = 8,
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    num_train_epochs = 1,
    max_steps = 100,
    save_steps = 100,
    report_to = "none",  # Can use Weights & Biases
    # run_name = run_name, # for Weights & Biases
    output_dir = "outputs",
    epsilon_high = 0.28,
    mask_truncated_completions = True,
    # log_completions = True, # uncomment to see rollouts printed to the console!
    # num_completions_to_print = 1,
    # For optional training + evaluation
    # fp16_full_eval = True,
    # per_device_eval_batch_size = 4,
    # eval_accumulation_steps = 1,
    # eval_strategy = "steps",
    # eval_steps = 1,
)

# For optional training + evaluation
# new_dataset = dataset.train_test_split(test_size = 0.05)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [reward_fn],
    args = training_args,
    train_dataset = train_dataset,
    # For optional training + evaluation
    # train_dataset = new_dataset["train"],
    # eval_dataset = new_dataset["test"],
)


# During training, you should see the average reward increases during training as the model learns how to complete sudoku and instruction following tasks better, though it will be noisy as the tasks are diverse. In order to monitor model improvements closer, an evaluation dataset can be added to training. 
# 
# Lets start training!

# In[ ]:


trainer.train()


# # Test the trained model!

# In[ ]:


text = tokenizer.apply_chat_template(
    [
        {
            "role": "user",
            "content": "In 4x4 Mini Sudoku:\n- Each row must contain each number from 1-4 exactly once\n- Each column must contain each number 1-4 exactly once\n- Each 2x2 subgrid must contain each number 1-4 exactly once\nSolve this 4x4 Mini Sudoku puzzle:\n4 _ _ _\n_ 3 _ _\n_ 1 3 _\n_ _ _ _\nFormat your response as the puzzle above, with spaces separating each number within a row, and newlines separating rows.\n",
        }
    ],
    tokenize = False,
    add_generation_prompt = True,
)

from transformers import TextStreamer

_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    do_sample = True,
    temperature = 1.0,
    max_new_tokens = 4096,
    streamer = TextStreamer(tokenizer, skip_prompt = False),
)


# <a name="Save"></a>
# ### Saving to float16 or MXFP4 for vLLM
# 
# Unsloth supports saving to `float16` directly. Select `merged_16bit` for float16. Unsloth also supports saving in low or mixed precision such as `mxfp4`, and allows `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.

# In[ ]:


# Merge and push to hub in mxfp4 4bit format
if False:
    model.save_pretrained_merged("nemo_gym_multi_finetune_4bit", tokenizer, save_method = "mxfp4")
if False:
    model.push_to_hub_merged(
        "repo_id/nemo_gym_multi_finetune_4bit", tokenizer, token = "YOUR_HF_TOKEN", save_method = "mxfp4"
    )

# Merge and push to hub in 16bit
if False:
    model.save_pretrained_merged(
        "nemo_gym_multi_finetune_16bit", tokenizer, save_method = "merged_16bit"
    )
if False:  # Pushing to HF Hub
    model.push_to_hub_merged(
        "HF_USERNAME/nemo_gym_multi_finetune_16bit", tokenizer, save_method = "merged_16bit", token = "YOUR_HF_TOKEN"
    )


# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other resources:
# 1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
# 2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
# 3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
# 4. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://unsloth.ai/docs/get-started/unsloth-notebooks)!
# 
# <div class="align-center">
#   <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#   <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
# 
#   Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# 
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme)
# </div>
