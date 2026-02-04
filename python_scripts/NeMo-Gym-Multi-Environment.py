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
# To install NeMo Gym, follow the guide [here](https://docs.nvidia.com/nemo/gym/latest/get-started/index.html).
# 
# To install Unsloth on your local device, follow the guide [here](https://unsloth.ai/docs/get-started/install-and-update).
# 
# 
# This notebook was developed on 1 H100 GPU. If you are using a GPU with lower VRAM, you should adjust configuration parameters accordingly, such as max output length, quantization, or parameter efficient finetuning. Unsloth has a bunch of examples of low VRAM training that work with Nemo Gym training environments!

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
# 

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


# # Nemo Gym resources server setup
# 
# Nemo Gym resources servers provide tool implementations, logic to process actions, update state, provide observations, and calculate rewards for actions taken.
# 
# The reasoning gym resource server is an integration of [reasoning gym](https://github.com/open-thought/reasoning-gym), which is a library of procedural dataset generators and algorithmically verifiable reasoning environments for training reasoning models with reinforcement learning (RL). It includes more than 100 tasks over many domains with configurable difficulty, including but not limited to algebra, arithmetic, computation, cognition, geometry, graph theory, logic, and many common games. 
# 
# The instruction following resources server evaluates language model responses against instruction-following criteria using Open-Instruct and IFEval constraints.
# 
# First, start the resources servers in a terminal:
# 
# ```
# cd ~/Gym
# uv venv
# source .venv/bin/activate
# uv sync --active
# ng_run "+config_paths=[resources_servers/reasoning_gym/configs/resources_only.yaml,resources_servers/instruction_following/configs/resources_only.yaml]"
# ```
# 
# 
# You should see a similar output in the terminal:
# 
# ```
# All 2 / 2 servers ready! Polling every 60s
# 
# ####################################################################################################
# #
# # Server Instances
# #
# ####################################################################################################
# 
# [1] reasoning_gym (resources_servers/reasoning_gym)
# {
#     'config_path': 'reasoning_gym',
#     'dir_path': '/home/ubuntu/Gym/resources_servers/reasoning_gym',
#     'entrypoint': 'app.py',
#     'host': '127.0.0.1',
#     'name': 'reasoning_gym',
#     'pid': 186628,
#     'port': 47151,
#     'process_name': 'reasoning_gym',
#     'server_type': 'resources_servers',
#     'url': 'http://127.0.0.1:47151',
# }
# [2] instruction_following (resources_servers/instruction_following)
# {
#     'config_path': 'instruction_following',
#     'dir_path': '/home/ubuntu/Gym/resources_servers/instruction_following',
#     'entrypoint': 'app.py',
#     'host': '127.0.0.1',
#     'name': 'instruction_following',
#     'pid': 186629,
#     'port': 58173,
#     'process_name': 'instruction_following',
#     'server_type': 'resources_servers',
#     'url': 'http://127.0.0.1:58173',
# }
# ####################################################################################################
# ```

# Nemo Gym starts a head server on port 11000 by default, and the resources server port is selected at random from available ports, unless specified otherwise. We can automatically extract the resources servers ports using the head server:

# In[ ]:


import yaml
import requests
from omegaconf import OmegaConf


head_port = 11000
response = requests.get(
    f"http://127.0.0.1:{head_port}/global_config_dict_yaml", timeout = 5
)
global_config_dict = OmegaConf.create(yaml.safe_load(response.text))


def get_verify_endpoint(server_name: str) -> str:
    config = global_config_dict[server_name].resources_servers[server_name]
    return f"http://{config.host}:{config.port}/verify"


# # Dataset prep
# 
# Next, let's create and load the dataset. We can generate a mini sudoku dataset using the create script in Nemo Gym, which uses the reasoning gym library.
# 
# ```
# cd ~/Gym
# 
# uv add reasoning-gym
# 
# python resources_servers/reasoning_gym/scripts/create_dataset.py \
#     --task mini_sudoku \
#     --size 2000 \
#     --seed 42 \
#     --output resources_servers/reasoning_gym/data/train_mini_sudoku.jsonl
# ```
# 
# Next, we need to download the instruction following dataset from https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following
# 
# Once downloaded, place it in `~/Gym/resources_servers/instruction_following/data/instruction_following.jsonl`
# 
# Now load the datasets! We will limit each dataset to 1000 samples to create an even task distribution.

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
    lines = open(os.path.expanduser(dataset_path), "r").readlines()
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
# Now lets create a reward function that uses Nemo Gym's verifiers, routing tasks to resources servers using the server names:

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
            "created_at": 0.0,
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
        except:
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
    temperature = 1.0,
    max_new_tokens = 4096,
    streamer = TextStreamer(tokenizer, skip_prompt = False),
)


# <a name="Save"></a>
# ### Saving to float16 or MXFP4 for vLLM
# 
# Unsloth supports saving to `float16` directly. Select `merged_16bit` for float16. Unsloth also supports saving in low or mixed precision such as `mxfp4`, and allows `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.

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
# Some other links:
# 1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
# 2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
# 3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
# 6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://unsloth.ai/docs/get-started/unsloth-notebooks)!
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
# 
