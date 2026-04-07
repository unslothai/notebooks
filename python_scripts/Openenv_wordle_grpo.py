#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Openenv_wordle_grpo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # <img width="35" height="35" alt="image" src="https://github.com/user-attachments/assets/2700a971-e5d6-4036-b03f-2f89c9791609" /> OpenEnv: Agentic Execution Environments
# We're using the new [OpenEnv](https://github.com/meta-pytorch/OpenEnv) library which has over 2000+ environments for RL!
# 
# To run this, press "*Runtime*" and press "*Run all*" on your A100 Google Colab Pro instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth on your local device, follow [our guide](https://unsloth.ai/docs/get-started/install).

# # Goal: Make Qwen3-4B play games with Reinforcement Learning
# 
# Our goal is to make OpenAI's open model Qwen3-4B play the wordle game with reinforcement learning. We want the model to devise a strategy to play wordle, and we will run this strategy until we win or lose.
# 
# <img width="270" height="265" alt="image" src="https://github.com/user-attachments/assets/1a51a172-ff1a-40a2-8bee-2746f5017aa0" />

# # Installation
# We'll be using [Unsloth](https://github.com/unslothai/unsloth) to do RL on Qwen3-4B, and [OpenEnv](https://github.com/meta-pytorch/OpenEnv) for the environment interactions. Unsloth saves 70% VRAM usage and makes reinforcement learning 2 to 6x faster!

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'import os, importlib.util\n!pip install --upgrade -qqq uv\nif importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):\n    try: import numpy; get_numpy = f"numpy=={numpy.__version__}"\n    except: get_numpy = "numpy"\n    !uv pip install -qqq \\\n        "torch>=2.8.0" "triton>=3.4.0" {get_numpy} torchvision bitsandbytes "transformers==4.56.2" trackio \\\n        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\\n        "unsloth[base] @ git+https://github.com/unslothai/unsloth" \\\n        git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels\nelif importlib.util.find_spec("unsloth") is None:\n    !uv pip install -qqq unsloth trackio\n!uv pip install --upgrade --no-deps transformers==4.56.2 tokenizers trl==0.29.1 unsloth unsloth_zoo\n')


# We will then install [OpenEnv](https://github.com/meta-pytorch/OpenEnv) from source:

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install -qqq fastapi uvicorn requests open_spiel --prefer-binary\n!pip install "openenv-core[core]>=0.2.1"\n!git clone https://github.com/meta-pytorch/OpenEnv.git > /dev/null 2>&1\n%cd OpenEnv\nimport subprocess, sys, os\nfrom pathlib import Path\nsys.path.insert(0, \'./envs\')  # Add OpenEnv envs for textarena_env module\nsys.path.insert(0, \'./src\')\nworking_directory = str(Path.cwd().absolute())\n')


# We'll load Qwen3-4B and set some parameters:
# * `max_seq_length = 4096` The maximum context length of the model. Increasing it will use more memory.
# * `lora_rank = 32` The larger this number, the smarter the RL process, but the slower and more memory usage.

# In[ ]:


import os
from unsloth import FastLanguageModel
import torch
max_seq_length = 4096 # Can increase for longer RL output
lora_rank = 32       # Larger rank = smarter, but slower
model, processor = FastLanguageModel.from_pretrained(
    # model_name = "unsloth/gemma-3-1b-it",
    model_name = 'unsloth/Qwen3-4B',
    load_in_4bit = False,
    max_seq_length = max_seq_length,
    fast_inference = True,
)
tokenizer = getattr(processor, 'tokenizer', processor)


# To do efficient RL, we will use [LoRA](https://arxiv.org/abs/2106.09685), which allows us to only add 1 to 5% of extra weights to the model for finetuning purposes. This allows us to save memory usage by over 60%, and yet it retains good accuracy. Read Unsloth's [Reinforcement Learning Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) for more details.

# In[ ]:


model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank*2, # *2 speeds up training
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    random_state = 3407,
)


# ## Initialize the Environment
# 
# Let's begin by setting up the environment that will be used during training.  
# For this task, we'll rely on the **TextArena** environment from **OpenEnv**, which exposes a familiar Gymnasium-style API (`reset()`, `step()`, etc.) to simplify interaction.
# 
# In this example, we'll connect to the hosted environment at [openenv/wordle](https://huggingface.co/spaces/openenv/wordle).  
# For production use or custom configurations, we **strongly recommend** running the environment locally via Docker. The hosted versions on the Hub currently have limited concurrency support, so duplicating the Space to your own account is the preferred approach in those cases.
# 
# For more information, refer to the [TRL-OpenEnv documentation](https://huggingface.co/docs/trl/main/en/openenv).

# In[ ]:


# notebook_login() # Uncomment if you want to push to HF Hub
# from huggingface_hub import notebook_login
# notebook_login()


# In[ ]:


from textarena_env import TextArenaEnv

wordle_url = "https://openenv-wordle.hf.space" # Duplicate the Space and update this!
env = TextArenaEnv(base_url = wordle_url).sync()  # .sync() needed: OpenEnv client is async by default
# wordle_url = "openenv/wordle"
# env = TextArenaEnv.from_hub(repo_id = wordle_url)


# ## Rollout function with helpers
# 
# The **rollout function** defines how the agent interacts with the environment during GRPO training.
# It's responsible for generating model completions, collecting feedback (rewards), and returning all necessary information for optimization.
# 
# In this setup:
# - The function is called automatically by the **GRPOTrainer** during each training step.  
# - It uses the trainer's built-in `generate_rollout_completions()` method for efficient generation with vLLM in colocate mode.
# - Each rollout represents a full interaction loop. The model guesses, receives feedback from Wordle, and updates based on reward signals.
# - The **`env_mask`** tracks which tokens are model-generated vs environment-generated, ensuring only model tokens contribute to the training loss.
# 
# The rewards track different aspects of the agent's performance. Helper functions (like `rollout_once`) handle one episode of interaction, keeping the main `rollout_func` clean and modular.
# 
# This modular approach allows GRPO to efficiently sample, evaluate, and improve the model's guessing strategy through reinforcement learning.
# 
# First, we define the `system_prompt` that guides the model's behavior as an expert Wordle solver with strategic reasoning and structured responses.

# In[ ]:


# @title System prompt (click to expand)
system_prompt = """
You are an expert Wordle solver with deep knowledge of English vocabulary, letter frequency patterns, and optimal guessing strategies.

## GAME RULES

1. The target is a 5-letter English word
2. You have 6 attempts to guess the correct word
3. After each guess, you receive color-coded feedback:
   - GREEN or G: Letter is correct and in the correct position
   - YELLOW or Y: Letter is in the word but in the wrong position
   - GRAY or X: Letter is not in the word at all
4. All guesses must be valid 5-letter English words
5. You cannot reuse a word you've already guessed

## RESPONSE FORMAT

Format your response as follows:
<reasoning>
Briefly analyze the feedback from previous guesses, eliminating letters and finding possible words.
</reasoning>
[guess]

## IMPORTANT CONSTRAINTS

- Use lowercase only
- One guess per response
- Must be exactly 5 letters
- Must be a real English word from standard dictionaries
- Never repeat a previous guess

## YOUR GOAL

Solve the Wordle in as few guesses as possible by strategically using feedback to eliminate impossible words and narrow down the solution space efficiently."""


# Now, let's define the `rollout_func`:
# 
# This function orchestrates the interaction between the model and the Wordle environment. For each prompt in the batch, it runs the episode interaction, collecting rewards and model outputs for GRPO optimization.

# In[ ]:


max_new_tokens = 256
max_turns = 6

def rollout_func(prompts, trainer):
    """
    Rollout function for GRPO training with environment interaction.

    This function is called by GRPOTrainer to generate completions and compute rewards.
    It uses trainer.generate_rollout_completions() for inference.

    Args:
        prompts: List of prompts to generate from
        trainer: GRPOTrainer instance containing context and configuration

    Returns:
        Dictionary with prompt_ids, completion_ids, logprobs, env_mask, and reward signals
    """
    episode_prompt_ids = []
    episode_completion_ids = []
    episode_logprobs = []
    episode_env_masks = []
    correctness_rewards = []
    position_rewards = []
    format_rewards = []

    for prompt_text in prompts:
        episode = rollout_once(
            trainer = trainer,
            env = env,
            tokenizer = tokenizer,
            dataset_prompt = prompt_text,
            system_prompt = system_prompt,
            max_turns = max_turns,
            max_new_tokens = max_new_tokens,
        )
        episode_prompt_ids.append(episode["prompt_ids"])
        episode_completion_ids.append(episode["completion_ids"])
        episode_logprobs.append(episode["logprobs"])
        episode_env_masks.append(episode["env_mask"])
        correctness_rewards.append(episode["correct_reward"])
        position_rewards.append(episode["position_reward"])
        format_rewards.append(compute_format_reward(episode["model_outputs"]))

    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "env_mask": episode_env_masks,
        "correct_reward": correctness_rewards,
        "position_reward": position_rewards,
        "format_reward": format_rewards,
    }


# ### Define `rollout_once`
# 
# The `rollout_once` function runs **one full interaction loop** between the model and the Wordle environment using the trainer's generation method.  
# It executes a mini episode of gameplay, from generating a guess to receiving and processing feedback.
# 
# Here's the step-by-step breakdown:
# 
# 1. **Environment reset:** Start a new game session and initialize the observation.  
# 2. **Prompt construction:** Combine the system prompt, current state, and user messages to form the model input.  
# 3. **Generation:** Use `trl.experimental.openenv.generate_rollout_completions()` to produce the model's guess efficiently.  
# 4. **Feedback extraction:** Parse the environment's response using helpers like `extract_guess()` and `extract_wordle_feedback()`.  
# 5. **Reward calculation:** Compute rewards based on correctness, green/yellow feedback, and repetition penalty.
# 6. **Return structured rollout data:** Includes prompt/completion IDs, logprobs, `env_mask`, and all computed reward components.
# 
# **Important: The `env_mask` mechanism**
# 
# In multi-turn environments like Wordle, the completion includes both:
# - **Model-generated tokens** (the guesses): These should contribute to the loss during training.
# - **Environment feedback tokens** (game responses): These should NOT contribute to the loss.
# 
# The `env_mask` is a list of 1s and 0s that marks which tokens are model-generated (`1`) vs environment-generated (`0`).  
# The GRPOTrainer uses this mask to exclude environment tokens from the loss calculation, ensuring the model only learns from its own outputs.
# 
# This modular design ensures that each episode can be processed independently while still providing rich feedback for the **GRPO training loop**.

# In[ ]:


import re
from textarena_env import TextArenaAction
from textarena_env.rewards import extract_feedback_counts, extract_guess, extract_wordle_feedback
from trl.experimental.openenv import generate_rollout_completions

def rollout_once(trainer, env, tokenizer, dataset_prompt, system_prompt, max_turns, max_new_tokens):
    result = env.reset()
    observation = result.observation

    prompt_ids = []
    completion_ids = []
    logprobs = []
    env_mask = []  # 1 for model-generated tokens, 0 for environment tokens
    model_outputs = []
    raw_rewards = []
    position_scores = []
    correct_scores = []
    prev_env_output_len = 0  # Track length to only add NEW portion each turn

    accumulated_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    # Build initial prompt (only once, at the start)
    # The initial env messages are included in the prompt, not completion
    base_prompt = observation.prompt or dataset_prompt
    initial_user_prompt = make_user_prompt(observation.messages)
    # Track initial env output length so we don't add it again
    initial_env_output = format_history(observation.messages) if observation.messages else ""
    prev_env_output_len = len(initial_env_output)
    initial_messages = accumulated_messages + [{"role": "user", "content": initial_user_prompt}]
    initial_prompt_text = tokenizer.apply_chat_template(
        initial_messages,
        add_generation_prompt = True,
        tokenize = False,
        enable_thinking = False,
    )
    # Tokenize initial prompt once - this is the base prompt for the entire episode.
    # GRPO expects one prompt-completion pair per episode, where:
    # - prompt_ids = the initial/base prompt (what the model sees at episode start)
    # - completion_ids = all model responses + env feedback from all turns concatenated
    # Note: The actual prompts used for generation in each turn are longer (include conversation history),
    # but we only count the initial prompt tokens here.
    initial_prompt_ids = tokenizer.encode(initial_prompt_text, add_special_tokens = False)
    prompt_ids.extend(initial_prompt_ids)

    for _turn in range(max_turns):
        if result.done:
            break

        base_prompt = observation.prompt or dataset_prompt
        user_prompt = make_user_prompt(observation.messages)
        messages = accumulated_messages + [{"role": "user", "content": user_prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            tokenize = False,
            enable_thinking = False,
        )

        rollout_outputs = generate_rollout_completions(
            trainer, [prompt_text], generation_overrides = {"max_tokens": max_new_tokens}
        )[0]
        # Add model-generated completion tokens and logprobs with newlines for readability
        newline_tokens = tokenizer.encode("\n", add_special_tokens = False)
        completion_ids.extend(newline_tokens)  # newline before guess
        logprobs.extend([0.0] * len(newline_tokens))
        env_mask.extend([0] * len(newline_tokens))  # synthetic delimiter, not model-generated

        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])
        env_mask.extend([1] * len(rollout_outputs["completion_ids"]))  # model-generated tokens

        completion_ids.extend(newline_tokens)  # newline after guess
        logprobs.extend([0.0] * len(newline_tokens))
        env_mask.extend([0] * len(newline_tokens))  # synthetic delimiter, not model-generated
        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens = True
        )
        guess = extract_guess(completion_text)
        model_outputs.append(completion_text.strip())  # Store raw model output for format reward

        result = env.step(TextArenaAction(message = guess))

        raw_rewards.append(float(result.reward or 0.0))
        observation = result.observation
        correct_score = float(result.reward or 0.0)
        feedback = extract_wordle_feedback(observation)

        full_env_output = format_history(observation.messages) if observation.messages else ""
        new_env_output = full_env_output[prev_env_output_len:].lstrip("\n")
        prev_env_output_len = len(full_env_output)

        if new_env_output:
            env_output_tokens = tokenizer.encode(new_env_output, add_special_tokens = False)
            completion_ids.extend(env_output_tokens)  # Add to completion_ids
            logprobs.extend([0.0] * len(env_output_tokens))  # Placeholder (ignored via env_mask = 0)
            env_mask.extend([0] * len(env_output_tokens))  # Environment tokens - mask out from loss
            completion_with_env = completion_text + "\n" + new_env_output
        else:
            completion_with_env = completion_text

        accumulated_messages.append({"role": "user", "content": user_prompt})
        accumulated_messages.append({"role": "assistant", "content": completion_with_env})

        if not feedback:
            position_score = 0.0
        else:
            green_count, yellow_count = extract_feedback_counts(feedback)
            position_score = (green_count + 0.5 * yellow_count) / 5.0

        position_scores.append(position_score)
        correct_scores.append(correct_score)

    # Use the final correct reward (win/lose is binary at end)
    correct_reward_value = correct_scores[-1] if correct_scores else (raw_rewards[-1] if raw_rewards else 0.0)

    # Position reward as shaping signal:
    # - If model WINS: position_reward = 1.0 (no penalty for winning fast)
    # - If model LOSES: position_reward = last attempt (where it ended up)
    if correct_reward_value >= 1.0:
        final_position_reward = 1.0
    else:
        final_position_reward = position_scores[-1] if position_scores else 0.0

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "env_mask": env_mask,
        "raw_rewards": raw_rewards,
        "correct_reward": correct_reward_value,
        "position_reward": final_position_reward,
        "model_outputs": model_outputs,
    }


# ### Helper functions
# 
# Supporting utilities used in `rollout_once`:
# 
# - **`make_user_prompt`**: builds the user prompt combining the conversation history.
# - **`format_history`**: formats the conversation log for consistent context.

# In[ ]:


# @title Helpers definition (click to expand)
def format_history(messages) -> str:
    lines = []
    for message in messages:
        tag = message.category or "MESSAGE"
        content = message.content.strip()
        if not content:
            continue
        lines.append(f"[{tag}] {content}")
    return "\n".join(lines)


def make_user_prompt(messages) -> str:
    history = format_history(messages)
    # Only use messages for conversation history - the prompt is already included as the first message
    history_section = history if history else "[PROMPT] Awaiting first feedback."
    return f"Conversation so far:\n{history_section}\n\nReply with your next guess enclosed in square brackets."


# ## Define reward functions
# 
# To guide the agent's learning process, we define simple reward functions that map the feedback from the environment into numeric signals.  
# Each function corresponds to a specific aspect of the **Wordle** game:
# 
# - ✅ **`reward_correct`**: rewards the model when it guesses the correct word (binary: 0 or 1).  
# - 🎯 **`reward_position`**: rewards progress based on letter feedback. Green letters worth 1.0, yellow worth 0.5, normalized by 5. If the model wins, this is set to 1.0.
# - 📝 **`reward_format_strict`**: rewards correct output format `[xxxxx]`. Returns proportion of correctly formatted outputs across all turns.
# 
# These functions return lists of float values that the **GRPOTrainer** uses during optimization.  
# By combining them, the model learns to balance correctness, information gathering, and proper formatting in its guessing strategy.

# In[ ]:


os.environ['STEP_NUM'] = '0'
def reward_correct(completions, **kwargs):
    """Reward from environment (correct answer)."""
    rewards = kwargs.get("correct_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    os.environ['STEP_NUM'] = str(int(os.environ['STEP_NUM'])+1)
    if int(os.environ['STEP_NUM'])%10==0:
        print(f'{completions=}')
    return [float(r) for r in rewards]


def reward_position(completions, **kwargs):
    """Position reward: green worth 1.0, yellow worth 0.5, normalized by 5."""
    rewards = kwargs.get("position_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def compute_format_reward(model_outputs):
    """Compute format reward from a list of model outputs (one per turn).

    Each output should have a <reasoning> block and a [guess].
    Returns proportion of correctly formatted outputs.
    """
    if not model_outputs:
        return 0.0

    exact_pattern = re.compile(r"^.*<reasoning>.*?</reasoning>\s*\[[A-Za-z]{5}\]\s*$", re.DOTALL)
    correct_count = sum(1 for output in model_outputs if exact_pattern.match(output))

    return correct_count / len(model_outputs)


def reward_format_strict(completions, **kwargs):
    """Format reward - pre-computed in rollout_func."""
    rewards = kwargs.get("format_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


# ## Create dataset
# 
# We create a dataset with repeated prompts to control the number of training episodes.  
# Each entry in the dataset triggers one rollout episode during training. The `dataset_prompt` provides the initial instruction to the model before each game starts.

# In[ ]:


from datasets import Dataset

dataset_size = 3000
dataset_prompt = "Play Wordle like an expert."

dataset = Dataset.from_dict({"prompt": [dataset_prompt] * dataset_size})


# ## Set GRPO Config
# 
# Next, we define the **GRPOConfig**, which controls all key training parameters.  
# This configuration specifies how the model interacts with **vLLM**, manages memory, and logs results.

# In[ ]:


from trl import GRPOConfig

grpo_config = GRPOConfig(
    # Training schedule / optimization
    num_train_epochs = 1,                     # Number of full dataset passes
    learning_rate = 1e-5,                     # Learning rate for the optimizer
    gradient_accumulation_steps = 4,         # Accumulate gradients over multiple steps
    per_device_train_batch_size = 1,          # Batch size per GPU (number of prompts processed together)
    warmup_steps = 20,                        # Steps for learning rate warmup
    optim = "adamw_8bit",                      # Optimizer
    max_grad_norm = 1.0,                        # Clip gradients to prevent explosion

    # GRPO configuration
    num_generations = 4,                      # Number of rollout episodes per prompt (for variance reduction)
    max_completion_length = 1024,               # Full episode length, not per-turn
    log_completions = False,                  # Log completions for debugging

    # Logging / reporting
    output_dir = 'outputs',                  # Directory for checkpoints and logs
    report_to = "trackio",                      # Experiment tracking tool (integrates with HF Spaces)
    trackio_space_id = 'outputs',            # HF Space where experiment tracking will be saved
    logging_steps = 10,                        # Log metrics every N steps
    # save_steps = 10,                          # Interval for saving checkpoints

    top_p = 0.95,
    top_k = 50,

    gradient_checkpointing = True,            # Enable activation recomputation to save memory
    beta = 0.02
)


# ## Create `GRPOTrainer` and start training
# 
# Now we initialize the `GRPOTrainer`, which manages the entire reinforcement learning loop.
# 
# It takes the model, tokenizer, reward functions, rollout function, and dataset defined earlier.  
# The trainer coordinates the interaction between the model and the environment, applies the reward signals, and updates the policy.
# 
# Finally, we call `trainer.train()` to start the fine-tuning process and let the model learn to play Wordle through feedback and iteration.

# In[ ]:


import sys
sys.stdout.fileno = lambda: 1
sys.stderr.fileno = lambda: 2


# In[ ]:


from trl import GRPOTrainer

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        reward_correct,
        reward_position,
        reward_format_strict,
    ],
    train_dataset = dataset,
    args = grpo_config,
    rollout_func = rollout_func,
)


# In[ ]:


type(trainer.processing_class)


# Show memory stats before training

# In[ ]:


import torch
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# And train!

# In[ ]:


trainer_stats = trainer.train()


# Show memory stats after training

# In[ ]:


used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_training = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
training_memory_percentage = round(used_memory_for_training / max_memory * 100, 3)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_training} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {training_memory_percentage} %.")


# In[ ]:


env.close()
trainer.save_model(grpo_config.output_dir)
# trainer.push_to_hub() # Uncomment to push to HF Hub


# ## Inference
# 
# Now let's test our fine-tuned model by running **inference** using the model already in memory.  
# We switch the model to inference mode with `FastLanguageModel.for_inference(model)` and play a game of Wordle.

# In[ ]:


# Use the model already in memory (no need to reload from disk)
FastLanguageModel.for_inference(model)


# Now that we have the fine-tuned model loaded, we can start playing Wordle.  
# To make this easier, we'll define a reusable function so we can play multiple rounds.  
# The function implements the same logic we explored earlier.

# In[ ]:


MAX_TURNS = 6

def play_wordle(env, model, tokenizer):
    result = env.reset()
    observation = result.observation

    print("📜 Initial Prompt:\n" + observation.prompt)

    for turn in range(MAX_TURNS):
        if result.done:
            break

        user_prompt = make_user_prompt(observation.messages)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            tokenize = False,
            enable_thinking = False,
        )

        model_inputs = tokenizer([prompt_text], return_tensors = "pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens = 512
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]

        # Decode and extract model response
        generated_text = tokenizer.decode(output_ids, skip_special_tokens = True)
        guess = extract_guess(generated_text)

        print(f"\n🎯 Turn {turn}: model replied with -> {generated_text}")
        print(f"   Parsed guess: {guess}")

        result = env.step(TextArenaAction(message = guess))
        observation = result.observation

        print("   Feedback messages:")
        for message in observation.messages:
            print(f"     [{message.category}] {message.content}")

    print("\n✅ Game finished")
    print(f"   Reward: {result.reward}")
    print(f"   Done: {result.done}")


# Let's play the game!

# In[ ]:


# Re-open env for inference (training env was closed)
inference_env = TextArenaEnv(base_url = wordle_url).sync()
try:
    play_wordle(inference_env, model, tokenizer)
finally:
    inference_env.close()


# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other resources:
# 1. Looking to use Unsloth locally? Read our [Installation Guide](https://unsloth.ai/docs/get-started/install) for details on installing Unsloth on Windows, Docker, AMD, Intel GPUs.
# 2. Learn how to do Reinforcement Learning with our [RL Guide and notebooks](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide).
# 3. Read our guides and notebooks for [Text-to-speech (TTS)](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning) and [vision](https://unsloth.ai/docs/basics/vision-fine-tuning) model support.
# 4. Explore our [LLM Tutorials Directory](https://unsloth.ai/docs/models/tutorials-how-to-fine-tune-and-run-llms) to find dedicated guides for each model.
# 5. Need help with Inference? Read our [Inference & Deployment page](https://unsloth.ai/docs/basics/inference-and-deployment) for details on using vLLM, llama.cpp, Ollama etc.
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
