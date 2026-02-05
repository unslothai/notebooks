#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # <img width="35" height="35" alt="image" src="https://github.com/user-attachments/assets/2700a971-e5d6-4036-b03f-2f89c9791609" /> OpenEnv: Agentic Execution Environments
# We're using the new [OpenEnv](https://github.com/meta-pytorch/OpenEnv) library which has over 2000+ environments for RL!
# 
# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth on your local device, follow [our guide](https://unsloth.ai/docs/get-started/install).

# # Goal: Make gpt-oss play games with Reinforcement Learning
# 
# Our goal is to make OpenAI's open model gpt-oss 20b play the 2048 game with reinforcement learning. We want the model to devise a strategy to play 2048, and we will run this strategy until we win or lose.
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/2048_win.png/500px-2048_win.png" height=300 />

# # Installation
# We'll be using [Unsloth](https://github.com/unslothai/unsloth) to do RL on GPT-OSS 20B, and [OpenEnv](https://github.com/meta-pytorch/OpenEnv) for the environment interactions. Unsloth saves 70% VRAM usage and makes reinforcement learning 2 to 6x faster!

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'import os, importlib.util\n\n!pip install --upgrade -qqq uv\nif importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):\n    try:\n        import numpy\n\n        get_numpy = f"numpy=={numpy.__version__}"\n    except:\n        get_numpy = "numpy"\n    !uv pip install -qqq \\\n        "torch>=2.8.0" "triton>=3.4.0" {get_numpy} torchvision bitsandbytes "transformers==4.56.2" trackio \\\n        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\\n        "unsloth[base] @ git+https://github.com/unslothai/unsloth" \\\n        git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels\nelif importlib.util.find_spec("unsloth") is None:\n    !uv pip install -qqq unsloth trackio\n!uv pip install --upgrade --no-deps transformers==4.56.2 tokenizers trl==0.22.2 unsloth unsloth_zoo\n')


# We will then install [OpenEnv](https://github.com/meta-pytorch/OpenEnv) and connect to the OpenSpiel environment on Hugging Face:

# In[ ]:


get_ipython().run_cell_magic('capture', '', '!pip install -qqq git+https://huggingface.co/spaces/openenv/openspiel_env\n')


# We'll load GPT-OSS 20B and set some parameters:
# * `max_seq_length = 768` The maximum context length of the model. Increasing it will use more memory.
# * `lora_rank = 4` The larger this number, the smarter the RL process, but the slower and more memory usage`load_in_16bit` will be faster but will need a 64GB GPU or more (MI300)
# * `offload_embedding = True` New Unsloth optimization which moves the embedding to CPU RAM, reducing VRAM by 1GB.

# In[ ]:


import os
from unsloth import FastLanguageModel
import torch

max_seq_length = 768  # Can increase for longer RL output
lora_rank = 4  # Larger rank = smarter, but slower
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gpt-oss-20b",
    load_in_4bit=True,
    max_seq_length=max_seq_length,
    offload_embedding=True,  # Offload embeddings to save more VRAM
)


# To do efficient RL, we will use [LoRA](https://arxiv.org/abs/2106.09685), which allows us to only add 1 to 5% of extra weights to the model for finetuning purposes. This allows us to save memory usage by over 60%, and yet it retains good accuracy. Read Unsloth's [GPT-OSS RL Guide](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning) for more details.

# In[ ]:


model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_rank * 2,  # *2 speeds up training
    use_gradient_checkpointing="unsloth",  # Reduces memory usage
    random_state=3407,
)


# # 2048 game environment with OpenEnv
# 
# We connect to the OpenSpiel 2048 environment via websocket! This will allow us to see how the 2048 implementation looks like!

# In[ ]:


from openspiel_env import OpenSpielEnv
from openspiel_env.models import OpenSpielAction, OpenSpielObservation


# You can use the hosted Space or run locally:

# In[ ]:


# Connect to OpenSpiel 2048 environment on HuggingFace Spaces
# The game is configured server-side via OPENSPIEL_GAME=2048
OPENSPIEL_URL = "https://openenv-openspiel-env.hf.space"
# For local: OPENSPIEL_URL = "http://localhost:8000"

env = OpenSpielEnv(base_url=OPENSPIEL_URL)


# Let's see how the current 2048 game state looks like:

# In[ ]:


result = env.reset()
current_state = result.observation
current_state


# First let's convert the state into a list of list of numbers!

# In[ ]:


import numpy as np


def convert_to_board(current_state):
    n = len(current_state.info_state)
    size = int(np.sqrt(n))
    board = np.array_split(np.array(current_state.info_state, dtype=int), size)
    board = [x.tolist() for x in board]
    return board, size


convert_to_board(current_state)


# We also want to pretty print the game board!

# In[ ]:


# @title (Collapsible) 2048 Game Renderer
def render_board(obs, colors: bool = True, border: bool = True, dot_for_zero: bool = True) -> str:
    """
    Pretty-print the board with colors that scale from 0 up to self.target.
    Uses ANSI 256-color codes (works in most terminals). Set colors=False to disable.
    """
    import math

    b, size = convert_to_board(obs)
    mx = max((max(row) for row in b), default=0)
    cell_w = max(3, len(str(mx)))

    RESET = "\x1b[0m"

    # A smooth-ish gradient from cool → warm
    # (blue/cyan/green → yellow/orange/red). Tweak or expand as you like.
    GRAD = [33, 39, 45, 51, 50, 49, 48, 47, 46, 82, 118, 154, 190, 226, 220, 214, 208, 202, 196]
    ZERO_FG = 239  # dim gray

    def color_code(v: int) -> str:
        if not colors:
            return ""
        if v == 0:
            return f"\x1b[38;5;{ZERO_FG}m"
        # Normalize by exponent relative to target: r in [0,1]
        t = max(2, 2048)  # safety; avoid log2(1)
        # Guard: if v is not a power of two or is <1, handle gracefully
        try:
            r = max(0.0, min(1.0, math.log2(v) / math.log2(t)))
        except ValueError:
            r = 0.0
        idx = int(round(r * (len(GRAD) - 1)))
        return f"\x1b[38;5;{GRAD[idx]}m"

    def fmt(v: int) -> str:
        s = "." if (v == 0 and dot_for_zero) else str(v)
        s = s.rjust(cell_w)
        return color_code(v) + s + (RESET if colors else "")

    def hline(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * cell_w for _ in range(size)) + right

    rows = []
    if border:
        rows.append(hline("┌", "┬", "┐"))
    for r in range(size):
        content = "│".join(fmt(v) for v in b[r])
        rows.append(("│" + content + "│") if border else content)
        if border:
            rows.append(
                hline("└" if r == size - 1 else "├", "┴" if r == size - 1 else "┼", "┘" if r == size - 1 else "┤")
            )
    return "\n".join(rows)


# In[11]:


print(render_board(current_state))


# We can see the `legal_actions` ie what you can take as `[0, 1, 2, 3]` Let's try doing the action `0`.

# In[ ]:


action = OpenSpielAction(action_id=0, game_name="2048")
result = env.step(action)
current_state = result.observation
print(render_board(current_state))


# So it looks like `0` is a move up action! Let's try `1`.

# In[ ]:


action = OpenSpielAction(action_id=1, game_name="2048")
result = env.step(action)
current_state = result.observation
print(render_board(current_state))


# `1` is a move right action. And `2`:

# In[ ]:


action = OpenSpielAction(action_id=2, game_name="2048")
result = env.step(action)
current_state = result.observation
print(render_board(current_state))


# `2` is a move down. And I guess `3` is just move left!

# In[ ]:


action = OpenSpielAction(action_id=3, game_name="2048")
result = env.step(action)
current_state = result.observation
print(render_board(current_state))


# We can also print the game status which indicates if no more moves are possible, and also the possible actions you can take!

# In[16]:


print(current_state.done)
print(current_state.legal_actions)


# # RL Environment Setup
# 
# We'll set up a function to accept some strategy that'll emit an action within `0123` and check the game state.
# 
# We'll also add a timer to only execute the strategy for 2 seconds maximum, otherwise it might never terminate!

# In[ ]:


from typing import Callable
from unsloth import execute_with_time_limit
import itertools


def _execute_strategy(strategy, current_state: OpenSpielObservation):
    assert callable(strategy)

    steps = 0
    total_reward = 0
    while not current_state.done:
        board, size = convert_to_board(current_state)
        action = strategy(board)
        try:
            action = int(action)
        except:
            return steps, False
        steps += 1
        if type(action) is not int or action not in current_state.legal_actions:
            return steps, max(itertools.chain.from_iterable(board)) == 2048

        action = OpenSpielAction(action_id=action, game_name="2048")
        result = env.step(action)
        current_state = result.observation
        if result.reward is not None:
            total_reward += result.reward
    return steps, max(itertools.chain.from_iterable(board)) == 2048


@execute_with_time_limit(2)
def execute_strategy(strategy: Callable, current_state: OpenSpielObservation):
    return _execute_strategy(strategy, current_state)


# Let's make a generic strategy to just hit `3`. We should expect this generic strategy to fail:

# In[ ]:


def always_move_left(board):
    return 3


# Reset OpenEnv to an initial state!
result = env.reset()
current_state = result.observation
try:
    steps, if_done = execute_strategy(always_move_left, current_state)
except TimeoutError as e:
    print(f"Timed out with error = {str(e)}")

steps, if_done


# To allow longer strategies for GPT-OSS Reinforcement Learning, we shall allow a 5 second timer.

# In[ ]:


@execute_with_time_limit(5)
def execute_strategy(strategy: Callable, current_state: OpenSpielObservation):
    return _execute_strategy(strategy, current_state)


# # Code Execution
# 
# To execute and create a new Python function, we first have to check if the function does not call other global variables or cheat. This is called `countering reward hacking` since we don't want the function to cheat.
# 
# For example the below piece of code is fine, since it only imports Python level functions. We use `check_python_modules`:

# In[20]:


from unsloth import check_python_modules

sample = """
def strategy(board):
    import math
    from typing import Callable
    return "0"
"""
ok, info = check_python_modules(sample)
print("Only Python imports?", ok)
print(info)


# For the below piece of code, since we import `numpy`, we should not allow the execution:

# In[21]:


sample = """
def strategy(board):
    from numpy import matmul
    return "0"
"""
ok, info = check_python_modules(sample)
print("Only Python imports?", ok)
print(info)


# We also disallow global variable access. We'll use Unsloth's `create_locked_down_function` function
# 

# In[ ]:


from unsloth import create_locked_down_function

function = """
def import_numpy():
    np.matmul
    print("Success")
"""
f = create_locked_down_function(function)
try:
    f()
except Exception as e:
    print(str(e))


# In[ ]:


from unsloth import create_locked_down_function

function = """
def add(a, b):
    def adder(a):
        return a + b
    return adder(b) + b
"""
f = create_locked_down_function(function)
try:
    print(f(10, 20))
except Exception as e:
    print(str(e))


# # Data & RL task setup
# 
# We now have to create a prompt to tell the model to create a strategy for the 2048 game. You can customize this to some other task for another RL task.

# In[24]:


prompt = """
Create a new short 2048 strategy using only native Python code.
You are given a list of list of numbers for the current board state.
Output one action for "0", "1", "2", "3" on what is the optimal next step.
Output your new short function in backticks using the format below:
```python
def strategy(board):
    return "0" # Example
```
All helper functions should be inside def strategy. Only output the short function `strategy`.
""".strip()
print(prompt)


# First, let's prompt GPT-OSS without RL and see how it goes:

# In[ ]:


text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
    reasoning_effort="low",
)

from transformers import TextStreamer

_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    temperature=1.0,
    max_new_tokens=512,
    streamer=TextStreamer(tokenizer, skip_prompt=False),
)


# # Reward functions
# 
# We now design a `extract_function` function which simply extracts the function wrapped in 3 back ticks.
# 
# And 3 reward functions:
# 
# 1. `function_works` which rewards the model if the strategy is a valid Python function.
# 2. `no_cheating` which checks if the function imported other modules, and if it did, we penalize it.
# 3. `strategy_succeeds` which checks if the game strategy actually succeeds in attaining 2048 after running the auto-generated strategy.

# In[ ]:


def extract_function(text):
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first:second].strip()
        fx = fx.removeprefix("python\n")
        fx = fx[fx.find("def") :]
        if fx.startswith("def strategy(board):"):
            return fx
    return None


print(extract_function(prompt))


# Below is our `function_works` reward function which uses Python's `exec` but guarded by not allowing leakage of local and global variables. We can also use `check_python_modules` first to check if there are errors before even executing the function:

# In[27]:


ok, info = check_python_modules("def a")
ok, info


# In[28]:


def function_works(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_python_modules(function)
        if function is None or "error" in info:
            score = -2.0
        else:
            try:
                new_strategy = create_locked_down_function(function)
                score = 1.0
            except:
                score = -0.5
        scores.append(score)
    return scores


# `no_cheating` checks if the function cheated since it might have imported Numpy or other functions:

# In[ ]:


def no_cheating(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_python_modules(function)
            scores.append(1.0 if ok else -20.0)  # Penalize heavily!
        else:
            scores.append(-1.0)  # Failed creating function
    return scores


# Next `strategy_succeeds` checks if the strategy actually allows the game to terminate. Imagine if the strategy simply returned "0" which would fail after a time limit of 10 seconds.
# 
# We also add a global `PRINTER` to print out the strategy and board state.

# In[ ]:


import numpy as np

global PRINTER
PRINTER = 0


def strategy_succeeds(completions, **kwargs):
    global PRINTER
    scores = []
    for completion in completions:
        printed = False
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        if PRINTER % 5 == 0:
            printed = True
            print(function)
        PRINTER += 1
        if function is not None:
            ok, info = check_python_modules(function)
        if function is None or "error" in info:
            scores.append(0)
            continue
        try:
            new_strategy = create_locked_down_function(function)
        except:
            scores.append(0)
            continue
        try:
            # Reset OpenEnv to an initial state!
            result = env.reset()
            current_state = result.observation
            steps, if_done = execute_strategy(new_strategy, current_state)
            print(f"Steps = {steps} If Done = {if_done}")
            if printed is False:
                print(function)
            print(render_board(current_state))
            if if_done:
                scores.append(20.0)  # Success - massively reward!
            else:
                scores.append(2.0)  # Failed but function works!
        except TimeoutError as e:
            print("Timeout")
            scores.append(-1.0)  # Failed with timeout
        except Exception as e:
            print(f"Exception = {str(e)}")
            scores.append(-3.0)  # Failed
    return scores


# We'll now create the dataset which includes a replica of our prompt. Remember to add a reasoning effort of low! You can choose high reasoning mode, but this'll only work on more memory GPUs like MI300s.

# In[ ]:


from datasets import Dataset

dataset = Dataset.from_list(
    [{"prompt": [{"role": "user", "content": prompt.strip()}], "answer": 0, "reasoning_effort": "low"}] * 1000
)
maximum_length = len(
    tokenizer.apply_chat_template([{"role": "user", "content": prompt.strip()}], add_generation_prompt=True)
)
print(maximum_length)
dataset[0]


# <a name="Train"></a>
# ### Train the model
# 
# Now set up GRPO Trainer and all configurations! We also support GSPO, GAPO, Dr GRPO and more! Go the Unsloth [Reinforcement Learning Docs](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) for more options.
# 
# We're also using [TrackIO](https://github.com/gradio-app/trackio) which allows you to visualize all training metrics straight inside the notebook fully locally!

# In[ ]:


max_prompt_length = maximum_length + 1  # + 1 just in case!
max_completion_length = max_seq_length - max_prompt_length

from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    temperature=1.0,
    learning_rate=2e-4,
    weight_decay=0.001,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # Increase to 4 for smoother training
    num_generations=2,  # Decrease if out of memory
    max_prompt_length=max_prompt_length,
    max_completion_length=max_completion_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps=600,
    save_steps=100,
    report_to="trackio",  # Can use Weights & Biases, TrackIO
    output_dir="outputs",
    # For optional training + evaluation
    # fp16_full_eval = True,
    # per_device_eval_batch_size = 4,
    # eval_accumulation_steps = 1,
    # eval_strategy = "steps",
    # eval_steps = 1,
)


# And let's run the trainer! If you scroll up, you'll see a table of rewards. The goal is to see the `reward` column increase!
# 
# You might have to wait 150 to 200 steps for any action. You'll probably get 0 reward for the first 100 steps. Please be patient!
# 
# | Step | Training Loss | reward    | reward_std | completion_length | kl       |
# |------|---------------|-----------|------------|-------------------|----------|
# | 1    | 0.000000      | 0.125000  | 0.000000   | 200.000000        | 0.000000 |
# | 2    | 0.000000      | 0.072375  | 0.248112   | 200.000000        | 0.000000 |
# | 3    | 0.000000      | -0.079000 | 0.163776   | 182.500000        | 0.000005 |
# 

# In[ ]:


# For optional training + evaluation
# new_dataset = dataset.train_test_split(test_size = 0.01)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        function_works,
        no_cheating,
        strategy_succeeds,
    ],
    args=training_args,
    train_dataset=dataset,
    # For optional training + evaluation
    # train_dataset = new_dataset["train"],
    # eval_dataset = new_dataset["test"],
)


# And let's train the model! **NOTE** This might be quite slow! 600 steps takes ~5 hours or longer.
# 
# [TrackIO](https://github.com/gradio-app/trackio) might be a bit slow to load - wait 2 minutes until the graphs pop up!

# In[34]:


trainer.train()


# <a name="Inference"></a>
# # Inference
# Now let's try the model we just trained!

# In[ ]:


text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
    reasoning_effort="low",
)

from transformers import TextStreamer

_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    temperature=1.0,
    max_new_tokens=1024,
    streamer=TextStreamer(tokenizer, skip_prompt=False),
)


# <a name="Save"></a>
# ### Saving to float16 or `MXFP4`
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `mxfp4` for MXFP4 (OpenAI's GPT-OSS native precision). We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.

# In[ ]:


# Merge and push to hub in mxfp4 4bit format
if False:
    model.save_pretrained_merged("finetuned_model", tokenizer, save_method="mxfp4")
if False:
    model.push_to_hub_merged("repo_id/repo_name", tokenizer, token="hf...", save_method="mxfp4")

# Merge and push to hub in 16bit
if False:
    model.save_pretrained_merged("finetuned_model", tokenizer, save_method="merged_16bit")
if False:  # Pushing to HF Hub
    model.push_to_hub_merged("hf/gpt-oss-finetune", tokenizer, save_method="merged_16bit", token="")


# # And we're done!
# Congratulations you just learned how to do reinforcement learning with GPT-OSS! There were some advanced topics explained in this notebook - to learn more about GPT-OSS and RL, there are more docs in Unsloth's [Reinforcement Learning Guide with GPT-OSS](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning)
# 
# This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
