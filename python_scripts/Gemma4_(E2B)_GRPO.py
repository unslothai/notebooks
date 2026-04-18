#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth on your local device, follow [our guide](https://unsloth.ai/docs/get-started/install). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & how to save it

# # ### Installation
# 
# # In[ ]:
# 
# 
# get_ipython().run_cell_magic('capture', '', 'import os, re\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth  # Do this in local & cloud setups\nelse:\n    import torch; v = re.match(r\'[\\d]{1,}\\.[\\d]{1,}\', str(torch.__version__)).group(0)\n    xformers = \'xformers==\' + {\'2.10\':\'0.0.34\',\'2.9\':\'0.0.33.post1\',\'2.8\':\'0.0.32.post2\'}.get(v, "0.0.34")\n    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer\n    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth\n!pip install --no-deps transformers==5.5.0\n!pip install torchcodec\nimport torch; torch._dynamo.config.recompile_limit = 64;\n')
# 
# 
# # In[ ]:
# 
# 
# #@title Colab Extra Install { display-mode: "form" }
# get_ipython().run_line_magic('%capture', '')
# import os
# get_ipython().system('pip install --upgrade -qqq uv')
# if "COLAB_" not in "".join(os.environ.keys()):
#     # If you're not in Colab, just use pip install!
#     get_ipython().system('pip install unsloth vllm')
# else:
#     try: import numpy, PIL; _numpy = f'numpy=={numpy.__version__}'; _pil = f'pillow=={PIL.__version__}'
#     except: _numpy = "numpy"; _pil = "pillow"
#     try: import subprocess; is_t4 = "Tesla T4" in str(subprocess.check_output(["nvidia-smi"]))
#     except: is_t4 = False
#     _vllm, _triton = ('vllm==0.9.2', 'triton==3.2.0') if is_t4 else ('vllm==0.15.1', 'triton')
#     get_ipython().system('uv pip install -qqq --upgrade {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers unsloth')
#     get_ipython().system('uv pip install -qqq {_triton}')
# get_ipython().system('uv pip install transformers==4.56.2')
# get_ipython().system('uv pip install --no-deps trl==0.22.2')
# 
# 
# # ### Unsloth

# # Goal: Make faster kernels with Reinforcement Learning
# 
# Our goal is to make a faster matrix multiplication kernel by doing RL on Gemma 4 with Unsloth.
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Matrix_multiplication_qtl1.svg/500px-Matrix_multiplication_qtl1.svg.png" height=200 />
# 
# You will learn how to:
# 1. Counteract **reward hacking** like cheating, caching, laziness.
# 2. Timing and correctness of kernels and time limits.
# 3. Making good **reward functions**
# 4. How to seriously do RL to make optimized kernels

# In[ ]:


from unsloth import FastVisionModel
import torch
max_seq_length = 4096 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

gemma4_models = [
    # Gemma-4 instruct models:
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B-it",
    "unsloth/gemma-4-31B-it",
    "unsloth/gemma-4-26B-A4B-it",
    # Gemma-4 base models:
    "unsloth/gemma-4-E2B",
    "unsloth/gemma-4-E4B",
    "unsloth/gemma-4-31B",
    "unsloth/gemma-4-26B-A4B",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "unsloth/gemma-4-E2B-it",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = False, # Enable vllm fast inference
)


# We now add some small amount of LoRA weights to Gemma 4 so we only need to train those, instead of training on the full model.

# In[ ]:


model = FastVisionModel.get_peft_model(
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


# # Optimized matrix multiplication
# 
# Numpy has optimized matrix multiplication kernels for CPUs via BLAS optimized operations. For GPUs, one can use CUDA accelerated cuBLAS kernels which PyTorch calls under the hood.
# 
# To generate some random matrices to do matrix multiplication, we can do the below:

# In[ ]:


import numpy as np
def generate_random_matrices(seed = 3407, n = 256):
    random_state = np.random.RandomState(seed)
    n, k, m = random_state.randint(1, n+1, size = 3)
    A = np.random.uniform(-10, 10, size = (n, k))
    B = np.random.uniform(-10, 10, size = (k, m))
    return A, A.tolist(), B, B.tolist()


# We shall generate a small matrix, and see the matrix multiplied output

# In[ ]:


A, A_list, B, B_list = generate_random_matrices(seed = 42, n = 5)
print(A)
print(B)
print(np.matmul(A, B))


# We can call a LLM to generate a simple matrix multiply kernel in Python only, and we can calculate the differences between the actual result and the kernel's result

# In[ ]:


def calculate_difference(pred, real):
    if pred is None: return 5, 5
    assert real is not None
    import numpy as np
    try:
        difference = pred - real
    except:
        return 5, 5
    amax_error = float(np.amax(difference))
    mse_error  = float(np.mean(np.square(difference)))
    return amax_error, mse_error


# In[ ]:


# Kernel generated by GPT-5
def matmul(A, B):
    z, s = zip, sum
    Bt = list(z(*B))
    return [[s(a*b for a, b in z(row, col)) for col in Bt] for row in A]


# We see the error below is very small, so that's good!

# In[ ]:


prediction = matmul(A_list, B_list)
calculate_difference(prediction, np.matmul(A, B))


# # Countering Reward Hacking
# 
# The ultimate goal of RL is to maximize some reward (say speed, revenue, some metric).
# 
# But RL can **cheat** When the RL algorithm learns a trick or exploits something to increase the reward, without actually doing the task at end, this is called "Reward Hacking".
# 
# Some good examples are in https://en.wikipedia.org/wiki/Reward_hacking
# 
# For matrix multiplication kernels, we might see the following issues:
# 
# * Laziness: RL learns to use Numpy, Torch, other libraries, which calls optimized kernels.
# * Caching: RL learns to cache the result of the output
# * Cheating: RL learns to find the actual output by inspecting Python global variables
# * RL learns to edit the timing function to make it output 0 time as passed.
# 
# And possibly more. We shall try to address each!

# # Countering Reward Hacking 1: Stop laziness
# We can stop the RL algorithm from calling optimized code by inspecting if the generated code imports other non standard Python libraries. We used GPT-5 to help generate this check `check_only_stdlib_imports`:

# In[ ]:


#@title (Collapsible code)
import ast
import sys
import sysconfig
from pathlib import Path

def _stdlib_names():
    """
    Build a set of canonical stdlib top-level module/package names.
    Uses sys.stdlib_module_names when available (3.10+), with a
    filesystem fallback for older versions/edge cases.
    """
    names = {m.lower() for m in getattr(sys, "stdlib_module_names", set())}
    names |= {m.lower() for m in sys.builtin_module_names}
    names.add("__future__")  # special-case

    # Fallback/augmentation: scan the stdlib directory
    try:
        stdlib_dir = Path(sysconfig.get_path("stdlib"))
        if stdlib_dir.exists():
            for p in stdlib_dir.iterdir():
                if p.name == "site-packages":
                    continue
                if p.suffix == ".py":
                    names.add(p.stem.lower())
                elif p.is_dir() and (p / "__init__.py").exists():
                    names.add(p.name.lower())
    except Exception:
        # conservative fallback; the names set above will still work well
        pass

    return names

_STDLIB_SET = _stdlib_names()

def check_only_stdlib_imports(code: str):
    """
    Return (ok: bool, details: dict)

    ok == True  -> all absolute imports are from the stdlib.
    ok == False -> details['non_stdlib'] lists offending top-level modules.

    details includes:
      - stdlib: sorted list of stdlib imports found
      - non_stdlib: sorted list of non-stdlib imports found
      - relative_imports: count of relative imports (always allowed here)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, {
            "error": f"SyntaxError: {e}",
            "stdlib": [],
            "non_stdlib": [],
            "relative_imports": 0,
        }

    abs_imports = set()
    relative_count = 0

    class Visitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import):
            for alias in node.names:
                abs_imports.add(alias.name.split(".")[0])
        def visit_ImportFrom(self, node: ast.ImportFrom):
            nonlocal relative_count
            if (node.level or 0) > 0:
                # relative import
                relative_count += 1
            else:
                if node.module:
                    abs_imports.add(node.module.split(".")[0])

    Visitor().visit(tree)

    stdlib_found = sorted(m for m in abs_imports if m.lower() in _STDLIB_SET)
    non_stdlib = sorted(m for m in abs_imports if m.lower() not in _STDLIB_SET)

    return len(non_stdlib) == 0, {
        "stdlib": stdlib_found,
        "non_stdlib": non_stdlib,
        "relative_imports": relative_count,
    }


# For example, let's call `check_only_stdlib_imports` on a random piece of matrix multiplication code generated by GPT-5:

# In[ ]:


sample = """
def matmul(A, B):
    import numpy as np
    from torch import matmul
    z, s = zip, sum
    Bt = list(z(*B))
    return [[s(a*b for a, b in z(row, col)) for col in Bt] for row in A]
"""
ok, info = check_only_stdlib_imports(sample)
print("Only stdlib imports?", ok)
print(info)


# # Countering Reward Hacking 2: Stop cheating
# We can stop the RL algorithm from using global or cached variables by restricting it's `locals` and `globals`.
# 
# We are also going to use `exec` to create the function, so we have to save the output to an empty dict.
# 
# We also disallow global variable access.

# In[ ]:


output_function = {}
exec(sample, {}, output_function)
output_function["matmul"]


# We also disallow global variable access via `types.FunctionType(f.__code__, {})`

# In[ ]:


import types
output_function["matmul"] = types.FunctionType(output_function["matmul"].__code__, {})

def import_numpy():
    np.matmul
    print("Success")

import_numpy()
import_numpy = types.FunctionType(import_numpy.__code__, {})
try:
    import_numpy()
except Exception as e:
    print(str(e))


# In[ ]:


def create_locked_down_function(function):
    output_function = {}
    exec(function, {}, output_function)
    new_matmul = output_function["matmul"]
    new_matmul = types.FunctionType(new_matmul.__code__, {})
    return new_matmul


# # Countering Reward Hacking 3: Stop caching
# We can stop the RL algorithm from using cached data by wiping the cache with a large fake matrix. We also have to benchmark carefully with multiple loops and turns.
# 
# We also add a **timer** to not make the algorithm go in an endless loop.

# In[ ]:


import os, gc, time, statistics
import signal
from contextlib import contextmanager
class TimeoutError(Exception): pass

@contextmanager
def time_limit(seconds):
    def _handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old)

class Benchmarker:
    def __init__(self, trials = 3, loops = 1, timeout = 30):
        self.buffer = np.zeros(2 * 1024 * 1024 * 1024, dtype = np.uint8)
        self.trials = trials
        self.loops = loops
        assert timeout > 0 # Cannot be 0 since it won't work!
        self.timeout = timeout
    def thrash(self):
        # Edit the buffer to wipe cache lines
        self.buffer ^= 1
        return int(self.buffer[::4096].sum())

    def benchmark(self, function, arguments):
        assert len(arguments) == self.loops
        samples = []
        exceptions = []
        timed_out = 0
        for _ in range(self.trials):
            gc.collect(); gc.disable(); self.thrash()
            t_start = time.perf_counter_ns()
            for i in range(self.loops):
                try:
                    with time_limit(self.timeout):
                        function(*arguments[i])
                except TimeoutError as e:
                    timed_out += 1
                except Exception as e:
                    exceptions.append(str(e))
            t_end = time.perf_counter_ns()
            gc.enable()
            samples.append((t_end - t_start) // max(1, self.loops))
        return {
            "median_ns": int(statistics.median(samples)),
            "mean_ns": int(statistics.fmean(samples)),
            "stdev_ns": int(statistics.pstdev(samples) if len(samples) > 1 else 0),
            "exceptions" : exceptions,
            "timeouts" : timed_out,
        }


# For example we use our matmul kernel we had, and benchmark it with a 10 second delay:

# In[ ]:


A, A_list, B, B_list = generate_random_matrices(seed = 0, n = 256)
Benchmarker(trials = 1, timeout = 10).benchmark(output_function["matmul"], [(A_list, B_list)])


# # Data & RL task setup
# 
# We now have to create a prompt to the model for which it will do some task. For our matrix multiply example, we use the below:

# In[ ]:


prompt = """
Create a new fast matrix multiplication function using only native Python code.
You are given a list of list of numbers.
Output your new function in backticks using the format below:
```python
def matmul(A, B):
    return ...
```
""".strip()
print(prompt)


# First, let's prompt Gemma 4 without RL and see how it goes:

# In[ ]:


text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt.strip()}],
    tokenize = False,
    add_generation_prompt = True,
)

from transformers import TextStreamer
print("=" * 50)
print("BASE MODEL OUTPUT (before RL training):")
print("=" * 50)

inputs = tokenizer(
    text = text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt = True)
result = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 512,
                        use_cache = True, temperature = 1.0, top_p = 0.95, top_k = 64)


# # Reward functions
# 
# We now design the `extract_function` function which simply extracts the function wrapped in 3 backticks.
# 
# And 4 reward functions:
# 
# 1. `function_works` which rewards the model if the strategy is a valid Python function.
# 2. `no_cheating` which checks if the function imported other modules, and if it did, we penalize it.
# 3. `correctness_check` which checks if the kernel was correct or wrong - it shouldn't generate gibberish!
# 4. `speed_check` checks the performance relative to Numpy matmul directly.

# In[ ]:


def extract_function(text):
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first : second].strip()
        fx = fx.removeprefix("python\n")
        fx = fx[fx.find("def"):]
        if fx.startswith("def matmul(A, B):"): return fx
    return None
print(extract_function(prompt))


# Below is our `function_works` reward function which uses Python's `exec` but guarded by not allowing leakage of local and global variables. We can also use `check_only_stdlib_imports` first to check if there are errors before even executing the function:

# In[ ]:


ok, info = check_only_stdlib_imports("def a")
ok, info


# In[ ]:


def function_works(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        print(function)
        if function is not None:
            ok, info = check_only_stdlib_imports(function)
        if function is None or "error" in info:
            score = -2.0
        else:
            try:
                new_matmul = create_locked_down_function(function)
                score = 1.0
            except:
                score = -0.5
        scores.append(score)
    return scores


# `no_cheating` checks if the function cheated since it might have imported Numpy or Torch optimized code.

# In[ ]:


def no_cheating(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_only_stdlib_imports(function)
        else:
            ok = False
        scores.append(1.0 if ok else -20.0) # Penalize heavily!
    return scores


# Next `correctness_check` checks if the kernel was correct. We want to penalize if the absolute error is larger than 1, and if the mean squared error is somewhat bigger then machine epsilon.
# 
# We have to execute the code now!

# In[ ]:


np.finfo(np.float64).eps


# In[ ]:


def correctness_check(completions, **kwargs):
    scores = []
    # Generate some random matrices of size less than 128
    A, A_list, B, B_list = generate_random_matrices(seed = np.random.randint(10000), n = 128)
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_only_stdlib_imports(function)
        if function is None or "error" in info:
            scores.append(0)
            continue
        try:
            new_matmul = create_locked_down_function(function)
        except:
            scores.append(0)
            continue
        try:
            pred = new_matmul(A_list.copy(), B_list.copy())
        except:
            # Failed!
            scores.append(-2.0)
            continue
        true = np.matmul(A, B)
        amax_error, mse_error = calculate_difference(pred, true)

        # Check correctness and score!
        machine_epsilon = 100*np.finfo(np.float64).eps
        if   amax_error >= 3:   score = -3.0
        elif amax_error >= 2:   score = -2.5
        elif amax_error >= 1:   score = -2.0
        elif amax_error >= 0.5: score = -1.0
        elif amax_error >= 100*machine_epsilon: score = 0.0
        elif amax_error >= machine_epsilon: score = 1.0
        else: score = 3.0

        if   mse_error >= 3:   score += -3.0
        elif mse_error >= 2:   score += -2.5
        elif mse_error >= 1:   score += -2.0
        elif mse_error >= 0.5: score += -1.0
        elif mse_error >= 100*machine_epsilon: score += 0.0
        elif mse_error >= machine_epsilon: score += 1.0
        else: score += 3.0
        scores.append(score)
    return scores


# Finally our benchmarking function for `speed_check`! We shall limit the timer to 10 seconds and do 3 trials.

# In[ ]:


A, A_list, B, B_list = generate_random_matrices(seed = 0, n = 256)
benchmarker = Benchmarker(trials = 3, timeout = 10)
numpy_results = benchmarker.benchmark(np.matmul, [(A, B)])
numpy_results


# In[ ]:


new_matmul = create_locked_down_function(extract_function(prompt))
new_results = benchmarker.benchmark(new_matmul, [(A_list, B_list)])
new_results


# We can take the difference and do a negative sign for slower ones. If the ratio is less than 1 (ie faster, we shall invert it!)

# In[ ]:


negative = -(new_results["median_ns"] / numpy_results["median_ns"]) / 100
positive = +(numpy_results["median_ns"] / new_results["median_ns"]) / 100
reward = negative if new_results["median_ns"] >= numpy_results["median_ns"] else positive
reward


# In[ ]:


new_results["median_ns"] = 3
numpy_results["median_ns"] = 1000
negative = -(new_results["median_ns"] / numpy_results["median_ns"]) / 100
positive = +(numpy_results["median_ns"] / new_results["median_ns"]) / 100
reward = negative if new_results["median_ns"] >= numpy_results["median_ns"] else positive
reward


# In[ ]:


import gc
def speed_check(completions, **kwargs):
    scores = []
    # Generate some random matrices of size less than 256
    A, A_list, B, B_list = generate_random_matrices(seed = np.random.randint(10000), n = 256)
    numpy_results = benchmarker.benchmark(np.matmul, [(A, B)])
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_only_stdlib_imports(function)
        if function is None or "error" in info:
            scores.append(0)
            continue
        try:
            new_matmul = create_locked_down_function(function)
        except:
            scores.append(0)
            continue
        new_results = benchmarker.benchmark(new_matmul, [(A_list.copy(), B_list.copy())])

        # Get score and clip to -10, 10
        negative = -(new_results["median_ns"] / numpy_results["median_ns"]) / 100
        positive = +(numpy_results["median_ns"] / new_results["median_ns"]) / 100
        score = negative if new_results["median_ns"] >= numpy_results["median_ns"] else positive
        if score >= 10:  score = 10
        if score <= -10: score = -10
        scores.append(score)
    # Free memory to counteract OOMs
    gc.collect()
    torch.cuda.empty_cache()
    return scores


# We create the dataset which includes a replica of our prompt.

# In[ ]:


from datasets import Dataset
dataset = Dataset.from_list([{"prompt" : [{"role": "user", "content": prompt.strip()}], "answer" : 0}]*1000)
maximum_length = len(tokenizer.apply_chat_template([{"role":"user", "content":prompt.strip()}], add_generation_prompt = True, tokenize = True))
print(maximum_length)
dataset[0]


# <a name="Train"></a>
# ### Train the model
# 
# Now set up GRPO Trainer and all configurations! We also support GSDP, GAPO, Dr GRPO and more! Go to our docs https://unsloth.ai/docs/ for more info!

# In[ ]:


# Leave room for the prompt (plus 1 token safety margin)
max_completion_length = max_seq_length - (maximum_length + 1)

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    temperature = 1.0,
    top_p = 0.95,
    top_k = 64,
    learning_rate = 5e-5,
    weight_decay = 0.001,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 2, # Increase to 4 for smoother training
    num_generations = 2, # Decrease if out of memory
    max_completion_length = max_completion_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 100,
    save_steps = 100,
    report_to = "none", # Can use Weights & Biases, TrackIO
    output_dir = "outputs",
    epsilon = 0.2,
    epsilon_high = 0.28, # one sided
    delta = 1.5, # two sided
    loss_type = 'bnpo',
    mask_truncated_completions = True
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

# In[ ]:


# For optional training + evaluation
# new_dataset = dataset.train_test_split(test_size = 0.01)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        function_works,
        no_cheating,
        correctness_check,
        speed_check,
    ],
    args = training_args,
    train_dataset = dataset,

    # For optional training + evaluation
    # train_dataset = new_dataset["train"],
    # eval_dataset = new_dataset["test"],
)


# And let's train the model!
# 
# **NOTE** A T4 free GPU might take 5 minutes for one generation sadly since it's an old GPU - A100 or H100 will be much faster!

# In[ ]:


trainer.train()


# And now with the LoRA we just trained with GRPO - we first save the LoRA first!

# In[ ]:


model.save_pretrained("gemma_4_lora")  # Local saving
tokenizer.save_pretrained("gemma_4_lora")


# Verify LoRA is actually trained!

# In[ ]:


from safetensors import safe_open

tensors = {}
with safe_open("grpo_saved_lora/adapter_model.safetensors", framework = "pt") as f:
    # Verify both A and B are non zero
    for key in f.keys():
        tensor = f.get_tensor(key)
        n_zeros = (tensor == 0).sum() / tensor.numel()
        assert(n_zeros.item() != tensor.numel())


# <a name="Inference"></a>
# # Inference
# Now let's try the model we just trained!

# In[ ]:


text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt.strip()}],
    tokenize = False,
    add_generation_prompt = True,
)

from transformers import TextStreamer

_ = model.generate(
    **tokenizer(images = None, text = text, return_tensors = "pt").to("cuda"),
    temperature = 1.0, top_p = 0.95, top_k = 64,
    max_new_tokens = 1024,
    streamer = TextStreamer(tokenizer, skip_prompt = False),
)


# <a name="Save"></a>
# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.

# In[ ]:


# Merge to 16bit
if False: model.save_pretrained_merged("gemma_4_finetune_16bit", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("HF_USERNAME/gemma_4_finetune_16bit", tokenizer, save_method = "merged_16bit", token = "YOUR_HF_TOKEN")

# Merge to 4bit
if False: model.save_pretrained_merged("gemma_4_finetune_4bit", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("HF_USERNAME/gemma_4_finetune_4bit", tokenizer, save_method = "merged_4bit", token = "YOUR_HF_TOKEN")

# Just LoRA adapters
if False:
    model.save_pretrained("gemma_4_lora")
    tokenizer.save_pretrained("gemma_4_lora")
if False:
    model.push_to_hub("HF_USERNAME/gemma_4_lora", token = "YOUR_HF_TOKEN")
    tokenizer.push_to_hub("HF_USERNAME/gemma_4_lora", token = "YOUR_HF_TOKEN")


# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
# 
# Some supported quant methods (full list on our [docs page](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.
# 
# [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)

# In[ ]:


# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("gemma_4_finetune", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False: model.push_to_hub_gguf("HF_USERNAME/gemma_4_finetune", tokenizer, token = "YOUR_HF_TOKEN")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("gemma_4_finetune", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("HF_USERNAME/gemma_4_finetune", tokenizer, quantization_method = "f16", token = "YOUR_HF_TOKEN")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("gemma_4_finetune", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("HF_USERNAME/gemma_4_finetune", tokenizer, quantization_method = "q4_k_m", token = "YOUR_HF_TOKEN")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "HF_USERNAME/gemma_4_finetune", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "YOUR_HF_TOKEN",
    )


# Now, use the `gemma_4_finetune.Q8_0.gguf` file or `gemma_4_finetune.Q4_K_M.gguf` file in llama.cpp.
# 
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
# </div>
# 
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
