# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "bitsandbytes>=0.43.0",
#     "git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels",
#     "marimo",
#     "tokenizers>=0.22.0,<=0.23.0",
#     "torch>=2.8.0",
#     "torchao>=0.16.0",
#     "torchvision",
#     "transformers>=4.56.0",
#     "triton>=3.2.0",
#     "trl==0.22.2",
#     "unsloth[base] @ git+https://github.com/unslothai/unsloth",
#     "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo",
#     "uv",
# ]
#
# [tool.uv]
# no-build-package = [
#     "bitsandbytes",
#     "triton",
#     "vllm",
#     "xformers",
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://github.com/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb" target="_parent"><img src="https://marimo.io/molab-shield.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To run this, press the **Run** button beside each cell!
    <div class="align-center">
    <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
    <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
    <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
    </div>

    To install Unsloth on your local device, follow [our guide](https://unsloth.ai/docs/get-started/install). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).

    You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & how to save it
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### News
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Introducing **Unsloth Studio** - a new open source, no-code web UI to train and run LLMs. [Blog](https://unsloth.ai/docs/new/studio) • [Notebook](https://github.com/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb)

    <table><tr>
    <td align="center"><a href="https://unsloth.ai/docs/new/studio"><img src="https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F~%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FxV1PO5DbF3ksB51nE2Tw%252Fmore%2520cropped%2520ui%2520for%2520homepage.png%3Falt%3Dmedia%26token%3Df75942c9-3d8d-4b59-8ba2-1a4a38de1b86&width=376&dpr=3&quality=100&sign=a663c397&sv=2" width="200" height="120" alt="Unsloth Studio Training UI"></a><br><sub><b>Train models</b> — no code needed</sub></td>
    <td align="center"><a href="https://unsloth.ai/docs/new/studio"><img src="https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F~%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FRCnTAZ6Uh88DIlU3g0Ij%252Fmainpage%2520unsloth.png%3Falt%3Dmedia%26token%3D837c96b6-bd09-4e81-bc76-fa50421e9bfb&width=376&dpr=3&quality=100&sign=c1a39da1&sv=2" width="200" height="120" alt="Unsloth Studio Chat UI"></a><br><sub><b>Run GGUF models</b> on Mac, Windows & Linux</sub></td>
    </tr></table>

    Train MoEs - DeepSeek, GLM, Qwen and gpt-oss 12x faster with 35% less VRAM. [Blog](https://unsloth.ai/docs/new/faster-moe)

    Ultra Long-Context Reinforcement Learning is here with 7x more context windows! [Blog](https://unsloth.ai/docs/new/grpo-long-context)

    New in Reinforcement Learning: [FP8 RL](https://unsloth.ai/docs/new/fp8-reinforcement-learning) • [Vision RL](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl) • [Standby](https://unsloth.ai/docs/basics/memory-efficient-rl) • [gpt-oss RL](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning)

    Visit our docs for all our [model uploads](https://unsloth.ai/docs/get-started/unsloth-model-catalog) and [notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Unsloth
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Goal: Make faster kernels with Reinforcement Learning

    Our goal is to make a faster matrix multiplication kernel by doing RL on GPT-OSS 20B with Unsloth.

    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Matrix_multiplication_qtl1.svg/500px-Matrix_multiplication_qtl1.svg.png" height=200 />

    You will learn how to:
    1. Counteract **reward hacking** like cheating, caching, laziness.
    2. Timing and correctness of kernels and time limits.
    3. Making good **reward functions**
    4. How to seriously do RL to make optimized CUDA kernels
    """)
    return


@app.cell
def _():
    from unsloth import FastLanguageModel
    import torch

    max_seq_length = 768  # Can increase for longer RL output
    lora_rank = 4  # Larger rank = smarter, but slower
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b",
        max_seq_length=max_seq_length,  # Can increase for longer RL output
        load_in_4bit=True,  # False for LoRA 16bit
        offload_embedding=True,  # Reduces VRAM by 1GB
    )
    return (
        FastLanguageModel,
        lora_rank,
        max_seq_length,
        model,
        tokenizer,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We now add some small amount of LoRA weights to GPT-OSS so we only need to train those, instead of training on the full model.
    """)
    return


@app.cell
def _(FastLanguageModel, lora_rank, model):
    model_1 = FastLanguageModel.get_peft_model(
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
    return (model_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimized matrix multiplication

    Numpy has optimized matrix multiplication kernels for CPUs via BLAS optimized operations. For GPUs, one can use CUDA accelerated cuBLAS kernels which PyTorch calls under the hood.

    To generate some random matrices to do matrix multiplication, we can do the below:
    """)
    return


@app.cell
def _():
    import numpy as np

    def generate_random_matrices(seed=3407, n=256):
        random_state = np.random.RandomState(seed)
        n, k, m = random_state.randint(1, n + 1, size=3)
        A = np.random.uniform(-10, 10, size=(n, k))
        B = np.random.uniform(-10, 10, size=(k, m))
        return A, A.tolist(), B, B.tolist()

    return generate_random_matrices, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We shall generate a small matrix, and see the matrix multiplied output
    """)
    return


@app.cell
def _(generate_random_matrices, np):
    A, A_list, B, B_list = generate_random_matrices(seed=42, n=5)
    print(A)
    print(B)
    print(np.matmul(A, B))
    return A, A_list, B, B_list


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can call a LLM to generate a simple matrix multiply kernel in Python only, and we can calculate the differences between the actual result and the kernel's result
    """)
    return


@app.function
def calculate_difference(pred, real):
    if pred is None:
        return 5, 5
    assert real is not None
    import numpy as np

    try:
        difference = pred - real
    except:
        return 5, 5
    amax_error = float(np.amax(difference))
    mse_error = float(np.mean(np.square(difference)))
    return amax_error, mse_error


@app.function
# Kernel generated by GPT-5
def matmul(A, B):
    z, s = zip, sum
    Bt = list(z(*B))
    return [[s(a * b for a, b in z(row, col)) for col in Bt] for row in A]


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see the error below is very small, so that's good!
    """)
    return


@app.cell
def _(A, A_list, B, B_list, np):
    prediction = matmul(A_list, B_list)
    calculate_difference(prediction, np.matmul(A, B))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Countering Reward Hacking

    The ultimate goal of RL is to maximize some reward (say speed, revenue, some metric).

    But RL can **cheat** When the RL algorithm learns a trick or exploits something to increase the reward, without actually doing the task at end, this is called "Reward Hacking".

    Some good examples are in https://en.wikipedia.org/wiki/Reward_hacking

    For matrix multiplication kernels, we might see the following issues:

    * Laziness: RL learns to use Numpy, Torch, other libraries, which calls optimized CUDA kernels.
    * Caching: RL learns to cache the result of the output
    * Cheating: RL learns to find the actual output by inspecting Python global variables
    * RL learns to edit the timing function to make it output 0 time as passed.

    And possibly more. We shall try to address each!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Countering Reward Hacking 1: Stop laziness
    We can stop the RL algorithm from calling optimized code by inspecting if the generated code imports other non standard Python libraries. We used GPT-5 to help generate this check `check_only_stdlib_imports`:
    """)
    return


@app.cell
def _():
    # @title (Collapsible code)
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
        names = names | {m.lower() for m in sys.builtin_module_names}
        names.add("__future__")  # special-case
        try:
            stdlib_dir = Path(
                sysconfig.get_path("stdlib")
            )  # Fallback/augmentation: scan the stdlib directory
            if stdlib_dir.exists():
                for p in stdlib_dir.iterdir():
                    if p.name == "site-packages":
                        continue
                    if p.suffix == ".py":
                        names.add(p.stem.lower())
                    elif p.is_dir() and (p / "__init__.py").exists():
                        names.add(p.name.lower())
        except Exception:
            pass
        return names

    _STDLIB_SET = (
        _stdlib_names()
    )  # conservative fallback; the names set above will still work well

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
            return (
                False,
                {
                    "error": f"SyntaxError: {e}",
                    "stdlib": [],
                    "non_stdlib": [],
                    "relative_imports": 0,
                },
            )
        abs_imports = set()
        relative_count = 0

        class Visitor(ast.NodeVisitor):
            def visit_Import(self, node: ast.Import):
                for alias in node.names:
                    abs_imports.add(alias.name.split(".")[0])

            def visit_ImportFrom(self, node: ast.ImportFrom):
                nonlocal relative_count
                if (node.level or 0) > 0:
                    relative_count = relative_count + 1
                elif node.module:
                    abs_imports.add(node.module.split(".")[0])

        Visitor().visit(tree)
        stdlib_found = sorted((m for m in abs_imports if m.lower() in _STDLIB_SET))
        non_stdlib = sorted((m for m in abs_imports if m.lower() not in _STDLIB_SET))
        return (
            len(non_stdlib) == 0,
            {
                "stdlib": stdlib_found,
                "non_stdlib": non_stdlib,
                "relative_imports": relative_count,
            },
        )  # relative import

    return (check_only_stdlib_imports,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For example, let's call `check_only_stdlib_imports` on a random piece of matrix multiplication code generated by GPT-5:
    """)
    return


@app.cell
def _(check_only_stdlib_imports):
    sample = "\ndef matmul(A, B):\n    import numpy as np\n    from torch import matmul\n    z, s = zip, sum\n    Bt = list(z(*B))\n    return [[s(a*b for a, b in z(row, col)) for col in Bt] for row in A]\n"
    _ok, _info = check_only_stdlib_imports(sample)
    print("Only stdlib imports?", _ok)
    print(_info)
    return (sample,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Countering Reward Hacking 2: Stop cheating
    We can stop the RL algorithm from using global or cached variables by restricting it's `locals` and `globals`.

    We are also going to use `exec` to create the function, so we have to save the output to an empty dict.

    We also disallow global variable access.
    """)
    return


@app.cell
def _(sample):
    output_function = {}
    exec(sample, {}, output_function)
    output_function["matmul"]
    return (output_function,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also disallow global variable access via `types.FunctionType(f.__code__, {})`
    """)
    return


@app.cell
def _(np, output_function):
    import types

    output_function["matmul"] = types.FunctionType(
        output_function["matmul"].__code__, {}
    )

    def import_numpy():
        np.matmul
        print("Success")

    import_numpy()
    import_numpy = types.FunctionType(import_numpy.__code__, {})
    try:
        import_numpy()
    except Exception as e:
        print(str(e))
    return (types,)


@app.cell
def _(types):
    def create_locked_down_function(function):
        output_function = {}
        exec(function, {}, output_function)
        new_matmul = output_function["matmul"]
        new_matmul = types.FunctionType(new_matmul.__code__, {})
        return new_matmul

    return (create_locked_down_function,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Countering Reward Hacking 3: Stop caching
    We can stop the RL algorithm from using cached data by wiping the cache with a large fake matrix. We also have to benchmark carefully with multiple loops and turns.

    We also add a **timer** to not make the algorithm go in an endless loop.
    """)
    return


@app.cell
def _(np):
    import os, gc, time, statistics
    import signal
    from contextlib import contextmanager

    class TimeoutError(Exception):
        pass

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
        def __init__(
            self, trials=3, loops=1, timeout=30
        ):  # Cannot be 0 since it won't work!
            self.buffer = np.zeros(2 * 1024 * 1024 * 1024, dtype=np.uint8)
            self.trials = trials
            self.loops = loops  # Edit the buffer to wipe cache lines
            assert timeout > 0
            self.timeout = timeout

        def thrash(self):
            self.buffer = self.buffer ^ 1
            return int(self.buffer[::4096].sum())

        def benchmark(self, function, arguments):
            assert len(arguments) == self.loops
            samples = []
            exceptions = []
            timed_out = 0
            for _ in range(self.trials):
                gc.collect()
                gc.disable()
                self.thrash()
                t_start = time.perf_counter_ns()
                for i in range(self.loops):
                    try:
                        with time_limit(self.timeout):
                            function(*arguments[i])
                    except TimeoutError as e:
                        timed_out = timed_out + 1
                    except Exception as e:
                        exceptions.append(str(e))
                t_end = time.perf_counter_ns()
                gc.enable()
                samples.append((t_end - t_start) // max(1, self.loops))
            return {
                "median_ns": int(statistics.median(samples)),
                "mean_ns": int(statistics.fmean(samples)),
                "stdev_ns": int(statistics.pstdev(samples) if len(samples) > 1 else 0),
                "exceptions": exceptions,
                "timeouts": timed_out,
            }

    return Benchmarker, gc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For example we use our matmul kernel we had, and benchmark it with a 10 second delay:
    """)
    return


@app.cell
def _(Benchmarker, generate_random_matrices, output_function):
    A_1, A_list_1, B_1, B_list_1 = generate_random_matrices(seed=0, n=256)
    Benchmarker(trials=1, timeout=10).benchmark(
        output_function["matmul"], [(A_list_1, B_list_1)]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data & RL task setup

    We now have to create a prompt to the model for which it will do some task. For our matrix multiply example, we use the below:
    """)
    return


@app.cell
def _():
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
    return (prompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, let's prompt GPT-OSS without RL and see how it goes:
    """)
    return


@app.cell
def _(model_1, prompt, tokenizer):
    _text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )
    from transformers import TextStreamer

    _ = model_1.generate(
        **tokenizer(_text, return_tensors="pt").to("cuda"),
        temperature=1.0,
        max_new_tokens=512,
        streamer=TextStreamer(tokenizer, skip_prompt=False),
    )
    return (TextStreamer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reward functions

    We now design the `extract_function` function which simply extracts the function wrapped in 3 backticks.

    And 4 reward functions:

    1. `function_works` which rewards the model if the strategy is a valid Python function.
    2. `no_cheating` which checks if the function imported other modules, and if it did, we penalize it.
    3. `correctness_check` which checks if the kernel was correct or wrong - it shouldn't generate gibberish!
    4. `speed_check` checks the performance relative to Numpy matmul directly.
    """)
    return


@app.cell
def _(prompt):
    def extract_function(text):
        if _text.count("```") >= 2:
            first = _text.find("```") + 3
            second = _text.find("```", first)
            fx = _text[first:second].strip()
            fx = fx.removeprefix("python\n")
            fx = fx[fx.find("def") :]
            if fx.startswith("def matmul(A, B):"):
                return fx
        return None

    print(extract_function(prompt))
    return (extract_function,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below is our `function_works` reward function which uses Python's `exec` but guarded by not allowing leakage of local and global variables. We can also use `check_only_stdlib_imports` first to check if there are errors before even executing the function:
    """)
    return


@app.cell
def _(check_only_stdlib_imports):
    _ok, _info = check_only_stdlib_imports("def a")
    (_ok, _info)
    return


@app.cell
def _(
    check_only_stdlib_imports,
    create_locked_down_function,
    extract_function,
):
    def function_works(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            function = extract_function(response)
            print(function)
            if function is not None:
                _ok, _info = check_only_stdlib_imports(function)
            if function is None or "error" in _info:
                score = -2.0
            else:
                try:
                    new_matmul = create_locked_down_function(function)
                    score = 1.0
                except:
                    score = -0.5
            scores.append(score)
        return scores

    return (function_works,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `no_cheating` checks if the function cheated since it might have imported Numpy or Torch optimized code.
    """)
    return


@app.cell
def _(check_only_stdlib_imports, extract_function):
    def no_cheating(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            function = extract_function(response)
            if function is not None:
                _ok, _info = check_only_stdlib_imports(function)
            else:
                _ok = False
            scores.append(1.0 if _ok else -20.0)  # Penalize heavily!
        return scores

    return (no_cheating,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next `correctness_check` checks if the kernel was correct. We want to penalize if the absolute error is larger than 1, and if the mean squared error is somewhat bigger then machine epsilon.

    We have to execute the code now!
    """)
    return


@app.cell
def _(np):
    np.finfo(np.float64).eps
    return


@app.cell
def _(
    check_only_stdlib_imports,
    create_locked_down_function,
    extract_function,
    generate_random_matrices,
    np,
):
    def correctness_check(completions, **kwargs):
        scores = []
        A, A_list, B, B_list = generate_random_matrices(
            seed=np.random.randint(10000), n=128
        )
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            function = extract_function(response)
            if function is not None:
                _ok, _info = check_only_stdlib_imports(function)
            if function is None or "error" in _info:
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
                scores.append(-2.0)
                continue
            true = np.matmul(A, B)
            amax_error, mse_error = calculate_difference(pred, true)
            machine_epsilon = 100 * np.finfo(np.float64).eps
            if amax_error >= 3:
                score = -3.0
            elif amax_error >= 2:
                score = -2.5
            elif amax_error >= 1:
                score = -2.0
            elif amax_error >= 0.5:
                score = -1.0
            elif amax_error >= 100 * machine_epsilon:
                score = 0.0
            elif amax_error >= machine_epsilon:
                score = 1.0
            else:
                score = 3.0
            if mse_error >= 3:
                score = score + -3.0
            elif mse_error >= 2:
                score = score + -2.5
            elif mse_error >= 1:
                score = score + -2.0
            elif mse_error >= 0.5:
                score = score + -1.0
            elif mse_error >= 100 * machine_epsilon:
                score = score + 0.0
            elif mse_error >= machine_epsilon:
                score = score + 1.0
            else:
                score = score + 3.0
            scores.append(score)
        return scores

    return (correctness_check,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally our benchmarking function for `speed_check`! We shall limit the timer to 10 seconds and do 3 trials.
    """)
    return


@app.cell
def _(Benchmarker, generate_random_matrices, np):
    A_2, A_list_2, B_2, B_list_2 = generate_random_matrices(seed=0, n=256)
    benchmarker = Benchmarker(trials=3, timeout=10)
    numpy_results = benchmarker.benchmark(np.matmul, [(A_2, B_2)])
    numpy_results
    return A_list_2, B_list_2, benchmarker, numpy_results


@app.cell
def _(
    A_list_2,
    B_list_2,
    benchmarker,
    create_locked_down_function,
    extract_function,
    prompt,
):
    new_matmul = create_locked_down_function(extract_function(prompt))
    new_results = benchmarker.benchmark(new_matmul, [(A_list_2, B_list_2)])
    new_results
    return (new_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can take the difference and do a negative sign for slower ones. If the ratio is less than 1 (ie faster, we shall invert it!)
    """)
    return


@app.cell
def _(new_results, numpy_results):
    _negative = -(new_results["median_ns"] / numpy_results["median_ns"]) / 100
    _positive = +(numpy_results["median_ns"] / new_results["median_ns"]) / 100
    _reward = (
        _negative
        if new_results["median_ns"] >= numpy_results["median_ns"]
        else _positive
    )
    _reward
    return


@app.cell
def _(new_results, numpy_results):
    new_results["median_ns"] = 3
    numpy_results["median_ns"] = 1000
    _negative = -(new_results["median_ns"] / numpy_results["median_ns"]) / 100
    _positive = +(numpy_results["median_ns"] / new_results["median_ns"]) / 100
    _reward = (
        _negative
        if new_results["median_ns"] >= numpy_results["median_ns"]
        else _positive
    )
    _reward
    return


@app.cell
def _(
    benchmarker,
    check_only_stdlib_imports,
    create_locked_down_function,
    extract_function,
    gc,
    generate_random_matrices,
    np,
    torch,
):
    def speed_check(completions, **kwargs):
        scores = []
        A, A_list, B, B_list = generate_random_matrices(
            seed=np.random.randint(10000), n=256
        )
        numpy_results = benchmarker.benchmark(np.matmul, [(A, B)])
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            function = extract_function(response)
            if function is not None:
                _ok, _info = check_only_stdlib_imports(function)
            if function is None or "error" in _info:
                scores.append(0)
                continue
            try:
                new_matmul = create_locked_down_function(function)
            except:
                scores.append(0)
                continue
            new_results = benchmarker.benchmark(
                new_matmul, [(A_list.copy(), B_list.copy())]
            )
            _negative = -(new_results["median_ns"] / numpy_results["median_ns"]) / 100
            _positive = +(numpy_results["median_ns"] / new_results["median_ns"]) / 100
            score = (
                _negative
                if new_results["median_ns"] >= numpy_results["median_ns"]
                else _positive
            )
            if score >= 10:
                score = 10
            if score <= -10:
                score = -10
            scores.append(score)
        gc.collect()
        torch.cuda.empty_cache()
        return scores

    return (speed_check,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We create the dataset which includes a replica of our prompt. Remember to add reasoning effort of low!
    """)
    return


@app.cell
def _(prompt, tokenizer):
    from datasets import Dataset

    dataset = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": prompt.strip()}],
                "answer": 0,
                "reasoning_effort": "low",
            }
        ]
        * 1000
    )
    maximum_length = len(tokenizer(prompt.strip())["input_ids"])
    print(maximum_length)
    dataset[0]
    return dataset, maximum_length


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model

    Now set up GRPO Trainer and all configurations! We also support GSDP, GAPO, Dr GRPO and more! Go to our docs https://unsloth.ai/docs/ for more info!
    """)
    return


@app.cell
def _(max_seq_length, maximum_length):
    max_prompt_length = maximum_length + 1  # + 1 just in case!
    max_completion_length = max_seq_length - max_prompt_length

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        temperature=1.0,
        learning_rate=5e-5,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,  # Increase to 4 for smoother training
        num_generations=2,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,  # + 1 just in case!
        max_completion_length=max_completion_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=100,
        save_steps=100,
        report_to="none",  # Can use Weights & Biases
        output_dir="outputs",
        # For optional training + evaluation
        # fp16_full_eval = True,
        # per_device_eval_batch_size = 4,
        # eval_accumulation_steps = 1,
        # eval_strategy = "steps",
        # eval_steps = 1,
    )
    return GRPOTrainer, training_args


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And let's run the trainer! If you scroll up, you'll see a table of rewards. The goal is to see the `reward` column increase!

    You might have to wait 150 to 200 steps for any action. You'll probably get 0 reward for the first 100 steps. Please be patient!

    | Step | Training Loss | reward    | reward_std | completion_length | kl       |
    |------|---------------|-----------|------------|-------------------|----------|
    | 1    | 0.000000      | 0.125000  | 0.000000   | 200.000000        | 0.000000 |
    | 2    | 0.000000      | 0.072375  | 0.248112   | 200.000000        | 0.000000 |
    | 3    | 0.000000      | -0.079000 | 0.163776   | 182.500000        | 0.000005 |
    """)
    return


@app.cell
def _(
    GRPOTrainer,
    correctness_check,
    dataset,
    function_works,
    model_1,
    no_cheating,
    speed_check,
    tokenizer,
    training_args,
):
    # For optional training + evaluation
    # new_dataset = dataset.train_test_split(test_size = 0.01)
    trainer = GRPOTrainer(
        model=model_1,
        processing_class=tokenizer,
        reward_funcs=[function_works, no_cheating, correctness_check, speed_check],
        args=training_args,
        train_dataset=dataset,
    )
    return (trainer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And let's train the model!

    **NOTE** A T4 free GPU might take 5 minutes for one generation sadly since it's an old GPU - A100 or H100 will be much faster!
    """)
    return


@app.cell
def _(trainer):
    trainer.train()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Inference"></a>
    # Inference
    Now let's try the model we just trained!
    """)
    return


@app.cell
def _(TextStreamer, model_1, prompt, tokenizer):
    _text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )
    _ = model_1.generate(
        **tokenizer(_text, return_tensors="pt").to("cuda"),
        temperature=1.0,
        max_new_tokens=1024,
        streamer=TextStreamer(tokenizer, skip_prompt=False),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### Saving to float16 or MXFP4 for VLLM

    We also support saving to `float16` directly. Select `merged_16bit` for float16 or `mxfp4` for MXFP4 (OpenAI's GPT-OSS native precision). We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    # Merge and push to hub in mxfp4 4bit format
    if False:
        model_1.save_pretrained_merged(
            "gpt_oss_finetune_4bit", tokenizer, save_method="mxfp4"
        )
    if False:
        model_1.push_to_hub_merged(
            "repo_id/gpt_oss_finetune_4bit",
            tokenizer,
            token="YOUR_HF_TOKEN",
            save_method="mxfp4",
        )
    # Merge and push to hub in 16bit
    if False:
        model_1.save_pretrained_merged(
            "gpt_oss_finetune_16bit", tokenizer, save_method="merged_16bit"
        )
    if False:  # Pushing to HF Hub
        model_1.push_to_hub_merged(
            "HF_USERNAME/gpt_oss_finetune_16bit",
            tokenizer,
            save_method="merged_16bit",
            token="YOUR_HF_TOKEN",
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!

    Some other resources:
    1. Train your own reasoning model - Llama GRPO notebook [Open in molab](https://github.com/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
    2. Saving finetunes to Ollama. [Free notebook](https://github.com/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
    3. Llama 3.2 Vision finetuning - Radiography use case. [Open in molab](https://github.com/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
    4. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://unsloth.ai/docs/get-started/unsloth-notebooks)!

    <div class="align-center">
      <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
      <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
      <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>

      Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
    </div>

      This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
    """)
    return


if __name__ == "__main__":
    app.run()
