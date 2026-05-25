# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "bitsandbytes>=0.43.0",
#     "fastapi",
#     "marimo",
#     "open_spiel",
#     "requests",
#     "tokenizers>=0.22.0,<=0.23.0",
#     "torch>=2.8.0",
#     "torchao>=0.16.0",
#     "torchvision",
#     "trackio",
#     "transformers==4.56.2",
#     "triton>=3.2.0",
#     "triton_kernels @ git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels",
#     "trl==0.22.2",
#     "unsloth @ git+https://github.com/unslothai/unsloth",
#     "uv",
#     "uvicorn",
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
    <a href="https://github.com/unslothai/notebooks/blob/main/nb/OpenEnv_gpt_oss_(20B)_Reinforcement_Learning_2048_Game_BF16.ipynb" target="_parent"><img src="https://marimo.io/molab-shield.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # <img width="35" height="35" alt="image" src="https://github.com/user-attachments/assets/2700a971-e5d6-4036-b03f-2f89c9791609" /> OpenEnv: Agentic Execution Environments
    We're using the new [OpenEnv](https://github.com/meta-pytorch/OpenEnv) library which has over 2000+ environments for RL!

    <div class="align-center">
    <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
    <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
    <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
    </div>

    To install Unsloth on your local device, follow [our guide](https://unsloth.ai/docs/get-started/install).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Goal: Make gpt-oss play games with Reinforcement Learning

    Our goal is to make OpenAI's open model gpt-oss 20b play the 2048 game with reinforcement learning. We want the model to devise a strategy to play 2048, and we will run this strategy until we win or lose.

    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/2048_win.png/500px-2048_win.png" height=300 />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Installation
    We'll be using [Unsloth](https://github.com/unslothai/unsloth) to do RL on GPT-OSS 20B, and [OpenEnv](https://github.com/meta-pytorch/OpenEnv) for the environment interactions. Unsloth saves 70% VRAM usage and makes reinforcement learning 2 to 6x faster!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will then install [OpenEnv](https://github.com/meta-pytorch/OpenEnv) from source:
    """)
    return


@app.cell
def _():
    import subprocess

    # packages added via marimo's package management: fastapi uvicorn requests open_spiel !pip install -qqq fastapi uvicorn requests open_spiel
    # packages added via marimo's package management: fastapi uvicorn requests !pip install fastapi uvicorn requests
    # packages added via marimo's package management: open_spiel !pip install open_spiel --prefer-binary
    #! git clone https://github.com/meta-pytorch/OpenEnv.git > /dev/null 2>&1
    subprocess.call(
        "git clone https://github.com/meta-pytorch/OpenEnv.git > /dev/null 2>&1",
        shell=True,
    )
    import os

    os.chdir("OpenEnv")
    #! git checkout 83dda10
    subprocess.call(["git", "checkout", "83dda10"])
    import subprocess, sys, os
    from pathlib import Path

    sys.path.insert(0, ".")  # Add OpenEnv root for envs module
    sys.path.insert(0, "./src")
    working_directory = str(Path.cwd().parent.absolute() / "OpenEnv")
    return os, working_directory


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll load GPT-OSS 20B and set some parameters:
    * `max_seq_length = 768` The maximum context length of the model. Increasing it will use more memory.
    * `lora_rank = 4` The larger this number, the smarter the RL process, but the slower and more memory usage`load_in_16bit` will be faster but will need a 64GB GPU or more (MI300)
    * `offload_embedding = True` New Unsloth optimization which moves the embedding to CPU RAM, reducing VRAM by 1GB.
    """)
    return


@app.cell
def _():
    from unsloth import FastLanguageModel
    import torch

    max_seq_length = 768  # Can increase for longer RL output
    lora_rank = 4  # Larger rank = smarter, but slower
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b-BF16",
        load_in_4bit=False,
        max_seq_length=max_seq_length,  # Can increase for longer RL output
    )
    return FastLanguageModel, lora_rank, max_seq_length, model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To do efficient RL, we will use [LoRA](https://arxiv.org/abs/2106.09685), which allows us to only add 1 to 5% of extra weights to the model for finetuning purposes. This allows us to save memory usage by over 60%, and yet it retains good accuracy. Read Unsloth's [GPT-OSS RL Guide](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning) for more details.
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
    # 2048 game environment with OpenEnv

    We first launch an OpenEnv process and import it! This will allows us to see how the 2048 implementation looks like!
    """)
    return


@app.cell
def _():
    from envs.openspiel_env import OpenSpielEnv
    from envs.openspiel_env.models import OpenSpielAction, OpenSpielObservation

    return OpenSpielAction, OpenSpielEnv, OpenSpielObservation


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll be using Unsloth's OpenEnv implementation and wrapping the `launch_openenv` with some setup arguments:
    """)
    return


@app.cell
def _(OpenSpielEnv, os, working_directory):
    global port
    global openenv_process
    port = 9000
    openenv_process = None
    server = "envs.openspiel_env.server.app:app"
    environment = {
        **os.environ,
        "PYTHONPATH": f"{working_directory}/src",
        "OPENSPIEL_GAME": "2048",
        "OPENSPIEL_AGENT_PLAYER": "0",
        "OPENSPIEL_OPPONENT_POLICY": "random",
    }

    # Augment Unsloth's OpenEnv creation function
    import functools
    from unsloth import is_port_open, launch_openenv

    launch_openenv = functools.partial(
        launch_openenv,
        working_directory=working_directory,
        server=server,
        environment=environment,
        openenv_class=OpenSpielEnv,
    )
    return launch_openenv, openenv_process, port


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see how the current 2048 game state looks like:
    """)
    return


@app.cell
def _(launch_openenv, openenv_process, port):
    port_1, openenv_process_1 = launch_openenv(port, openenv_process)
    result = openenv_process_1.reset()
    current_state = result.observation
    current_state
    return current_state, openenv_process_1, port_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First let's convert the state into a list of list of numbers!
    """)
    return


@app.cell
def _(current_state):
    import numpy as np

    def convert_to_board(current_state):
        n = len(current_state.info_state)
        size = int(np.sqrt(n))
        board = np.array_split(np.array(current_state.info_state, dtype=int), size)
        board = [x.tolist() for x in board]
        return board, size

    convert_to_board(current_state)
    return (convert_to_board,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also want to pretty print the game board!
    """)
    return


@app.cell
def _(convert_to_board):
    # @title (Collapsible) 2048 Game Renderer
    def render_board(
        obs, colors: bool = True, border: bool = True, dot_for_zero: bool = True
    ) -> str:
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
        GRAD = [
            33,
            39,
            45,
            51,
            50,
            49,
            48,
            47,
            46,
            82,
            118,
            154,
            190,
            226,
            220,
            214,
            208,
            202,
            196,
        ]
        ZERO_FG = 239  # dim gray

        def color_code(v: int) -> str:
            if not colors:
                return ""
            if v == 0:
                return f"\x1b[38;5;{ZERO_FG}m"
            # Normalize by exponent relative to target: r in [0,1]
            t = max(2, 2048)
            try:
                r = max(0.0, min(1.0, math.log2(v) / math.log2(t)))  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            except ValueError:
                r = 0.0  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
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
                    hline(
                        "└" if r == size - 1 else "├",
                        "┴" if r == size - 1 else "┼",
                        "┘" if r == size - 1 else "┤",
                    )
                )
        return "\n".join(rows)

    return (render_board,)


@app.cell
def _(current_state, render_board):
    print(render_board(current_state))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can see the `legal_actions` ie what you can take as `[0, 1, 2, 3]` Let's try doing the action `0`.
    """)
    return


@app.cell
def _(OpenSpielAction, openenv_process_1, render_board):
    action = OpenSpielAction(action_id=0, game_name="2048")
    result_1 = openenv_process_1.step(action)
    current_state_1 = result_1.observation
    print(render_board(current_state_1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So it looks like `0` is a move up action! Let's try `1`.
    """)
    return


@app.cell
def _(OpenSpielAction, openenv_process_1, render_board):
    action_1 = OpenSpielAction(action_id=1, game_name="2048")
    result_2 = openenv_process_1.step(action_1)
    current_state_2 = result_2.observation
    print(render_board(current_state_2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `1` is a move right action. And `2`:
    """)
    return


@app.cell
def _(OpenSpielAction, openenv_process_1, render_board):
    action_2 = OpenSpielAction(action_id=2, game_name="2048")
    result_3 = openenv_process_1.step(action_2)
    current_state_3 = result_3.observation
    print(render_board(current_state_3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `2` is a move down. And I guess `3` is just move left!
    """)
    return


@app.cell
def _(OpenSpielAction, openenv_process_1, render_board):
    action_3 = OpenSpielAction(action_id=3, game_name="2048")
    result_4 = openenv_process_1.step(action_3)
    current_state_4 = result_4.observation
    print(render_board(current_state_4))
    return (current_state_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also print the game status which indicates if no more moves are possible, and also the possible actions you can take!
    """)
    return


@app.cell
def _(current_state_4):
    print(current_state_4.done)
    print(current_state_4.legal_actions)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RL Environment Setup

    We'll set up a function to accept some strategy that'll emit an action within `0123` and check the game state.

    We'll also add a timer to only execute the strategy for 2 seconds maximum, otherwise it might never terminate!
    """)
    return


@app.cell
def _(
    OpenSpielAction,
    OpenSpielObservation,
    convert_to_board,
    launch_openenv,
    openenv_process,
    openenv_process_1,
    port,
    port_1,
):
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
                return (steps, False)
            steps = steps + 1
            if type(action) is not int or action not in current_state.legal_actions:
                return (steps, max(itertools.chain.from_iterable(board)) == 2048)
            global port, openenv_process
            globals()["port"], globals()["openenv_process"] = launch_openenv(
                port_1, openenv_process_1
            )
            action = OpenSpielAction(action_id=action, game_name="2048")
            result = openenv_process_1.step(action)
            current_state = result.observation
            if result.reward is not None:
                total_reward = total_reward + result.reward
        return (steps, max(itertools.chain.from_iterable(board)) == 2048)

    @execute_with_time_limit(2)
    def execute_strategy(strategy: Callable, current_state: OpenSpielObservation):
        return _execute_strategy(strategy, current_state)

    return Callable, execute_strategy, execute_with_time_limit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's make a generic strategy to just hit `3`. We should expect this generic strategy to fail:
    """)
    return


@app.cell
def _(execute_strategy, launch_openenv, openenv_process_1, port_1):
    def always_move_left(board):
        return 3

    port_2, openenv_process_2 = launch_openenv(port_1, openenv_process_1)
    # Reset OpenEnv to an initial state!
    result_5 = openenv_process_2.reset()
    current_state_5 = result_5.observation
    try:
        steps, if_done = execute_strategy(always_move_left, current_state_5)
    except TimeoutError as e:
        print(f"Timed out with error = {str(e)}")
    (steps, if_done)
    return openenv_process_2, port_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To allow longer strategies for GPT-OSS Reinforcement Learning, we shall allow a 5 second timer.
    """)
    return


@app.cell
def _(Callable, OpenSpielObservation, execute_with_time_limit):
    @execute_with_time_limit(5)
    def execute_strategy_1(strategy: Callable, current_state: OpenSpielObservation):
        return _execute_strategy(strategy, current_state)

    return (execute_strategy_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Code Execution

    To execute and create a new Python function, we first have to check if the function does not call other global variables or cheat. This is called `countering reward hacking` since we don't want the function to cheat.

    For example the below piece of code is fine, since it only imports Python level functions. We use `check_python_modules`:
    """)
    return


@app.cell
def _():
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
    return (check_python_modules,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the below piece of code, since we import `numpy`, we should not allow the execution:
    """)
    return


@app.cell
def _(check_python_modules):
    sample_1 = '\ndef strategy(board):\n    from numpy import matmul\n    return "0"\n'
    ok_1, info_1 = check_python_modules(sample_1)
    print("Only Python imports?", ok_1)
    print(info_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We also disallow global variable access. We'll use Unsloth's `create_locked_down_function` function
    """)
    return


@app.cell
def _():
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
    return (create_locked_down_function,)


@app.cell
def _(create_locked_down_function):
    function_1 = "\ndef add(a, b):\n    def adder(a):\n        return a + b\n    return adder(b) + b\n"
    f_1 = create_locked_down_function(function_1)
    try:
        print(f_1(10, 20))
    except Exception as e:
        print(str(e))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data & RL task setup

    We now have to create a prompt to tell the model to create a strategy for the 2048 game. You can customize this to some other task for another RL task.
    """)
    return


@app.cell
def _():
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
    return (prompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, let's prompt GPT-OSS without RL and see how it goes:
    """)
    return


@app.cell
def _(model_1, prompt, tokenizer):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )
    from transformers import TextStreamer

    _ = model_1.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        temperature=1.0,
        max_new_tokens=512,
        streamer=TextStreamer(tokenizer, skip_prompt=False),
    )
    return (TextStreamer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reward functions

    We now design a `extract_function` function which simply extracts the function wrapped in 3 back ticks.

    And 3 reward functions:

    1. `function_works` which rewards the model if the strategy is a valid Python function.
    2. `no_cheating` which checks if the function imported other modules, and if it did, we penalize it.
    3. `strategy_succeeds` which checks if the game strategy actually succeeds in attaining 2048 after running the auto-generated strategy.
    """)
    return


@app.cell
def _(prompt):
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
    return (extract_function,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below is our `function_works` reward function which uses Python's `exec` but guarded by not allowing leakage of local and global variables. We can also use `check_python_modules` first to check if there are errors before even executing the function:
    """)
    return


@app.cell
def _(check_python_modules):
    ok_2, info_2 = check_python_modules("def a")
    (ok_2, info_2)
    return


@app.cell
def _(check_python_modules, create_locked_down_function, extract_function):
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

    return (function_works,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `no_cheating` checks if the function cheated since it might have imported Numpy or other functions:
    """)
    return


@app.cell
def _(check_python_modules, extract_function):
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

    return (no_cheating,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next `strategy_succeeds` checks if the strategy actually allows the game to terminate. Imagine if the strategy simply returned "0" which would fail after a time limit of 10 seconds.

    We also add a global `PRINTER` to print out the strategy and board state.
    """)
    return


@app.cell
def _(
    check_python_modules,
    create_locked_down_function,
    execute_strategy_1,
    extract_function,
    launch_openenv,
    openenv_process,
    openenv_process_2,
    port,
    port_2,
    render_board,
):
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
            PRINTER = PRINTER + 1
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
                global port, openenv_process
                globals()["port"], globals()["openenv_process"] = launch_openenv(
                    port_2, openenv_process_2
                )
                result = openenv_process_2.reset()
                current_state = result.observation
                steps, if_done = execute_strategy_1(new_strategy, current_state)
                print(f"Steps = {steps} If Done = {if_done}")
                if printed is False:
                    print(function)
                print(render_board(current_state))
                if if_done:
                    scores.append(20.0)
                else:
                    scores.append(2.0)
            except TimeoutError as e:
                print("Timeout")
                scores.append(-1.0)
            except Exception as e:
                print(f"Exception = {str(e)}")
                scores.append(-3.0)
        return scores

    return PRINTER, strategy_succeeds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll now create the dataset which includes a replica of our prompt. Remember to add a reasoning effort of low! You can choose high reasoning mode, but this'll only work on more memory GPUs like MI300s.
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
    maximum_length = len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.strip()}], add_generation_prompt=True
        )
    )
    print(maximum_length)
    dataset[0]
    return dataset, maximum_length


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Train"></a>
    ### Train the model

    Now set up GRPO Trainer and all configurations! We also support GSPO, GAPO, Dr GRPO and more! Go the Unsloth [Reinforcement Learning Docs](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) for more options.

    We're also using [TrackIO](https://github.com/gradio-app/trackio) which allows you to visualize all training metrics straight inside the notebook fully locally!
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
        max_steps=600,
        save_steps=100,
        report_to="none",  # Can use Weights & Biases, TrackIO
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
    dataset,
    function_works,
    model_1,
    no_cheating,
    strategy_succeeds,
    tokenizer,
    training_args,
):
    # For optional training + evaluation
    # new_dataset = dataset.train_test_split(test_size = 0.01)
    trainer = GRPOTrainer(
        model=model_1,
        processing_class=tokenizer,
        reward_funcs=[function_works, no_cheating, strategy_succeeds],
        args=training_args,
        train_dataset=dataset,
    )
    return (trainer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And let's train the model! **NOTE** This might be quite slow! 600 steps takes ~5 hours or longer.

    [TrackIO](https://github.com/gradio-app/trackio) might be a bit slow to load - wait 2 minutes until the graphs pop up!
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
    text_1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )
    _ = model_1.generate(
        **tokenizer(text_1, return_tensors="pt").to("cuda"),
        temperature=1.0,
        max_new_tokens=1024,
        streamer=TextStreamer(tokenizer, skip_prompt=False),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### Saving to float16 or `MXFP4`

    We also support saving to `float16` directly. Select `merged_16bit` for float16 or `mxfp4` for MXFP4 (OpenAI's GPT-OSS native precision). We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    # Merge and push to hub in mxfp4 4bit format
    if False:
        model_1.save_pretrained_merged(
            "finetuned_model", tokenizer, save_method="mxfp4"
        )
    if False:
        model_1.push_to_hub_merged(
            "repo_id/repo_name", tokenizer, token="hf...", save_method="mxfp4"
        )
    if False:
        # Merge and push to hub in 16bit
        model_1.save_pretrained_merged(
            "finetuned_model", tokenizer, save_method="merged_16bit"
        )
    if False:
        model_1.push_to_hub_merged(
            "hf/gpt-oss-finetune", tokenizer, save_method="merged_16bit", token=""
        )  # Pushing to HF Hub
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # And we're done!
    Congratulations you just learned how to do reinforcement learning with GPT-OSS! There were some advanced topics explained in this notebook - to learn more about GPT-OSS and RL, there are more docs in Unsloth's [Reinforcement Learning Guide with GPT-OSS](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning)

    This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
    """)
    return


if __name__ == "__main__":
    app.run()
