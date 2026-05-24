# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "bitsandbytes>=0.43.0",
#     "git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels",
#     "marimo",
#     "timm",
#     "tokenizers>=0.22.0,<=0.23.0",
#     "torch>=2.8.0",
#     "torchvision",
#     "transformers>=4.56.0",
#     "triton>=3.2.0",
#     "trl>=0.28.0",
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
    # Goal: Make Gemma 4 play games with Reinforcement Learning

    Our goal is to make Gemma 4 play the 2048 game with reinforcement learning, or a variant of it called [GRPO](https://arxiv.org/abs/2501.12948).

    We want the model to devise a strategy to play 2048, and we will run this strategy until we win or lose. We then reward the model if it created a good strategy (winning the game), and we'll penalize it (negative reward) if the strategy was a bad one.

    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/2048_win.png/500px-2048_win.png" height=300 />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Installation
    We'll be using [Unsloth](https://github.com/unslothai/unsloth) to do RL on Gemma 4. Unsloth saves 70% VRAM usage and makes reinforcement learning 2 to 6x faster!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Unsloth
    """)
    return


@app.cell
def _():
    from unsloth import FastVisionModel
    import torch

    max_seq_length = 4096  # Can increase for longer reasoning traces
    lora_rank = 32  # Larger rank = smarter, but slower

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
    ]  # More models at https://huggingface.co/unsloth

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name="unsloth/gemma-4-E2B-it",
        max_seq_length=max_seq_length,  # Can increase for longer reasoning traces
        load_in_4bit=False,  # False for LoRA 16bit
        fast_inference=False,  # Enable vllm fast inference
    )
    return FastVisionModel, lora_rank, max_seq_length, model, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To do efficient RL, we will use [LoRA](https://arxiv.org/abs/2106.09685), which allows us to only add 1 to 5% of extra weights to the model for finetuning purposes. This allows us to save memory usage by over 60%, and yet it retains good accuracy.
    """)
    return


@app.cell
def _(FastVisionModel, lora_rank, model):
    model_1 = FastVisionModel.get_peft_model(
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
    # 2048 game

    We used GPT-5 to create a variant of the 2048 game. It should output the current game board state, and allow us to advance the game board state with 1 action (up, down, left, right).
    """)
    return


@app.cell
def _():
    # @title (Collapsible) 2048 Game Implementation
    from dataclasses import dataclass, field
    from typing import List, Tuple, Optional
    import random
    import copy

    def _compress_and_merge_row_left(row: List[int]) -> Tuple[List[int], int, bool]:
        n = len(row)
        tiles = [x for x in row if x != 0]
        gained = 0
        i = 0
        merged = []
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                v = tiles[i] * 2
                gained = gained + v
                merged.append(v)
                i = i + 2
            else:
                merged.append(tiles[i])
                i = i + 1
        merged = merged + [0] * (n - len(merged))
        changed = merged != row
        return (merged, gained, changed)

    def _move_left(board: List[List[int]]) -> Tuple[List[List[int]], int, bool]:
        changed_any = False
        total_gain = 0
        new_board = []
        for row in board:
            new_row, gained, changed = _compress_and_merge_row_left(row)
            new_board.append(new_row)
            total_gain = total_gain + gained
            changed_any = changed_any or changed
        return (new_board, total_gain, changed_any)

    def _move_right(board: List[List[int]]) -> Tuple[List[List[int]], int, bool]:
        changed_any = False
        total_gain = 0
        new_board = []
        for row in board:
            rev = list(reversed(row))
            new_rev, gained, changed = _compress_and_merge_row_left(rev)
            new_row = list(reversed(new_rev))
            new_board.append(new_row)
            total_gain = total_gain + gained
            changed_any = changed_any or changed
        return (new_board, total_gain, changed_any)

    def _transpose(board: List[List[int]]) -> List[List[int]]:
        return [list(row) for row in zip(*board)]

    def _move_up(board: List[List[int]]) -> Tuple[List[List[int]], int, bool]:
        t = _transpose(board)  # safety; avoid log2(1)
        moved, gain, changed = _move_left(t)
        return (_transpose(moved), gain, changed)

    def _move_down(board: List[List[int]]) -> Tuple[List[List[int]], int, bool]:
        t = _transpose(board)  # safety; avoid log2(1)
        moved, gain, changed = _move_right(t)
        return (_transpose(moved), gain, changed)

    def _empty_cells(board: List[List[int]]) -> List[Tuple[int, int]]:
        size = len(board)
        return [(r, c) for r in range(size) for c in range(size) if board[r][c] == 0]

    def _can_move(board: List[List[int]]) -> bool:
        if _empty_cells(board):
            return True
        size = len(board)
        for r in range(size):
            for c in range(size - 1):
                if board[r][c] == board[r][c + 1]:
                    return True
        for r in range(size - 1):
            for c in range(size):
                if board[r][c] == board[r + 1][c]:
                    return True
        return False

    @dataclass
    class GameBoard:
        size: int
        seed: Optional[int] = None
        target: int = 2048
        probability_fours: float = 0.1  # originally spawns (4) 10% of the time!
        _rng: random.Random = field(init=False, repr=False)
        _board: List[List[int]] = field(init=False, repr=False)
        _score: int = field(default=0, init=False, repr=False)
        _state: str = field(default="ongoing", init=False, repr=False)

        def __post_init__(self):
            if self.size < 2:
                raise ValueError("Board size must be at least 2.")
            self._rng = random.Random(self.seed)
            self._board = [[0 for _ in range(self.size)] for _ in range(self.size)]
            self._add_random_tile()
            self._add_random_tile()
            self._update_state_after_change()

        class _BoardView:
            def __init__(self, game: "GameBoard"):
                self._game = game

            def __iter__(self):
                return iter(self._game._board)

            def __len__(self):
                return len(self._game._board)

            def __getitem__(self, idx):
                return self._game._board[idx]

            def __repr__(self) -> str:
                return repr(self._game._board)

            __str__ = __repr__

            def do_action(self, key: str) -> None:
                self._game.do_action(key)

            def state(self) -> str:
                return self._game.state()

            def pretty(
                self,
                colors: bool = True,
                border: bool = True,
                dot_for_zero: bool = True,
            ) -> str:
                return self._game._render_pretty(
                    colors=colors, border=border, dot_for_zero=dot_for_zero
                )

        def board(self) -> "_BoardView":
            return GameBoard._BoardView(self)

        def state(self) -> str:
            return self._state

        def score(self) -> int:
            return self._score

        def do_action(self, key: str) -> None:
            if self._state != "ongoing":
                return
            if not isinstance(key, str) or len(key) == 0:
                self._state = "failed"
                return
            k = key.strip().lower()
            if k == "q":
                self._state = "failed"
                return
            move_map = {
                "a": _move_left,
                "d": _move_right,
                "w": _move_up,
                "s": _move_down,
            }
            if k not in move_map:
                self._state = "failed"
                return
            mover = move_map[k]
            new_board, gain, changed = mover(self._board)
            if changed:
                self._board = new_board
                self._score = self._score + gain
                self._add_random_tile()
            self._update_state_after_change()

        def _add_random_tile(self) -> bool:
            empties = _empty_cells(self._board)
            if not empties:
                return False
            r, c = self._rng.choice(empties)
            self._board[r][c] = 4 if self._rng.random() < self.probability_fours else 2
            return True

        def _update_state_after_change(self) -> None:
            if any((self.target in row for row in self._board)):
                self._state = "success"
                return
            if not _can_move(self._board):
                self._state = "failed"
                return
            self._state = "ongoing"

        # A smooth-ish gradient from cool → warm
        def _render_pretty(
            self, colors: bool = True, border: bool = True, dot_for_zero: bool = True
        ) -> str:  # (blue/cyan/green → yellow/orange/red). Tweak or expand as you like.
            """
            Pretty-print the board with colors that scale from 0 up to self.target.  # dim gray
            Uses ANSI 256-color codes (works in most terminals). Set colors = False to disable.
            """
            import math

            b = self._board
            mx = max((max(row) for row in b), default=0)
            cell_w = max(3, len(str(mx)))
            RESET = "\x1b[0m"  # Normalize by exponent relative to target: r in [0,1]
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
            ]  # safety; avoid log2(1)
            ZERO_FG = (  # dim gray
                239  # Guard: if v is not a power of two or is <1, handle gracefully
            )

            def color_code(v: int) -> str:
                if not colors:
                    return ""
                if v == 0:
                    return f"\x1b[38;5;{ZERO_FG}m"
                t = max(2, self.target)  # safety; avoid log2(1)
                try:
                    r = max(0.0, min(1.0, math.log2(v) / math.log2(t)))  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                except ValueError:
                    r = 0.0  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                idx = int(round(r * (len(GRAD) - 1)))
                return f"\x1b[38;5;{GRAD[idx]}m"

            def fmt(v: int) -> str:
                s = "." if v == 0 and dot_for_zero else str(v)
                s = s.rjust(cell_w)
                return color_code(v) + s + (RESET if colors else "")

            def hline(left: str, mid: str, right: str) -> str:
                return left + mid.join(("─" * cell_w for _ in range(self.size))) + right

            rows = []
            if border:
                rows.append(hline("┌", "┬", "┐"))
            for r in range(self.size):
                content = "│".join((fmt(v) for v in b[r]))
                rows.append("│" + content + "│" if border else content)
                if border:
                    rows.append(
                        hline(
                            "└" if r == self.size - 1 else "├",
                            "┴" if r == self.size - 1 else "┼",
                            "┘" if r == self.size - 1 else "┤",
                        )
                    )
            return "\n".join(rows)

    return (GameBoard,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For example let's create a board of size 5 X 5 and set the target to 8 instead of 2048.

    **[NOTE]** 2048 originally spawns a (4) 10% of the time! We can disable this for harder games. See [Wikipedia page](https://en.wikipedia.org/wiki/2048_(video_game)) for more details.
    """)
    return


@app.cell
def _(GameBoard):
    game = GameBoard(size=5, seed=42, target=8, probability_fours=0.10)
    print(game.board().pretty(), game.state())
    return (game,)


@app.cell
def _(game):
    game
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll use WASD for the action space:

    ```
       W
    A  S  D
    ```
    Also `game.state()` will say `success` if we succeeded in getting the target!
    """)
    return


@app.cell
def _(game):
    game.do_action("A")
    print(game.board().pretty(), game.state())
    return


@app.cell
def _(game):
    game.do_action("W")
    print(game.board().pretty(), game.state())
    return


@app.cell
def _(game):
    game.do_action("D")
    print(game.board().pretty(), game.state())
    return


@app.cell
def _(game):
    game.do_action("W")
    print(game.board().pretty(), game.state())
    return


@app.cell
def _(game):
    game.do_action("D")
    print(game.board().pretty(), game.state())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we do some other action that's not part of the action space, we will get an error, and the game will not accept anymore actions.
    """)
    return


@app.cell
def _(GameBoard):
    game_1 = GameBoard(size=3, seed=42, target=8, probability_fours=0.1)
    game_1.do_action("AA")  # Not in WASD
    game_1.do_action("W")  # Doesn't do anything
    game_1.do_action("A")  # Doesn't do anything
    print(game_1.board().pretty(), game_1.state())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RL Environment Setup

    We'll set up a function to accept some strategy that'll emit an action within `WASD` and check the game state.

    We'll also add a timer to only execute the strategy for 2 seconds maximum, otherwise it might never terminate!
    """)
    return


@app.cell
def _(GameBoard):
    from typing import Callable
    from unsloth import execute_with_time_limit

    def _execute_strategy(strategy: Callable, game: GameBoard):
        assert callable(strategy)
        steps = 0
        while game.state() == "ongoing":
            action = strategy(list(game.board()))
            steps = steps + 1
            if type(action) is not str:
                return (steps, "failed")
            game.do_action(action)
        return (steps, game.state())

    @execute_with_time_limit(2)
    def execute_strategy(strategy: Callable, game: GameBoard):
        return _execute_strategy(strategy, game)

    return Callable, execute_strategy, execute_with_time_limit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's make a generic strategy to just hit `W`. We should expect this generic strategy to fail:
    """)
    return


@app.cell
def _(GameBoard, execute_strategy):
    def always_move_left(board):
        return "W"

    game_2 = GameBoard(size=8, seed=42, target=2048, probability_fours=0.1)
    try:
        execute_strategy(always_move_left, game_2)
    except TimeoutError as e:
        print(f"Timed out with error = {str(e)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To allow longer strategies for Gemma 4 Reinforcement Learning, we shall allow a 5 second timer.
    """)
    return


@app.cell
def _(Callable, GameBoard, execute_with_time_limit):
    @execute_with_time_limit(5)
    def execute_strategy_1(strategy: Callable, game: GameBoard):
        return _execute_strategy(strategy, game)

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

    _sample = '\ndef strategy(board):\n    import math\n    from typing import Callable\n    return "W"\n'
    _ok, _info = check_python_modules(_sample)
    print("Only Python imports?", _ok)
    print(_info)
    return (check_python_modules,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the below piece of code, since we import `numpy`, we should not allow the execution:
    """)
    return


@app.cell
def _(check_python_modules):
    _sample = '\ndef strategy(board):\n    from numpy import matmul\n    return "W"\n'
    _ok, _info = check_python_modules(_sample)
    print("Only Python imports?", _ok)
    print(_info)
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

    _function = '\ndef import_numpy():\n    np.matmul\n    print("Success")\n'
    _f = create_locked_down_function(_function)
    try:
        _f()
    except Exception as e:
        print(str(e))
    return (create_locked_down_function,)


@app.cell
def _(create_locked_down_function):
    _function = "\ndef add(a, b):\n    def adder(a):\n        return a + b\n    return adder(b) + b\n"
    _f = create_locked_down_function(_function)
    try:
        print(_f(10, 20))
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
    Output one action for "W", "A", "S", "D" on what is the optimal next step.
    Output your new short function in backticks using the format below:
    ```python
    def strategy(board):
        return "W" # Example
    ```
    All helper functions should be inside def strategy. Only output the short function `strategy`.
    """.strip()
    print(prompt)
    return (prompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, let's prompt Gemma 4 without RL and see how it goes:
    """)
    return


@app.cell
def _(model_1, prompt, tokenizer):
    _text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt.strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )
    from transformers import TextStreamer

    print("=" * 50)
    print("BASE MODEL OUTPUT (before RL training):")
    print("=" * 50)
    inputs = tokenizer(text=_text, add_special_tokens=False, return_tensors="pt").to(
        "cuda"
    )
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    result = model_1.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=512,
        use_cache=True,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
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
        if _text.count("```") >= 2:
            first = _text.find("```") + 3
            second = _text.find("```", first)
            fx = _text[first:second].strip()
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
    _ok, _info = check_python_modules("def a")
    (_ok, _info)
    return


@app.cell
def _(check_python_modules, create_locked_down_function, extract_function):
    def function_works(completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            _function = extract_function(response)
            if _function is not None:
                _ok, _info = check_python_modules(_function)
            if _function is None or "error" in _info:
                score = -2.0
            else:
                try:
                    new_strategy = create_locked_down_function(_function)
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
            _function = extract_function(response)
            if _function is not None:
                _ok, _info = check_python_modules(_function)
                scores.append(1.0 if _ok else -20.0)  # Penalize heavily!
            else:
                scores.append(-1.0)  # Failed creating function
        return scores

    return (no_cheating,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next `strategy_succeeds` checks if the strategy actually allows the game to terminate. Imagine if the strategy simply returned "W" which would fail after a time limit of 10 seconds.

    We also add a global `PRINTER` to print out the strategy and board state.
    """)
    return


@app.cell
def _(
    GameBoard,
    check_python_modules,
    create_locked_down_function,
    execute_strategy_1,
    extract_function,
):
    import numpy as np

    global PRINTER
    PRINTER = 0

    def strategy_succeeds(completions, **kwargs):
        global PRINTER
        scores = []
        seed = np.random.randint(10000)
        for completion in completions:
            printed = False
            score = 0
            response = completion[0]["content"]
            _function = extract_function(response)
            if PRINTER % 5 == 0:
                printed = True
                print(_function)
            PRINTER = PRINTER + 1
            if _function is not None:
                _ok, _info = check_python_modules(_function)
            if _function is None or "error" in _info:
                scores.append(0)
                continue
            try:
                new_strategy = create_locked_down_function(_function)
            except:
                scores.append(0)
                continue
            try:
                game = GameBoard(size=6, seed=seed, target=2048, probability_fours=0.1)
                steps, game_state = execute_strategy_1(new_strategy, game)
                print(f"Steps = {steps} State = {game_state}")
                if printed is False:
                    print(_function)
                print(game.board().pretty())
                if game_state == "success":
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
    We'll now create the dataset which includes a replica of our prompt.
    """)
    return


@app.cell
def _(prompt, tokenizer):
    from datasets import Dataset

    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": prompt.strip()}], "answer": 0}] * 1000
    )
    maximum_length = len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.strip()}],
            add_generation_prompt=True,
            tokenize=True,
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
    """)
    return


@app.cell
def _(max_seq_length, maximum_length):
    # Leave room for the prompt (plus 1 token safety margin)
    max_completion_length = max_seq_length - (maximum_length + 1)

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        learning_rate=5e-5,
        weight_decay=0.001,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,  # Increase to 4 for smoother training
        num_generations=2,  # Decrease if out of memory
        max_completion_length=max_completion_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=60,
        save_steps=100,
        report_to="none",  # Can use Weights & Biases, TrackIO
        output_dir="outputs",
        epsilon=0.2,
        epsilon_high=0.28,  # one sided
        delta=1.5,  # two sided
        loss_type="bnpo",
        mask_truncated_completions=True,
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
    And now with the LoRA we just trained with GRPO - we first save the LoRA first!
    """)
    return


@app.cell
def _(model_1, tokenizer):
    model_1.save_pretrained("gemma_4_lora")  # Local saving
    tokenizer.save_pretrained("gemma_4_lora")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Verify LoRA is actually trained!
    """)
    return


@app.cell
def _():
    from safetensors import safe_open

    tensors = {}
    with safe_open("grpo_saved_lora/adapter_model.safetensors", framework="pt") as _f:
        for key in _f.keys():
            tensor = _f.get_tensor(key)  # Verify both A and B are non zero
            n_zeros = (tensor == 0).sum() / tensor.numel()
            assert n_zeros.item() != tensor.numel()
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
        [{"role": "user", "content": prompt.strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )
    _ = model_1.generate(
        **tokenizer(images=None, text=_text, return_tensors="pt").to("cuda"),
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        max_new_tokens=1024,
        streamer=TextStreamer(tokenizer, skip_prompt=False),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### Saving to float16 for VLLM

    We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    # Merge to 16bit
    if False:
        model_1.save_pretrained_merged(
            "gemma_4_finetune_16bit", tokenizer, save_method="merged_16bit"
        )
    if False:
        # Merge to 4bit
        model_1.push_to_hub_merged(
            "HF_USERNAME/gemma_4_finetune_16bit",
            tokenizer,
            save_method="merged_16bit",
            token="YOUR_HF_TOKEN",
        )
    if False:
        model_1.save_pretrained_merged(
            "gemma_4_finetune_4bit", tokenizer, save_method="merged_4bit"
        )
    # Just LoRA adapters
    if False:
        model_1.push_to_hub_merged(
            "HF_USERNAME/gemma_4_finetune_4bit",
            tokenizer,
            save_method="merged_4bit",
            token="YOUR_HF_TOKEN",
        )
    if False:
        model_1.save_pretrained("gemma_4_lora")
        tokenizer.save_pretrained("gemma_4_lora")
    if False:
        model_1.push_to_hub("HF_USERNAME/gemma_4_lora", token="YOUR_HF_TOKEN")
        tokenizer.push_to_hub("HF_USERNAME/gemma_4_lora", token="YOUR_HF_TOKEN")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GGUF / llama.cpp Conversion
    To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.

    Some supported quant methods (full list on our [docs page](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)):
    * `q8_0` - Fast conversion. High resource use, but generally acceptable.
    * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
    * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.

    [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://github.com/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
    """)
    return


@app.cell
def _(model_1, tokenizer):
    # Save to 8bit Q8_0
    if False:
        # Remember to go to https://huggingface.co/settings/tokens for a token!
        # And change hf to your username!
        model_1.save_pretrained_gguf("gemma_4_finetune", tokenizer)
    if False:
        # Save to 16bit GGUF
        model_1.push_to_hub_gguf(
            "HF_USERNAME/gemma_4_finetune", tokenizer, token="YOUR_HF_TOKEN"
        )
    if False:
        model_1.save_pretrained_gguf(
            "gemma_4_finetune", tokenizer, quantization_method="f16"
        )
    # Save to q4_k_m GGUF
    if False:
        model_1.push_to_hub_gguf(
            "HF_USERNAME/gemma_4_finetune",
            tokenizer,
            quantization_method="f16",
            token="YOUR_HF_TOKEN",
        )
    if False:
        # Save to multiple GGUF options - much faster if you want multiple!
        model_1.save_pretrained_gguf(
            "gemma_4_finetune", tokenizer, quantization_method="q4_k_m"
        )
    if False:
        model_1.push_to_hub_gguf(
            "HF_USERNAME/gemma_4_finetune",
            tokenizer,
            quantization_method="q4_k_m",
            token="YOUR_HF_TOKEN",
        )  # Change hf to your username!
    if False:
        model_1.push_to_hub_gguf(
            "HF_USERNAME/gemma_4_finetune",
            tokenizer,
            quantization_method=["q4_k_m", "q8_0", "q5_k_m"],
            token="YOUR_HF_TOKEN",
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, use the `gemma_4_finetune.Q8_0.gguf` file or `gemma_4_finetune.Q4_K_M.gguf` file in llama.cpp.

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
