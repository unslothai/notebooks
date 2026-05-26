# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "accelerate",
#     "bitsandbytes>=0.43.0",
#     "cut_cross_entropy",
#     "datasets==4.3.0",
#     "hf_transfer",
#     "huggingface_hub>=0.34.0",
#     "marimo",
#     "peft",
#     "protobuf",
#     "sentencepiece",
#     "torchao>=0.16.0",
#     "transformers @ git+https://github.com/huggingface/transformers.git@bf3f0ae70d0e902efab4b8517fce88f6697636ce",
#     "triton>=3.2.0",
#     "trl==0.22.2",
#     "unsloth @ git+https://github.com/unslothai/unsloth",
#     "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo",
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
    <a href="https://github.com/unslothai/notebooks/blob/main/nb/Ministral_3_(3B)_Reinforcement_Learning_Sudoku_Game.ipynb" target="_parent"><img src="https://marimo.io/molab-shield.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To run this notebook, hit the **▶ Run all** button in the bottom-right corner - or use `Ctrl/Cmd + Shift + R`.
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
    # Goal: Make Ministral solve Sudoku puzzles with Reinforcement Learning

    Our goal is to make Ministral learn to solve Sudoku puzzles using reinforcement learning (GRPO).
    The model will devise a strategy to fill in empty cells, and we'll reward it for correct placements
    and completing valid puzzles.

    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Sudoku_Puzzle_by_L2G-20050714_solution_standardized_layout.svg/1280px-Sudoku_Puzzle_by_L2G-20050714_solution_standardized_layout.svg.png" height="300" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Installation
    We'll be using [Unsloth](https://github.com/unslothai/unsloth) to do RL on Ministral. Unsloth saves 70% VRAM usage and makes reinforcement learning 2 to 6x faster.
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

    ministral_models = [
        "unsloth/Ministral-3-3B-Instruct-2512",  # Ministral instruct models
        "unsloth/Ministral-3-8B-Instruct-2512",
        "unsloth/Ministral-3-14B-Instruct-2512",
        "unsloth/Ministral-3-3B-Reasoning-2512",  # Ministral reasoning models
        "unsloth/Ministral-3-8B-Reasoning-2512",
        "unsloth/Ministral-3-14B-Reasoning-2512",
        "unsloth/Ministral-3-3B-Base-2512",  # Ministral base models
        "unsloth/Ministral-3-8B-Base-2512",
        "unsloth/Ministral-3-14B-Base-2512",
    ]  # More models at https://huggingface.co/unsloth

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name="unsloth/Ministral-3-3B-Instruct-2512",
        max_seq_length=max_seq_length,  # Can increase for longer reasoning traces
        load_in_4bit=False,  # False for LoRA 16bit
        fast_inference=False,  # Enable vLLM fast inference
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
    # Sudoku Game Implementation

    We use GPT-5 to create a clean Sudoku solver environment. The strategy outputs "row,col,value" to fill cells.
    """)
    return


@app.cell
def _():
    # @title Sudoku Game Implementation
    from dataclasses import dataclass, field
    from typing import List, Tuple, Optional
    import random
    import copy

    def _is_valid_placement(
        board: List[List[int]], row: int, col: int, num: int
    ) -> bool:
        """Check if placing num at (row, col) is valid."""
        if num in board[row]:  # Check row
            return False
        if num in [board[r][col] for r in range(9)]:
            return False
        box_row, box_col = (3 * (row // 3), 3 * (col // 3))  # Check column
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if board[r][c] == num:
                    return False  # Check 3x3 box
        return True

    def _solve_sudoku(board: List[List[int]]) -> bool:
        """Solve sudoku using backtracking (for puzzle generation)."""
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    for num in range(1, 10):
                        if _is_valid_placement(board, row, col, num):
                            board[row][col] = num
                            if _solve_sudoku(board):
                                return True
                            board[row][col] = 0
                    return False
        return True

    def _generate_complete_board(rng: random.Random) -> List[List[int]]:
        """Generate a complete valid Sudoku board."""
        board = [[0 for _ in range(9)] for _ in range(9)]
        for box in range(3):
            nums = list(range(1, 10))
            rng.shuffle(nums)
            for i in range(3):
                for j in range(3):
                    board[box * 3 + i][box * 3 + j] = nums[i * 3 + j]
        _solve_sudoku(board)
        return board  # Fill diagonal 3x3 boxes first (they don't affect each other)

    @dataclass
    class SudokuGame:
        difficulty: int = 40
        seed: Optional[int] = None
        _rng: random.Random = field(init=False, repr=False)
        _board: List[List[int]] = field(init=False, repr=False)
        _solution: List[List[int]] = field(init=False, repr=False)  # Solve the rest
        _initial_board: List[List[int]] = field(init=False, repr=False)
        _moves: int = field(default=0, init=False, repr=False)
        _state: str = field(default="ongoing", init=False, repr=False)

        def __post_init__(self):
            self._rng = random.Random(
                self.seed
            )  # Number of cells to remove (20=easy, 40=medium, 50=hard)
            complete_board = _generate_complete_board(self._rng)
            self._solution = copy.deepcopy(complete_board)
            self._board = copy.deepcopy(complete_board)
            cells = [(r, c) for r in range(9) for c in range(9)]
            self._rng.shuffle(cells)
            for r, c in cells[: self.difficulty]:
                self._board[r][c] = 0
            self._initial_board = copy.deepcopy(self._board)
            self._update_state()

        def board(self) -> List[List[int]]:
            """Return current board state."""  # Generate complete board
            return [row[:] for row in self._board]

        def initial_board(self) -> List[List[int]]:
            """Return initial puzzle state."""  # Remove cells to create puzzle
            return [row[:] for row in self._initial_board]

        def state(self) -> str:
            """Return game state: 'ongoing', 'success', or 'failed'."""
            return self._state

        def moves(self) -> int:
            """Return number of moves made."""
            return self._moves

        def place_number(self, row: int, col: int, num: int) -> bool:
            """Place a number on the board. Returns True if valid move."""
            if not (0 <= row < 9 and 0 <= col < 9):
                self._state = "failed"
                return False
            if not 1 <= num <= 9:
                self._state = "failed"
                return False
            if self._initial_board[row][col] != 0:
                self._state = "failed"
                return False
            if self._board[row][col] != 0:
                self._state = "failed"
                return False
            if not _is_valid_placement(self._board, row, col, num):
                self._state = "failed"
                return False
            self._board[row][col] = num
            self._moves = self._moves + 1  # Validate input
            self._update_state()
            return True

        def _update_state(self) -> None:
            """Update game state based on current board."""
            if all((self._board[r][c] != 0 for r in range(9) for c in range(9))):
                if self._board == self._solution:
                    self._state = "success"
                else:  # Can't modify initial cells
                    self._state = "failed"
            else:
                self._state = "ongoing"

        def pretty(self, colors: bool = True) -> str:
            """Pretty print the Sudoku board."""
            RESET = "\x1b[0m"  # Check if placement is valid
            INITIAL = "\x1b[38;5;45m"  # Cyan for initial numbers
            PLACED = "\x1b[38;5;226m"  # Yellow for placed numbers
            EMPTY = "\x1b[38;5;239m"  # Gray for empty cells
            lines = []
            lines.append("┌───────┬───────┬───────┐")  # Place number
            for row in range(9):
                row_str = "│ "
                for col in range(9):
                    num = self._board[row][col]
                    if colors:
                        if num == 0:
                            row_str = row_str + f"{EMPTY}.{RESET}"
                        elif (
                            self._initial_board[row][col] != 0
                        ):  # Check if puzzle is complete
                            row_str = row_str + f"{INITIAL}{num}{RESET}"
                        else:  # Verify solution is correct
                            row_str = row_str + f"{PLACED}{num}{RESET}"
                    else:
                        row_str = row_str + (str(num) if num != 0 else ".")
                    if col % 3 == 2:
                        row_str = row_str + " │ "
                    else:
                        row_str = row_str + " "
                lines.append(row_str.rstrip())
                if row == 8:
                    lines.append("└───────┴───────┴───────┘")
                elif row % 3 == 2:  # Cyan for initial numbers
                    lines.append(
                        "├───────┼───────┼───────┤"
                    )  # Yellow for placed numbers
            return "\n".join(lines)  # Gray for empty cells

    return (SudokuGame,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Test the Sudoku environment:
    """)
    return


@app.cell
def _(SudokuGame):
    # Create an easy puzzle
    game = SudokuGame(difficulty=30, seed=42)
    print("Initial puzzle:")
    print(game.pretty())
    print(f"\nState: {game.state()}, Moves: {game.moves()}")
    return (game,)


@app.cell
def _(game):
    game
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try making some moves:
    """)
    return


@app.cell
def _(game):
    # Make a valid move
    game.place_number(0, 1, 7)
    print("\nAfter placing 7 at (1,0):")
    print(game.pretty())
    print(f"State: {game.state()}, Moves: {game.moves()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we do some other action that's not part of the action space, we will get an error, and the game will not accept anymore actions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RL Environment Setup

    Execute strategies with time limits to prevent infinite loops.
    """)
    return


@app.cell
def _(SudokuGame):
    from typing import Callable
    from unsloth import execute_with_time_limit

    def _execute_strategy(strategy: Callable, game: SudokuGame):
        """Execute a strategy function on a Sudoku game."""
        assert callable(strategy)
        max_moves = 100
        valid_moves = 0  # Track successful moves
        while (
            game.state() == "ongoing" and valid_moves < max_moves
        ):  # Track successful moves
            try:
                board = game.board()
                initial = game.initial_board()
                result = strategy(board, initial)
                if not isinstance(result, (tuple, list)) or len(result) != 3:
                    return (valid_moves, "failed")
                row, col, num = result
                if not all(
                    (isinstance(x, int) for x in [row, col, num])
                ):  # Validate result format
                    return (valid_moves, "failed")
                success = game.place_number(
                    row, col, num
                )  # Invalid format = immediate fail, but return valid moves made
                if success:
                    valid_moves = valid_moves + 1  # Track successful moves
                else:
                    return (valid_moves, "failed")
            except Exception:  # Validate types
                return (valid_moves, "failed")
        if valid_moves >= max_moves and game.state() == "ongoing":
            return (valid_moves, "failed")
        return (
            valid_moves,
            game.state(),
        )

    return Callable, execute_with_time_limit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To allow longer strategies for Reinforcement Learning, we shall allow a 10 second timer.
    """)
    return


@app.cell
def _(Callable, SudokuGame, execute_with_time_limit):
    @execute_with_time_limit(10)
    def execute_strategy(strategy: Callable, game: SudokuGame):
        """Execute strategy with 10 second time limit."""
        return _execute_strategy(strategy, game)

    return (execute_strategy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Test with a simple strategy:
    """)
    return


@app.cell
def _(SudokuGame, execute_strategy):
    def simple_strategy(board, initial):
        """Simple strategy: fill first empty cell with 1."""
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0 and initial[r][c] == 0:
                    return (r, c, 7)
        return (0, 0, 7)

    game_1 = SudokuGame(difficulty=30, seed=42)
    try:
        moves, state = execute_strategy(simple_strategy, game_1)
        print(f"Moves: {moves}, State: {state}")
    except TimeoutError as e:
        print(f"Timed out: {e}")
    return (game_1,)


@app.cell
def _(game_1):
    print(game_1.pretty())
    return


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
    from unsloth import check_python_modules, create_locked_down_function

    # Test safe code
    sample = """
    def strategy(board, initial):
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    return (r, c, 1)
        return (0, 0, 1)
    """

    ok, info = check_python_modules(sample)
    print("Safe Python code?", ok)
    print(info)
    return check_python_modules, create_locked_down_function


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the below piece of code, since we import `numpy`, we should not allow the execution:
    """)
    return


@app.cell
def _(check_python_modules):
    sample_1 = "\ndef strategy(board, initial):\n    import numpy as np\n    return (0, 0, 1)\n"
    ok_1, info_1 = check_python_modules(sample_1)
    print("Safe Python code?", ok_1)
    print(info_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data & RL task setup

    Create the prompt that instructs the model to generate a Sudoku solving strategy. You can customize this to some other task for another RL task.
    """)
    return


@app.cell
def _():
    prompt = """
    Create a Sudoku solving strategy using only native Python built-in functions without any import statements.
    You are given two lists of lists (9x9 grids):
    - board: current state (0 means empty)
    - initial: starting puzzle (0 means was empty, numbers are fixed)

    Return a tuple (row, col, number) for the next move.
    - row: 0-8 (row index)
    - col: 0-8 (column index)
    - number: 1-9 (digit to place)

    Only place numbers in cells that are BOTH empty in initial AND empty in board (initial[row][col] == 0 AND board[row][col] == 0)
    Use Sudoku rules: no duplicates in rows, columns, or 3x3 boxes.
    Output your function in backticks:
    ```python
    def strategy(board, initial):
        # Your logic here
        return (row, col, number)
    ```
    All helper functions must be inside def strategy. Output only the function.
    """.strip()

    print(prompt)
    return (prompt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, let's prompt the model without RL and see how it goes:
    """)
    return


@app.cell
def _(model_1, prompt, tokenizer):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt.strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )
    from transformers import TextStreamer

    print("=" * 50)
    print("BASE MODEL OUTPUT (before RL training):")
    print("=" * 50)
    _ = model_1.generate(
        **tokenizer(images=None, text=text, return_tensors="pt").to("cuda"),
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
    3. `strategy_succeeds` which checks if the game strategy actually succeeds in attaining Sudoku after running the auto-generated strategy.
    """)
    return


@app.function
def extract_function(text):
    """Extract Python function from markdown code blocks."""
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first:second].strip()
        fx = fx.removeprefix("python\n")
        fx = fx[fx.find("def") :]
        if fx.startswith("def strategy(board, initial):"):
            return fx
    return None


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reward 1: Function Works**

    Checks if the generated code is valid Python and can be executed.
    """)
    return


@app.cell
def _(check_python_modules, create_locked_down_function):
    def function_works(completions, **kwargs):
        """Reward for generating valid executable Python code."""
        scores = []
        for completion in completions:
            score = 0  # Invalid function
            response = completion[0]["content"]
            function = extract_function(response)

            if function is not None:
                ok, info = check_python_modules(function)

            if function is None or "error" in info:
                score = -2.0  # Invalid function
            else:
                try:
                    new_strategy = create_locked_down_function(function)
                    score = 1.0  # Valid function
                except:
                    score = -1.0  # Function has errors

            scores.append(score)
        return scores

    return (function_works,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reward 2: No Cheating**

    Penalizes functions that import external libraries.
    """)
    return


@app.cell
def _(check_python_modules):
    def no_cheating(completions, **kwargs):
        """Penalize use of external imports."""
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            function = extract_function(response)

            if function is not None:
                ok, info = check_python_modules(function)
                scores.append(1.0 if ok else -20.0)  # Heavy penalty for cheating
            else:
                scores.append(-1.0)  # Failed to create function

        return scores

    return (no_cheating,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reward 3: Strategy Succeeds**

    Rewards strategies that successfully solve Sudoku puzzles.
    """)
    return


@app.cell
def _(
    SudokuGame,
    check_python_modules,
    create_locked_down_function,
    execute_strategy,
):
    import numpy as np

    global PRINTER
    PRINTER = 0

    def strategy_succeeds(completions, **kwargs):
        """Reward valid moves even if strategy eventually fails."""
        global PRINTER
        scores = []
        seed = np.random.randint(10000)
        difficulty = 35
        for completion in completions:
            printed = False
            response = completion[0]["content"]
            function = extract_function(response)
            if PRINTER % 5 == 0:
                printed = True
                print("\n" + "=" * 60)
                print(function)
                print("=" * 60)
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
                game = SudokuGame(difficulty=difficulty, seed=seed)
                valid_moves, game_state = execute_strategy(new_strategy, game)
                if valid_moves == difficulty:
                    game_state = "success"
                print(f"\n Valid moves: {valid_moves}, Final state: {game_state}")
                if not printed:
                    print("Strategy:")
                    print(function[:200] + "..." if len(function) > 200 else function)
                print("\nFinal board:")
                print(game.pretty())
                if game_state == "success":
                    scores.append(30.0)
                elif valid_moves > 0:
                    reward = valid_moves * 0.2
                    scores.append(reward)
                else:
                    scores.append(-2.0)
            except TimeoutError:
                print("Timeout")
                scores.append(-1.0)
            except Exception as e:
                print(f"Exception: {str(e)[:100]}")
                scores.append(-3.0)  # Solved the puzzle!
        return scores  # Reward based on valid moves made before failure  # Each valid move is worth 0.2 points  # Failed immediately with no valid moves

    return PRINTER, strategy_succeeds


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dataset Preparation

    Create the training dataset.
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
            }
        ]
        * 1000
    )

    maximum_length = len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.strip()}], add_generation_prompt=True
        )
    )

    print(f"Maximum prompt length: {maximum_length}")
    print("\nDataset sample:")
    print(dataset[0])
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
        num_generations=4,  # Decrease if out of memory
        max_prompt_length=max_prompt_length,  # + 1 just in case!
        max_completion_length=max_completion_length,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps=200,
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

    You might have to wait 150 to 200 steps for any action. You'll probably get low reward for the first 100 steps. Please be patient!

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
    model_1.save_pretrained("grpo_saved_lora")  # Local saving
    tokenizer.save_pretrained("grpo_saved_lora")
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
    with safe_open("grpo_saved_lora/adapter_model.safetensors", framework="pt") as f:
        # Verify both A and B are non zero
        for key in f.keys():
            tensor = f.get_tensor(key)
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
    text_1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt.strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )
    _ = model_1.generate(
        **tokenizer(images=None, text=text_1, return_tensors="pt").to("cuda"),
        temperature=1.0,
        max_new_tokens=512,
        streamer=TextStreamer(tokenizer, skip_prompt=False),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a name="Save"></a>
    ### Saving to float16 for VLLM

    We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.
    """)
    return


@app.cell
def _(model_1, tokenizer):
    # Merge to 16bit
    if False:
        model_1.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
    if False:
        # Merge to 4bit
        model_1.push_to_hub_merged(
            "hf/model", tokenizer, save_method="merged_16bit", token=""
        )
    if False:
        model_1.save_pretrained_merged("model", tokenizer, save_method="merged_4bit")
    # Just LoRA adapters
    if False:
        model_1.push_to_hub_merged(
            "hf/model", tokenizer, save_method="merged_4bit", token=""
        )
    if False:
        model_1.save_pretrained("model")
        tokenizer.save_pretrained("model")
    if False:
        model_1.push_to_hub("hf/model", token="")
        tokenizer.push_to_hub("hf/model", token="")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### GGUF / llama.cpp Conversion
    To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.

    Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
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
        model_1.save_pretrained_gguf("model", tokenizer)
    if False:
        # Save to 16bit GGUF
        model_1.push_to_hub_gguf("hf/model", tokenizer, token="")
    if False:
        model_1.save_pretrained_gguf("model", tokenizer, quantization_method="f16")
    # Save to q4_k_m GGUF
    if False:
        model_1.push_to_hub_gguf(
            "hf/model", tokenizer, quantization_method="f16", token=""
        )
    if False:
        # Save to multiple GGUF options - much faster if you want multiple!
        model_1.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
    if False:
        model_1.push_to_hub_gguf(
            "hf/model", tokenizer, quantization_method="q4_k_m", token=""
        )  # Change hf to your username!
    if False:
        model_1.push_to_hub_gguf(
            "hf/model",
            tokenizer,
            quantization_method=["q4_k_m", "q8_0", "q5_k_m"],
            token="",
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, use the `model-unsloth.gguf` file or `model-unsloth-Q4_K_M.gguf` file in llama.cpp.

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
