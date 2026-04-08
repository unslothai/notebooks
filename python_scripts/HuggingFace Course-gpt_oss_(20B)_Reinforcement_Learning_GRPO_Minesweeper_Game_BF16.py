#!/usr/bin/env python
# coding: utf-8

# # Minesweeper LLM - Custom GRPO Training
# 
# ## Goal
# Finetune an LLM with LoRA using GRPO to play Minesweeper by:
# - **Input**: JSON game state (board configuration)
# - **Output**: JSON action (reveal or flag a cell)
# 
# Teams will compete to train the best Minesweeper-playing LLM!
# 
# ## Training Approach
# - **Model**: GPT-OSS 20B with LoRA
# - **Method**: GRPO (Group Relative Policy Optimization)
# - **Framework**: Unsloth (2-6x faster, 70% less VRAM)
# - **Hardware**: AMD GPU (ROCm)

# # Installation
# 
# Install Unsloth and dependencies optimized for AMD GPUs:

# In[ ]:


get_ipython().run_cell_magic('capture', '', 'import os, importlib.util\n!pip install --upgrade -qqq uv\nif importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):\n    try: import numpy; get_numpy = f"numpy=={numpy.__version__}"\n    except: get_numpy = "numpy"\n    !uv pip install -qqq \\\n        "torch>=2.8.0" "triton>=3.4.0" {get_numpy} torchvision bitsandbytes "transformers==4.56.2" \\\n        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \\\n        "unsloth[base] @ git+https://github.com/unslothai/unsloth" \\\n        git+https://github.com/triton-lang/triton.git@05b2c186c1b6c9a08375389d5efe9cb4c401c075#subdirectory=python/triton_kernels\nelif importlib.util.find_spec("unsloth") is None:\n    !uv pip install -qqq unsloth\n!uv pip install --upgrade --no-deps transformers==4.56.2 tokenizers trl==0.22.2 unsloth unsloth_zoo\n')


# # Load Model with Unsloth
# 
# Load GPT-OSS 20B with LoRA configuration:

# In[ ]:


from unsloth import FastLanguageModel
import torch

max_seq_length = 1024  # Max context length
lora_rank = 16         # LoRA rank (higher = smarter but slower; 4 is too low for reasoning tasks)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b-BF16",
    load_in_4bit = False,
    max_seq_length = max_seq_length,
    torch_dtype = torch.bfloat16,
)

# Force model to cuda explicitly
print(f"Model device: {model.device}")
print("Model loaded successfully!")


# # Add LoRA Adapters
# 
# Add LoRA layers for efficient finetuning:

# In[ ]:


model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = lora_rank * 2,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)


# # Minesweeper Game Implementation
# 
# Custom Minesweeper environment supporting:
# - Customizable board size and mine count
# - Actions: reveal or flag cells
# - Win: reveal all safe cells
# - Lose: reveal a mine

# In[ ]:


from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set
import random

@dataclass
class MinesweeperGame:
    rows: int
    cols: int
    num_mines: int
    seed: Optional[int] = None
    _rng: random.Random = field(init = False, repr = False)
    _board: List[List[int]] = field(init = False, repr = False)  # -1 = mine, 0-8 = count
    _revealed: Set[Tuple[int, int]] = field(init = False, repr = False, default_factory = set)
    _flagged: Set[Tuple[int, int]] = field(init = False, repr = False, default_factory = set)
    _state: str = field(default = "ongoing", init = False, repr = False)

    def __post_init__(self):
        if self.num_mines >= self.rows * self.cols:
            raise ValueError("Too many mines for board size")
        self._rng = random.Random(self.seed)
        self._board = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self._place_mines()
        self._calculate_numbers()

    def _place_mines(self):
        """Place mines randomly on the board"""
        positions = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        mine_positions = self._rng.sample(positions, self.num_mines)
        for r, c in mine_positions:
            self._board[r][c] = -1

    def _calculate_numbers(self):
        """Calculate numbers for each cell based on adjacent mines"""
        for r in range(self.rows):
            for c in range(self.cols):
                if self._board[r][c] == -1:
                    continue
                count = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            if self._board[nr][nc] == -1:
                                count += 1
                self._board[r][c] = count

    def _reveal_cell(self, row: int, col: int) -> bool:
        """Reveal a cell. Returns True if valid move, False if invalid.
        Uses iterative flood-fill to avoid recursion limit on large boards.
        (Issue #11: was recursive; Issue typo: fixed 'bself' -> 'self')
        """
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        if (row, col) in self._revealed or (row, col) in self._flagged:
            return False

        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if (r, c) in self._revealed:
                continue

            self._revealed.add((r, c))

            # Hit a mine!
            if self._board[r][c] == -1:
                self._state = "failed"
                return True

            # Auto-reveal neighbors if cell is 0
            if self._board[r][c] == 0:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < self.rows and 0 <= nc < self.cols
                                and (nr, nc) not in self._revealed
                                and (nr, nc) not in self._flagged):
                            stack.append((nr, nc))

        return True

    def _flag_cell(self, row: int, col: int) -> bool:
        """Flag/unflag a cell. Returns True if valid, False if invalid"""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        if (row, col) in self._revealed:
            return False

        if (row, col) in self._flagged:
            self._flagged.remove((row, col))
        else:
            self._flagged.add((row, col))
        return True

    def do_action(self, action: dict) -> str:
        """Execute an action and return a status string.

        Returns one of:
          'ok'               - valid move executed
          'mine'             - revealed a mine (game over)
          'win'              - game won after this move
          'invalid_format'   - bad action dict / missing keys / bad types
          'out_of_bounds'    - coordinates outside the board
          'already_revealed' - cell was already revealed
          'flagged_cell'     - tried to reveal a flagged cell
          'invalid_flag'     - tried to flag a revealed cell
          'game_over'        - game was already over before this call

        Only sets state = 'failed' for moves that would change the board state,
        while formatting errors return 'invalid_format' without changing state.
        """
        if self._state != "ongoing":
            return "game_over"

        if not isinstance(action, dict):
            self._state = "failed"
            return "invalid_format"

        action_type = action.get("type")
        row = action.get("row")
        col = action.get("col")

        if action_type not in ["reveal", "flag"] or row is None or col is None:
            self._state = "failed"
            return "invalid_format"

        try:
            row, col = int(row), int(col)
        except (ValueError, TypeError):
            self._state = "failed"
            return "invalid_format"

        if not (0 <= row < self.rows and 0 <= col < self.cols):
            self._state = "failed"
            return "out_of_bounds"

        if action_type == "reveal":
            if (row, col) in self._revealed:
                self._state = "failed"
                return "already_revealed"
            if (row, col) in self._flagged:
                self._state = "failed"
                return "flagged_cell"
            valid = self._reveal_cell(row, col)
        else:
            if (row, col) in self._revealed:
                self._state = "failed"
                return "invalid_flag"
            valid = self._flag_cell(row, col)

        if not valid:
            self._state = "failed"
            return "invalid_format"

        self._check_win()

        if self._state == "failed":
            return "mine"
        if self._state == "success":
            return "win"
        return "ok"

    def _check_win(self):
        """Check if player has won"""
        total_cells = self.rows * self.cols
        safe_cells = total_cells - self.num_mines
        if len(self._revealed) == safe_cells:
            self._state = "success"

    def get_visible_board(self) -> List[List[str]]:
        """Get board state as player sees it"""
        visible = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) in self._flagged:
                    row.append('F')
                elif (r, c) in self._revealed:
                    val = self._board[r][c]
                    row.append('*' if val == -1 else str(val))
                else:
                    row.append('.')
            visible.append(row)
        return visible

    def state(self) -> str:
        return self._state

    def pretty_print(self) -> str:
        """Pretty print the board"""
        visible = self.get_visible_board()
        lines = []

        # Header
        header = "   " + " ".join(f"{i:2d}" for i in range(self.cols))
        lines.append(header)
        lines.append("  " + "─" * (self.cols * 3 + 1))

        # Board
        for r, row in enumerate(visible):
            line = f"{r:2d}│ " + "  ".join(row)
            lines.append(line)

        return "\n".join(lines)


# # Test the Game

# In[ ]:


# Create test game
game = MinesweeperGame(rows = 6, cols = 6, num_mines = 5, seed = 42)
print(game.pretty_print())
print(f"State: {game.state()}")

# Test action
game.do_action({"type": "reveal", "row": 0, "col": 0})
print("\nAfter revealing (0,0):")
print(game.pretty_print())
print(f"State: {game.state()}")


# # JSON Input/Output Format
# 
# ## Input Format (Game State)
# ```json
# {
#   "board": [
#     [".", ".", ".", ".", ".", "."],
#     [".", ".", ".", ".", ".", "."],
#     [".", ".", ".", ".", ".", "."],
#     [".", ".", ".", ".", ".", "."],
#     [".", ".", ".", ".", ".", "."],
#     [".", ".", ".", ".", ".", "."]
#   ],
#   "rows": 6,
#   "cols": 6,
#   "mines": 5
# }
# ```
# 
# ## Output Format (Action)
# ```json
# {"type": "reveal", "row": 2, "col": 3}
# ```
# or
# ```json
# {"type": "flag", "row": 1, "col": 4}
# ```

# In[ ]:


import json

def format_state_for_llm(game: MinesweeperGame) -> str:
    """Convert game state to JSON prompt for LLM"""
    state = {
        "board": game.get_visible_board(),
        "rows": game.rows,
        "cols": game.cols,
        "mines": game.num_mines,
        "flags_placed": len(game._flagged),
        "cells_revealed": len(game._revealed),
    }

    prompt = f"""You are playing Minesweeper. Analyze the game state and output your next move.

Game state:
{json.dumps(state, indent = 2)}

Legend:
- "." = unrevealed cell
- "F" = flagged cell (suspected mine)
- "0"-"8" = number of adjacent mines
- "*" = revealed mine (game over)

Output your next action as JSON:
{{"type": "reveal", "row": <row_index>, "col": <col_index>}}
or
{{"type": "flag", "row": <row_index>, "col": <col_index>}}

Your action:"""

    return prompt

def parse_llm_action(response: str) -> dict:
    """Extract JSON action from LLM response.

    Finds all JSON-like objects and returns the LAST one matching the
    expected schema.  LLMs typically reason through options and place
    their final answer at the end, so taking the last valid match is
    more robust than taking the first.
    """
    import re
    best = None
    for match in re.finditer(r'\{[^{}]*\}', response):
        try:
            action = json.loads(match.group())
            if ("type" in action and "row" in action and "col" in action
                    and action["type"] in ["reveal", "flag"]):
                best = action
        except json.JSONDecodeError:
            continue
    return best

# Test formatting
game = MinesweeperGame(rows = 6, cols = 6, num_mines = 5, seed = 42)
prompt = format_state_for_llm(game)
print(prompt[:500] + "...")


# # Test Model Before Training
# 
# See how the base model performs without finetuning:

# In[ ]:


from transformers import TextStreamer

game = MinesweeperGame(rows = 6, cols = 6, num_mines = 5, seed = 42)
prompt = format_state_for_llm(game)

# reasoning_effort = "low" — GRPOTrainer does NOT pass it
# during training, so using it only at eval creates a train/eval mismatch.
text = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize = False,
    add_generation_prompt = True,
)

print("=== Base Model Response ===")
output = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    temperature = 1.0,
    max_new_tokens = 128,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)


# # GRPO Reward Functions
# 
# Define reward functions to guide the model's learning:

# In[ ]:


import numpy as np


def _is_logically_safe(game, row, col):
    """Check if revealing (row, col) is provably safe via constraint propagation.

    A cell is logically safe if there is at least one adjacent *revealed* number
    cell whose mine count equals its number of flagged neighbors.  That means
    every remaining unrevealed neighbor of that number cell (including (row, col))
    must be safe.
    """
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if not (0 <= nr < game.rows and 0 <= nc < game.cols):
                continue
            if (nr, nc) not in game._revealed:
                continue
            number = game._board[nr][nc]
            if number <= 0:
                continue
            # Count flagged neighbors of the numbered cell
            flagged_neighbors = 0
            for dr2 in [-1, 0, 1]:
                for dc2 in [-1, 0, 1]:
                    if dr2 == 0 and dc2 == 0:
                        continue
                    nnr, nnc = nr + dr2, nc + dc2
                    if (0 <= nnr < game.rows and 0 <= nnc < game.cols
                            and (nnr, nnc) in game._flagged):
                        flagged_neighbors += 1
            if flagged_neighbors == number:
                return True  # All mines accounted for → (row, col) is provably safe
    return False


def valid_json_reward(completions, **kwargs):
    """Reward valid JSON action format"""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        action = parse_llm_action(response)

        if action is None:
            scores.append(-5.0)  # Invalid format
        else:
            scores.append(1.0)   # Valid format

    return scores


def gameplay_reward(completions, **kwargs):
    """
    Clean reward function based on defined criteria.

    Reward Criteria:
    1.  Flag cell that IS a mine       → +15
    2.  Flag cell that is NOT a mine    → -10
    3.  Reveal cell that IS a mine      → -25 (game over)
    4.  Reveal cell that is safe        → +10  (+15 if logically deducible)
    5.  Flag already flagged cell       → -8
    6.  Reveal already revealed cell    → -12
    7.  Out of bounds                   → -15
    8.  Total flags > total mines       → -10
    9.  Invalid JSON                    → -10
    10. Win the game                    → +100 (big bonus)
    11. Reveal a flagged cell           → -8  (Issue #3)
    """
    scores = []

    # Get game state info passed from dataset
    seeds = kwargs.get("seed", [])
    move_histories = kwargs.get("move_history", [])

    for idx, completion in enumerate(completions):
        response = completion[0]["content"]
        action = parse_llm_action(response)

        # Criterion 9: Invalid JSON
        if action is None:
            scores.append(-10.0)
            continue

        # Reconstruct EXACT game state
        if idx < len(seeds) and idx < len(move_histories):
            seed = seeds[idx]
            move_history_raw = move_histories[idx]

            # Issue #1: move_history is stored as a JSON string in the dataset
            # to avoid HF Dataset schema inference mangling list-of-dicts.
            if isinstance(move_history_raw, str):
                move_history = json.loads(move_history_raw)
            else:
                move_history = move_history_raw  # backward compat

            # Reconstruct game to exact state from prompt
            game = MinesweeperGame(rows = 6, cols = 6, num_mines = 5, seed = seed)
            for prev_action in move_history:
                game.do_action(prev_action)

            board = game.get_visible_board()
            row, col = action["row"], action["col"]
            action_type = action["type"]

            # Criterion 7: Out of bounds
            if not (0 <= row < game.rows and 0 <= col < game.cols):
                scores.append(-15.0)
                continue

            # Get ground truth and current state
            is_mine = game._board[row][col] == -1  # Ground truth (hidden from LLM)
            is_revealed = (row, col) in game._revealed
            is_flagged = (row, col) in game._flagged

            # --- FLAG ACTION ---
            if action_type == "flag":
                # Criterion 6: Can't flag already revealed cells
                if is_revealed:
                    scores.append(-12.0)
                    continue

                # Criterion 5: Flag already flagged cell
                if is_flagged:
                    scores.append(-8.0)
                    continue

                # Criterion 8: Check if exceeding mine count
                current_flag_count = len(game._flagged)
                if current_flag_count >= game.num_mines:
                    scores.append(-10.0)
                    continue

                # Criterion 1 & 2: Check if flagged cell is actually a mine
                if is_mine:
                    scores.append(15.0)  # Correct flag!
                else:
                    scores.append(-10.0)  # Wrong flag
                continue

            # --- REVEAL ACTION ---
            elif action_type == "reveal":
                # Criterion 6: Reveal already revealed cell
                if is_revealed:
                    scores.append(-12.0)
                    continue

                # Issue #3: Reveal a flagged cell — was falling through to
                # game.do_action which returns False → state = "failed" → -25 mine penalty.
                if is_flagged:
                    scores.append(-8.0)
                    continue

                # Apply the reveal action
                game.do_action(action)

                # Criterion 10: Win the game
                if game.state() == "success":
                    scores.append(100.0)  # Big bonus!
                    continue

                # Criterion 3: Revealed a mine (game over)
                if game.state() == "failed":
                    scores.append(-25.0)
                    continue

                # Criterion 4: Revealed safe cell
                # Issue #9: Bonus for logically deducible moves
                if _is_logically_safe(game, row, col):
                    scores.append(15.0)  # Provably safe → higher reward
                else:
                    scores.append(10.0)  # Random/uncertain safe reveal
                continue

            # Unknown action type
            else:
                scores.append(-10.0)
        else:
            # Fallback
            scores.append(0.0)

    return scores

print("Reward function created!")
print("")
print("Rewards:")
print("  +100: Win game (revealed all safe cells)")
print("  +15: Flag actual mine / logically deducible safe reveal")
print("  +10: Reveal safe cell (uncertain/random)")
print("  +1:  Valid JSON format")
print("")
print("Penalties:")
print("  -25: Reveal mine (game over)")
print("  -15: Out of bounds")
print("  -12: Reveal already revealed cell")
print("  -10: Flag non-mine, exceed mine count, invalid JSON")
print("  -8:  Flag already flagged / reveal flagged cell")
print("  -5:  Invalid JSON format")


# # Create Training Dataset
# 
# Generate diverse game states for training:

# In[ ]:


from datasets import Dataset

def generate_game_states(num_samples = 1000, rows = 6, cols = 6, num_mines = 5,
                         rng_seed = 42):
    """
    Generate EXACTLY num_samples diverse Minesweeper game states.

    Mix of:
    - Fresh games (20-30%)
    - Mid-game states (70-80%)

    IMPORTANTLY: Stores seed + move_history (as JSON string) so reward
    function can reconstruct the EXACT game state!

    Keeps generating until we have exactly num_samples valid ongoing games.

    Args:
        rng_seed: Seed for numpy/random RNG for reproducibility (Issue #10).
    """
    #Seed RNG for reproducibility across runs
    np.random.seed(rng_seed)
    random.seed(rng_seed)

    dataset_items = []
    attempts = 0
    max_attempts = num_samples * 3  # Safety limit

    while len(dataset_items) < num_samples and attempts < max_attempts:
        attempts += 1
        seed = np.random.randint(100000)
        game = MinesweeperGame(rows = rows, cols = cols, num_mines = num_mines, seed = seed)

        # Make 0-5 random moves (0 = fresh game, 1-5 = mid-game)
        num_moves = np.random.randint(0, 6)
        move_history = []

        for _ in range(num_moves):
            board = game.get_visible_board()
            unrevealed = []
            for r in range(rows):
                for c in range(cols):
                    if board[r][c] == '.':
                        unrevealed.append((r, c))

            if unrevealed and game.state() == "ongoing":
                r, c = random.choice(unrevealed)
                action = {"type": "reveal", "row": r, "col": c}
                game.do_action(action)
                move_history.append(action)
            else:
                break

        # Only add ongoing games (skip failed/completed games)
        if game.state() == "ongoing":
            prompt_text = format_state_for_llm(game)
            dataset_items.append({
                "prompt": [{"role": "user", "content": prompt_text}],
                "seed": seed,  # Store seed to reconstruct game
                # Serialize as JSON string to avoid HF Dataset
                # schema inference mangling list-of-dicts into dict-of-lists
                "move_history": json.dumps(move_history),
            })

    return Dataset.from_list(dataset_items)

# Generate training dataset
print("Generating training dataset...")
dataset = generate_game_states(num_samples = 1000, rows = 6, cols = 6, num_mines = 5)
print(f"Created EXACTLY {len(dataset)} training examples (all ongoing games)")

# Count fresh vs mid-game
fresh_count = sum(1 for item in dataset if item["move_history"] == "[]")
print(f"  Fresh games: {fresh_count} ({fresh_count/len(dataset)*100:.1f}%)")
print(f"  Mid-game states: {len(dataset) - fresh_count} ({(len(dataset)-fresh_count)/len(dataset)*100:.1f}%)")

# Show example
print("\nExample training prompt:")
print(dataset[0]["prompt"][0]["content"][:400] + "...")
print(f"Seed: {dataset[0]['seed']}, Previous moves: {len(json.loads(dataset[0]['move_history']))}")


# # Configure GRPO Training
# 
# Set up GRPO trainer with all hyperparameters:

# In[ ]:


from trl import GRPOConfig, GRPOTrainer

# Calculate max lengths
max_prompt_length = 600   # JSON state prompt
max_completion_length = max_seq_length - max_prompt_length

# GRPO Configuration
training_args = GRPOConfig(
    temperature = 1.0,
    learning_rate = 5e-5,
    weight_decay = 0.01,
    warmup_ratio = 0.1,
    lr_scheduler_type = "linear",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,  # 4 gives 16 effective completions per update
    num_generations = 4,  # Generate 4 actions per state
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    max_steps = 500,      # Adjust based on compute budget
    save_steps = 100,
    report_to = "none",
    output_dir = "minesweeper_custom_outputs",
)

print("Training configuration:")
print(f"  Max steps: {training_args.max_steps}")
print(f"  Generations per state: {training_args.num_generations}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  LoRA rank: {lora_rank}")


# In[ ]:


from transformers import TrainerCallback

class MinesweeperEvalCallback(TrainerCallback):
    """Periodically play games during training and log win rate.
    (Issue #8: no validation / reward tracking in the original notebook.)
    """

    def __init__(self, eval_every_steps = 50, num_games = 5):
        self.eval_every_steps = eval_every_steps
        self.num_games = num_games

    def on_step_end(self, args, state, control, model = None, processing_class = None, **kwargs):
        if state.global_step % self.eval_every_steps != 0:
            return

        tokenizer = processing_class
        if tokenizer is None or model is None:
            return

        # Temporarily set model to eval mode
        was_training = model.training
        model.eval()

        wins = 0
        for i in range(self.num_games):
            game = MinesweeperGame(rows = 6, cols = 6, num_mines = 5, seed = 10000 + i)
            moves = 0
            while game.state() == "ongoing" and moves < 50:
                prompt = format_state_for_llm(game)
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize = False,
                    add_generation_prompt = True,
                )
                output = model.generate(
                    **tokenizer(text, return_tensors = "pt").to(model.device),
                    temperature = 0.7,
                    max_new_tokens = 128,
                    do_sample = True,
                )
                response = tokenizer.decode(output[0], skip_special_tokens = True)
                action = parse_llm_action(response)
                if action is None:
                    break
                game.do_action(action)
                moves += 1
            if game.state() == "success":
                wins += 1

        win_rate = wins / self.num_games
        print(f"\n[Eval @ step {state.global_step}] Win rate: {wins}/{self.num_games} ({win_rate*100:.0f}%)\n")

        if was_training:
            model.train()

eval_callback = MinesweeperEvalCallback(eval_every_steps = 50, num_games = 5)
print("Eval callback created: plays 5 games every 50 steps")


# # Train the Model
# 
# Start GRPO training with reward functions:

# In[ ]:


trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        valid_json_reward,   # Reward valid JSON format
        gameplay_reward,     # Reward good gameplay
    ],
    args = training_args,
    train_dataset = dataset,
    callbacks = [eval_callback],  # Issue #8: periodic gameplay evaluation
)

print("Starting training...")
trainer.train()


# # Test Trained Model
# 
# Evaluate the finetuned model:

# In[ ]:


# Test on new game
test_game = MinesweeperGame(rows = 6, cols = 6, num_mines = 5, seed = 999)
test_prompt = format_state_for_llm(test_game)

# Issue #7: removed reasoning_effort = "low" for train/eval consistency
test_text = tokenizer.apply_chat_template(
    [{"role": "user", "content": test_prompt}],
    tokenize = False,
    add_generation_prompt = True,
)

print("=== Trained Model Response ===")
output = model.generate(
    **tokenizer(test_text, return_tensors = "pt").to("cuda"),
    temperature = 0.7,
    max_new_tokens = 128,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

# Parse and test action
response_text = tokenizer.decode(output[0])
action = parse_llm_action(response_text)
print(f"\nParsed action: {action}")

if action:
    test_game.do_action(action)
    print(f"\nGame state after action: {test_game.state()}")
    print(test_game.pretty_print())


# # Evaluation: Play Complete Games
# 
# Test the model on multiple complete games:

# In[ ]:


def play_full_game(model, tokenizer, rows = 6, cols = 6, num_mines = 5, seed = None, max_moves = 50):
    """Play a complete Minesweeper game with the model"""
    game = MinesweeperGame(rows = rows, cols = cols, num_mines = num_mines, seed = seed)
    moves = 0

    while game.state() == "ongoing" and moves < max_moves:
        # Get current state
        prompt = format_state_for_llm(game)
        # Removed reasoning_effort = "low" for train/eval consistency
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize = False,
            add_generation_prompt = True,
        )

        # Generate action
        output = model.generate(
            **tokenizer(text, return_tensors = "pt").to("cuda"),
            temperature = 0.7,
            max_new_tokens = 128,
            do_sample = True,
        )

        response = tokenizer.decode(output[0])
        action = parse_llm_action(response)

        if action is None:
            break  # Invalid action

        game.do_action(action)
        moves += 1

    return game, moves

# Evaluate on 100 games for statistically meaningful win rates
NUM_EVAL_GAMES = 100
print(f"Evaluating model on {NUM_EVAL_GAMES} games...\n")
wins = 0
total_moves = 0

for i in range(NUM_EVAL_GAMES):
    game, moves = play_full_game(model, tokenizer, seed = i)
    result = game.state()

    if result == "success":
        wins += 1
    # Only print individual results for first 10 + any wins
    if i < 10 or result == "success":
        tag = "WIN" if result == "success" else "LOSS"
        print(f"Game {i+1}: {tag} ({result}) after {moves} moves")

    total_moves += moves

if NUM_EVAL_GAMES > 10:
    print(f"... (showing first 10 + wins; {NUM_EVAL_GAMES} total)")

print(f"\nResults:")
print(f"  Win rate: {wins}/{NUM_EVAL_GAMES} ({wins/NUM_EVAL_GAMES*100:.1f}%)")
print(f"  Average moves: {total_moves/NUM_EVAL_GAMES:.1f}")


# # Save the Model
# 
# Save your trained model for competition submission:

# In[ ]:


# Save LoRA adapters
model.save_pretrained("gpt_oss_lora")
tokenizer.save_pretrained("gpt_oss_lora")

print("Model saved to: my_minesweeper_model/")

# Optional: Save merged model in 16bit
if False:
    model.save_pretrained_merged(
        "gpt_oss_finetune_16bit",
        tokenizer,
        save_method = "merged_16bit"
    )

# Optional: Push to Hugging Face Hub
if False:
    model.push_to_hub_merged(
        "your-username/minesweeper-gpt-oss",
        tokenizer,
        save_method = "lora",
        token = "YOUR_HF_TOKEN"
    )


# # Tips
# 
# ## Improve Your Model:
# 
# 1. **Adjust Reward Functions**
#    - Increase rewards for logical deduction
#    - Add penalties for random moves
#    - Reward flagging correct mines
# 
# 2. **Tune Hyperparameters**
#    - Increase `max_steps` for longer training
#    - Adjust `learning_rate` (try 1e-5 to 1e-4)
#    - Increase `lora_rank` for more capacity
#    - Adjust `num_generations` (2-8)
# 
# 3. **Better Training Data**
#    - Generate more diverse states
#    - Include harder scenarios (more mines)
#    - Add states requiring logical deduction
# 
# 4. **Advanced Techniques**
#    - Multi-step rollouts in reward function
#    - Curriculum learning (easy → hard boards)
#    - Ensemble multiple models
# 
# ## Useful Strategies:
# - Experiment with different reward functions
# - Try different board sizes during training
# - Analyze failed games to improve rewards
# - Use temperature sampling during evaluation
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
