#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth on your local device, follow [our guide](https://unsloth.ai/docs/get-started/install-and-update). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)
# 

# ### News

# 
# Long-Context GRPO for reinforcement learning - train stably at massive sequence lengths. Fine-tune models with up to 7x more context length efficiently. [Read Blog](https://unsloth.ai/docs/new/grpo-long-context)
# 
# New 3x faster training & 30% less VRAM. New kernels, padding-free & packing. [Blog](https://unsloth.ai/docs/new/3x-faster-training-packing)
# 
# You can now train with 500K context windows on a single 80GB GPU. [Blog](https://unsloth.ai/docs/new/500k-context-length-fine-tuning)
# 
# Unsloth's [Docker image](https://hub.docker.com/r/unsloth/unsloth) is here! Start training with no setup & environment issues. [Read our Guide](https://unsloth.ai/docs/new/how-to-train-llms-with-unsloth-and-docker).
# 
# New in Reinforcement Learning: [FP8 RL](https://unsloth.ai/docs/new/fp8-reinforcement-learning) • [Vision RL](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl) • [Standby](https://unsloth.ai/docs/basics/memory-efficient-rl) (faster, less VRAM RL) • [gpt-oss RL](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning)
# 
# Visit our docs for all our [model uploads](https://unsloth.ai/docs/get-started/all-our-models) and [notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks).
# 

# # ### Installation
# 
# # In[ ]:
# 
# 
# get_ipython().run_cell_magic('capture', '', 'import os\nos.environ["UNSLOTH_vllm_STANDBY"] = "1" # [NEW] Extra 30% context lengths!\nif "COLAB_" not in "".join(os.environ.keys()):\n    # If you\'re not in Colab, just use pip install or uv pip install\n    !pip install unsloth vllm\nelse:\n    pass # For Colab / Kaggle, we need extra instructions hidden below \\/\n')
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
#     _vllm, _triton = ('vllm==0.9.2', 'triton==3.2.0') if is_t4 else ('vllm==0.10.2', 'triton')
#     get_ipython().system('uv pip install -qqq --upgrade {_vllm} {_numpy} {_pil} torchvision bitsandbytes xformers unsloth')
#     get_ipython().system('uv pip install -qqq {_triton}')
# get_ipython().system('uv pip install transformers==4.56.2')
# get_ipython().system('uv pip install --no-deps trl==0.22.2')
# 
# 
# # ### Unsloth

# In[3]:


from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

    "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/gemma-3-4b-it",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)


# We now add LoRA adapters for parameter efficient fine-tuning, allowing us to train only 1% of all model parameters efficiently.
# 
# **[NEW]** We also support fine-tuning only the vision component, only the language component, or both. Additionally, you can choose to fine-tune the attention modules, the MLP layers, or both!

# In[4]:


model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    use_gradient_checkpointing = "unsloth", # Reduces memory usage
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)


# <a name="Data"></a>
# ### Data Prep
# AI4Math/MathVista is a dataset that involves using images to solve logic and math problems, for this notebook, it will only be math problems with numeric answers for simpilicity.

# In[5]:


from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import torch

dataset = load_dataset("AI4Math/MathVista",split = "testmini")


# Let us see what our data looks like

# In[6]:


dataset["decoded_image"][5]


# In[7]:


dataset["question"][5], dataset["answer"][5]


# The image of our data is a part of the math problem.
# 
# To make the rewarding easy later on, let us filter only the numeric answer so that we can create reward function that gives score if reward is float or not.

# In[8]:


def is_numeric_answer(example):
  try:
    float(example["answer"])
    return True
  except:
    return False

dataset = dataset.filter(is_numeric_answer) #


# We also resize the images to be 512 by 512 pixels to make the images managable in context length. We also convert them to RGB so they are compatible with TRL's trainer!

# In[9]:


# Filter have big images
def resize_images(example):
    image = example["decoded_image"]
    image = image.resize((512,512))
    example["decoded_image"] = image
    return example

dataset = dataset.map(resize_images)

def convert_to_rgb(example):
    image = example["decoded_image"]
    if image.mode != "RGB":
        image = image.convert("RGB")
    example["decoded_image"] = image
    return example

dataset = dataset.map(convert_to_rgb)


# This is the conversational template that is needed to collate the dataset

# In[10]:


# Define the delimiter variables for clarity and easy modification
REASONING_START = "<REASONING>"
REASONING_END = "</REASONING>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

def make_conversation(example):
    # Define placeholder constants if they are not defined globally

    # The user's text prompt
    text_content = (
        f"{example['question']}, provide your reasoning between {REASONING_START} and {REASONING_END} "
        f"and then your final answer between {SOLUTION_START} and (put a float here) {SOLUTION_END}"
    )

    # Construct the prompt in the desired multi-modal format
    prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image"},  # Placeholder for the image
                {"type": "text", "text": text_content},  # The text part of the prompt
            ],
        },
    ]

    # The actual image data is kept separate for the processor
    return {"prompt": prompt, "image": example["decoded_image"], "answer": example["answer"]}

train_dataset = dataset.map(make_conversation)

#We reformatting dataset like this because decoded_images are the actual images
#The "image": example["decoded_image"] does not properly format the dataset correctly

# 1. Remove the original 'image' column
train_dataset = train_dataset.remove_columns("image")

# 2. Rename 'decoded_image' to 'image'
train_dataset = train_dataset.rename_column("decoded_image", "image")



# Applying Chat Template across the entire dataset

# In[11]:


train_dataset = train_dataset.map(
    lambda example: {
        "prompt": tokenizer.apply_chat_template(
            example["prompt"],
            tokenize = False,
            add_generation_prompt = False
        )
    }
)


# We use a basic formatting functions to see if reasoning starts and ends as well as if the answers were written correctly.

# In[12]:


# Reward functions
def formatting_reward_func(completions,**kwargs):
    import re
    thinking_pattern = f'{REASONING_START}(.*?){REASONING_END}'
    answer_pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'

    scores = []
    for completion in completions :
      score = 0
      thinking_matches = re.findall(thinking_pattern, completion, re.DOTALL)
      answer_matches = re.findall(answer_pattern, completion, re.DOTALL)
      if len(thinking_matches) == 1 :
        score +=1.0
      if len(answer_matches) == 1 :
        score +=1.0
      scores.append(score)
    return scores


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    import re

    answer_pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'

    responses = [re.findall(answer_pattern, completion, re.DOTALL) for completion in completions]
    q = prompts[0]

    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:{completions[0]}")
    return [2.0 if len(r)==1 and a == r[0].replace('\n','') else 0.0 for r, a in zip(responses, answer)]


# Here is the first example prompt in the dataset

# In[13]:


train_dataset[0]["prompt"]


# <a name="Train"></a>
# ### Train the model
# 
# Now set up GRPO Trainer and all configurations!

# In[14]:


from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 2, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = 1024,
    max_completion_length = 1024,
    #num_train_epochs = 2, # Set to 1 for a full training run
    importance_sampling_level = "sequence",
    mask_truncated_completions = False,
    loss_type = 'dr_grpo',
    max_steps = 60,
    save_steps = 60,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
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

# In[15]:


trainer = GRPOTrainer(
    model = model,
    args = training_args,
    # Pass the processor to handle multimodal inputs
    processing_class = tokenizer,
    reward_funcs = [formatting_reward_func, correctness_reward_func],
    train_dataset = train_dataset,
)

trainer.train()


# <a name="Inference"></a>
# ### Inference
# Let's run the model! You can modify the instruction and input—just leave the output blank.
# 
# We'll use the best hyperparameters for inference on Gemma: `top_p=0.95`, `top_k=64`, and `temperature=1.0`.

# In[25]:


FastVisionModel.for_inference(model)  # Enable for inference!

image = dataset[100]["decoded_image"]
instruction = (
    f"{dataset[100]["question"]}, provide your reasoning between {REASONING_START} and {REASONING_END} "
    f"and then your final answer between {SOLUTION_START} and (put a float here) {SOLUTION_END}"
)

messages = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": instruction}],
    }
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(tokenizer, skip_prompt = True)
result = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                        use_cache = True, temperature = 1.0, top_p = 0.95, top_k = 64)


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, use Hugging Face’s `push_to_hub` for online saving, or `save_pretrained` for local storage.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[18]:


model.save_pretrained("gemma_3_lora")  # Local saving
tokenizer.save_pretrained("gemma_3_lora")
# model.push_to_hub("your_name/gemma_3_lora", token = "YOUR_HF_TOKEN") # Online saving
# processor.push_to_hub("your_name/gemma_3_lora", token = "YOUR_HF_TOKEN") # Online saving


# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# In[27]:


if False:
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        model_name = "gemma_3_lora",  # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = True,  # Set to False for 16bit LoRA
    )
    FastVisionModel.for_inference(model)  # Enable for inference!

FastVisionModel.for_inference(model)  # Enable for inference!

sample = dataset[1]
image = sample["decoded_image"].convert("RGB")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"{sample["question"]}, provide your reasoning between {REASONING_START} and {REASONING_END} "
                    f"and then your final answer between {SOLUTION_START} and (put a float here) {SOLUTION_END}"
                )
            },
            {
                "type": "image",
            },
        ],
    },
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(tokenizer, skip_prompt = True)
result = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                        use_cache = True, temperature = 1.0, top_p = 0.95, top_k = 64)


# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens. See [our docs](https://unsloth.ai/docs/basics/inference-and-deployment) for more deployment options.

# In[28]:


# Select ONLY 1 to save! (Both not needed!)

# Save locally to 16bit
if False: model.save_pretrained_merged("unsloth_finetune", tokenizer,)

# To export and save to your Hugging Face account
if False: model.push_to_hub_merged("YOUR_USERNAME/unsloth_finetune", tokenizer, token = "YOUR_HF_TOKEN")


# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other resources:
# 1. Looking to use Unsloth locally? Read our [Installation Guide](https://unsloth.ai/docs/get-started/install-and-update) for details on installing Unsloth on Windows, Docker, AMD, Intel GPUs.
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
# 
