#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth your local device, follow [our guide](https://docs.unsloth.ai/get-started/install-and-update). This notebook is licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)
# 

# ### News

# 
# Introducing FP8 precision training for faster RL inference. [Read Blog](https://docs.unsloth.ai/new/fp8-reinforcement-learning).
# 
# Unsloth's [Docker image](https://hub.docker.com/r/unsloth/unsloth) is here! Start training with no setup & environment issues. [Read our Guide](https://docs.unsloth.ai/new/how-to-train-llms-with-unsloth-and-docker).
# 
# [gpt-oss RL](https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning) is now supported with the fastest inference & lowest VRAM. Try our [new notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb) which creates kernels!
# 
# Introducing [Vision](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl) and [Standby](https://docs.unsloth.ai/basics/memory-efficient-rl) for RL! Train Qwen, Gemma etc. VLMs with GSPO - even faster with less VRAM.
# 
# Visit our docs for all our [model uploads](https://docs.unsloth.ai/get-started/all-our-models) and [notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks).
# 

# # ### Installation
# 
# # In[ ]:
# 
# 
# get_ipython().run_cell_magic('capture', '', 'import os\n\n!pip install pip3-autoremove\n!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128\n!pip install unsloth\n!pip install transformers==4.56.2\n!pip install --no-deps trl==0.22.2\n')
# 
# 
# # In[ ]:
# 
# 
# get_ipython().run_cell_magic('capture', '', '# These are mamba kernels and we must have these for faster training\n!pip install --no-build-isolation mamba_ssm==2.2.5\n!pip install --no-build-isolation causal_conv1d==1.5.2\n')
# 
# 
# # ### Unsloth

# In[3]:


from unsloth import FastLanguageModel
import torch

fourbit_models = [
    "unsloth/granite-4.0-micro",
    "unsloth/granite-4.0-h-micro",
    "unsloth/granite-4.0-h-tiny",
    "unsloth/granite-4.0-h-small",

    # Base pretrained Granite 4 models
    "unsloth/granite-4.0-micro-base",
    "unsloth/granite-4.0-h-micro-base",
    "unsloth/granite-4.0-h-tiny-base",
    "unsloth/granite-4.0-h-small-base",

    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/Phi-4",
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit" # [NEW] We support TTS models!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/granite-4.0-350m-unsloth-bnb-4bit",
    max_seq_length = 2048,   # Choose any for long context!
    load_in_4bit = False,    # 4 bit quantization to reduce memory
    load_in_8bit = False,    # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
)


# We now add LoRA adapters so we only need to update a small amount of parameters!

# In[4]:


model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "shared_mlp.input_linear", "shared_mlp.output_linear"],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # rsLoRA: For correct scaling, set alpha ~ sqrt(r) if True.
    loftq_config = None,  # We support LoftQ
)


# <a name="Data"></a>
# ### Data Prep
# #### 📄 Using Google Sheets as Training Data
# Our goal is to create a customer support bot that proactively helps and solves issues.
# 
# We’re storing examples in a Google Sheet with two columns:
# 
# - **Snippet**: A short customer support interaction
# - **Recommendation**: A suggestion for how the agent should respond
# 
# This keeps things simple and collaborative. Anyone can edit the sheet, no database setup required.  
# <br>
# 
# ---
# <br>
# 
# #### 🔍 Why This Format?
# 
# This setup works well for tasks like:
# 
# - `Input snippet → Suggested reply`
# - `Prompt → Rewrite`
# - `Bug report → Diagnosis`
# - `Text → Label or Category`
# 
# Just collect examples in a spreadsheet, and you’ve got usable training data.  
# <br>
# 
# ---
# <br>
# 
# #### ✅ What You'll Learn
# 
# We’ll show how to:
# 
# 1. Load the Google Sheet into your notebook
# 2. Format it into a dataset
# 3. Use it to train or prompt an LLM
# 
# 
# The chat template for granite-4 look like this:
# ```
# <|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.
# Today's Date: June 24, 2025.
# You are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>
# 
# <|start_of_role|>user<|end_of_role|>How do astronomers determine the original wavelength of light emitted by a celestial body at rest, which is necessary for measuring its speed using the Doppler effect?<|end_of_text|>
# 
# <|start_of_role|>assistant<|end_of_role|>Astronomers make use of the unique spectral fingerprints of elements found in stars...<|end_of_text|>
# ```

# In[5]:


from datasets import load_dataset, Dataset

# Use the below shared sheet
# sheet_url = 'https://docs.google.com/spreadsheets/d/1NrjI5AGNIwRtKTAse5TW_hWq2CwAS03qCHif6vaaRh0/export?format=csv&gid=0'

# Or unsloth/Support-Bot-Recommendation
sheet_url = "https://huggingface.co/datasets/unsloth/Support-Bot-Recommendation/raw/main/support_recs.csv"

dataset = load_dataset(
    "csv",
    data_files={"train": sheet_url},
    column_names=["snippet", "recommendation"], # Replace with the actual column names of your sheet
    skiprows=1  # skip header rows
)["train"]


# We've just loaded the Google Sheet as a csv style Dataset, but we still need to format it into conversational style like below and then apply the chat template.
# 
# ```
# {"role": "system", "content": "You are an assistant"}
# {"role": "user", "content": "What is 2+2?"}
# {"role": "assistant", "content": "It's 4."}
# ```
# 
# We'll use a helper function `formatting_prompts_func` to do both!

# In[6]:


def formatting_prompts_func(examples):
    user_texts = examples['snippet']
    response_texts = examples['recommendation']
    messages = [
        [{"role": "user", "content": user_text},
        {"role": "assistant", "content": response_text}] for user_text, response_text in zip(user_texts, response_texts)
    ]
    texts = [tokenizer.apply_chat_template(message, tokenize = False, add_generation_prompt = False) for message in messages]

    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True,)


# We now look at the raw input data before formatting.

# In[7]:


dataset[5]["snippet"]


# In[8]:


dataset[5]['recommendation']


# And we see how the chat template transformed these conversations.

# In[9]:


dataset[5]["text"]


# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

# In[10]:


from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use TrackIO/WandB etc
    ),
)


# We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs. This helps increase accuracy of finetunes!

# In[11]:


from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_of_role|>user|end_of_role|>",
    response_part = "<|start_of_role|>assistant<|end_of_role|>",
)


# Let's verify masking the instruction part is done! Let's print the 100th row again.

# In[12]:


tokenizer.decode(trainer.train_dataset[100]["input_ids"])


# Now let's print the masked out example - you should see only the answer is present:

# In[13]:


tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")


# In[14]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`
# 
# ```
# Notice you might have to wait ~10 minutes for the Mamba kernels to compile! Please be patient!
# ```

# In[15]:


trainer_stats = trainer.train()


# In[16]:


# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# <a name="Inference"></a>
# ### Inference
# Let's run the model via Unsloth native inference! We'll use some example snippets not contained in our training data to get a sense of what was learned.

# In[17]:


# @title Test Scenarios
# --- Scenario 1: Video-Conferencing Screen-Share Bug (11 turns) ---
scenario_1 = """
User: Everyone in my meeting just sees a black screen when I share.
Agent: Sorry about that—are you sharing a window or your entire screen?
User: Entire screen on macOS Sonoma.
Agent: Thanks. Do you have “Enable hardware acceleration” toggled on in Settings → Video?
User: Yeah, that switch is on.
Agent: Could you try toggling it off and start a quick test share?
User: Did that—still black for attendees.
Agent: Understood. Are you on the desktop app v5.4.2 or the browser client?
User: Desktop v5.4.2—just updated this morning.
"""

# --- Scenario 2: Smart-Lock Low-Battery Loop (9 turns) ---
scenario_2 = """
User: I changed the batteries, but the lock app still says 5 % and won’t auto-lock.
Agent: Let’s check firmware. In the app, go to Settings → Device Info—what version shows?
User: 3.18.0-alpha.
Agent: Latest stable is 3.17.5. Did you enroll in the beta program?
User: I might have months ago.
Agent: Beta builds sometimes misreport battery. Remove one battery, wait ten seconds, reinsert, and watch the LED pattern.
User: LED blinks blue twice, then red once.
Agent: That blink code means “config mismatch.” Do you still have the old batteries handy?
User: Tossed them already.
"""

# --- Scenario 3: Accounting SaaS — Corrupted Invoice Export (10 turns) ---
scenario_3 = """
User: Every invoice I download today opens as a blank PDF.
Agent: Is this happening to historic invoices, new ones, or both?
User: Both. Anything I export is 0 bytes.
Agent: Are you exporting through “Bulk Actions” or individual invoice pages?
User: Individual pages.
Agent: Which browser/OS combo?
User: Chrome on Windows 11, latest update.
Agent: We released a new PDF renderer at 10 a.m. UTC. Could you try Edge quickly, just to rule out a caching quirk?
User: Tried Edge—same zero-byte file.
"""

# --- Scenario 4: Fitness-Tracker App — Stuck Step Count (8 turns) ---
scenario_4 = """
User: My step count has been frozen at 4,237 since last night.
Agent: Which phone are you syncing with?
User: iPhone 15, iOS 17.5.
Agent: In the Health Permissions screen, does “Motion & Fitness” show as ON?
User: Yes, it’s toggled on.
Agent: When you pull down to refresh the dashboard, does the sync spinner appear?
User: Spinner flashes for a second, then nothing changes.
"""

# --- Scenario 5: Online-Course Platform — Quiz Submission Error (12 turns) ---
scenario_5 = """
User: My quiz submits but then shows “Unknown grading error” and resets the answers.
Agent: Which course and quiz name?
User: History 301, Unit 2 Quiz.
Agent: Do you notice a red banner or any code like GR-### in the corner?
User: Banner says “GR-412”.
Agent: That code points to answer-payload size. Were you pasting images or long text into any answers?
User: Maybe a long essay—about 800 words in Question 5.
Agent: Are you on a laptop or mobile?
User: Laptop, Safari on macOS.
"""


# In[18]:


FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"role": "user", "content": scenario_1},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    padding = True,
    return_tensors = "pt",
    return_dict=True,
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = False)

_ = model.generate(**inputs,
                   streamer = text_streamer,
                   max_new_tokens = 512, # Increase if tokens are getting cut off
                   use_cache = True,
                   # Adjust the sampling params to your preference
                   do_sample=True,
                   temperature = 0.7, top_p = 0.8, top_k = 20,
)


# In[19]:


FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    {"role": "user", "content": scenario_2},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    padding = True,
    return_tensors = "pt",
    return_dict=True,
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = False)

_ = model.generate(**inputs,
                   streamer = text_streamer,
                   max_new_tokens = 512, # Increase if tokens are getting cut off
                   use_cache = True,
                   # Adjust the sampling params to your preference
                   do_sample=False,
                   temperature = 0.7, top_p = 0.8, top_k = 20,
)


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[20]:


model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving


# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# In[21]:


if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = True,
    )


# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly. Select `merged_16bit` for float16 or `merged_4bit` for int4. We also allow `lora` adapters as a fallback. Use `push_to_hub_merged` to upload to your Hugging Face account! You can go to https://huggingface.co/settings/tokens for your personal tokens.

# In[22]:


# Merge to 16bit
if False:
    model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: # Pushing to HF Hub
    model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# Merge to 4bit
if False:
    model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: # Pushing to HF Hub
    model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# Just LoRA adapters
if False:
    model.save_pretrained("model")
    tokenizer.save_pretrained("model")
if False: # Pushing to HF Hub
    model.push_to_hub("hf/model", token = "")
    tokenizer.push_to_hub("hf/model", token = "")


# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now! We clone `llama.cpp` and we default save it to `q8_0`. We allow all methods like `q4_k_m`. Use `save_pretrained_gguf` for local saving and `push_to_hub_gguf` for uploading to HF.
# 
# Some supported quant methods (full list on our [Wiki page](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)):
# * `q8_0` - Fast conversion. High resource use, but generally acceptable.
# * `q4_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K.
# * `q5_k_m` - Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K.
# 
# [**NEW**] To finetune and auto export to Ollama, try our [Ollama notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)

# Likewise, if you want to instead push to GGUF to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!

# In[23]:


# Save to 8bit Q8_0
if False:
    model.save_pretrained_gguf("model", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False:
    model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: # Pushing to HF Hub
    model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False:
    model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: # Pushing to HF Hub
    model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "hf/model", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "", # Get a token at https://huggingface.co/settings/tokens
    )


# Now, use the `model-unsloth.gguf` file or `model-unsloth-Q4_K_M.gguf` file in llama.cpp.
# 
# And we're done! If you have any questions on Unsloth, we have a [Discord](https://discord.gg/unsloth) channel! If you find any bugs or want to keep updated with the latest LLM stuff, or need help, join projects etc, feel free to join our Discord!
# 
# Some other links:
# 1. Train your own reasoning model - Llama GRPO notebook [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb)
# 2. Saving finetunes to Ollama. [Free notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-Ollama.ipynb)
# 3. Llama 3.2 Vision finetuning - Radiography use case. [Free Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb)
# 6. See notebooks for DPO, ORPO, Continued pretraining, conversational finetuning and more on our [documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)!
# 
# <div class="align-center">
#   <a href="https://unsloth.ai"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
#   <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord.png" width="145"></a>
#   <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a>
# 
#   Join Discord if you need help + ⭐️ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐️
# </div>
# 
#   This notebook and all Unsloth notebooks are licensed [LGPL-3.0](https://github.com/unslothai/notebooks?tab=LGPL-3.0-1-ov-file#readme).
# 
