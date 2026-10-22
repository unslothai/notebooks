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

# ### News

# Introducing **[Unsloth Desktop](https://unsloth.ai/docs/desktop)**, the first desktop app to run and train models. Free and open-source for macOS, Windows and Linux. [GitHub](https://github.com/unslothai/unsloth) • [Download](https://unsloth.ai/download)
# 
# <p>
# <a href="https://unsloth.ai/docs/desktop"><img src="https://raw.githubusercontent.com/unslothai/notebooks/refs/heads/main/assets/unsloth-qwen3-8.png" width="350" alt="Introducing Unsloth Desktop"></a>
# </p>
# 
# Train MoEs - DeepSeek, GLM, Qwen and gpt-oss 12x faster with 35% less VRAM. [Blog](https://unsloth.ai/docs/new/faster-moe)
# 
# Ultra Long-Context Reinforcement Learning is here with 7x more context windows! [Blog](https://unsloth.ai/docs/new/grpo-long-context)
# 
# New in Reinforcement Learning: [FP8 RL](https://unsloth.ai/docs/new/fp8-reinforcement-learning) • [Vision RL](https://unsloth.ai/docs/new/vision-reinforcement-learning-vlm-rl) • [Standby](https://unsloth.ai/docs/basics/memory-efficient-rl) • [gpt-oss RL](https://unsloth.ai/docs/new/gpt-oss-reinforcement-learning)
# 
# Visit our docs for all our [model uploads](https://unsloth.ai/docs/get-started/unsloth-model-catalog) and [notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks).

# # ### Installation
# 
# # In[1]:
# 
# 
# get_ipython().run_cell_magic('capture', '', 'import os, re\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth  # Do this in local & cloud setups\nelse:\n    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer\n    !pip install --no-deps unsloth_zoo bitsandbytes accelerate peft trl triton unsloth\n    !unsloth install-kernels\n    !pip install --no-deps --upgrade "torchao>=0.16.0"\n!pip install --no-deps transformers==5.5.0 "tokenizers>=0.22.0,<=0.23.0"\n!pip install "huggingface_hub>=1.5.0,<2.0"\nimport importlib.metadata as _md\ntry: _tv = tuple(int(x) for x in _md.version("torch").split("+")[0].split(".")[:2])\nexcept Exception: _tv = (2, 11)\n_torchcodec = {(2, 11): "torchcodec>=0.11.0", (2, 10): "torchcodec==0.10.*", (2, 9): "torchcodec==0.9.*", (2, 8): "torchcodec==0.7.*", (2, 7): "torchcodec==0.5.*"}.get(_tv, "torchcodec>=0.12.0" if _tv >= (2, 12) else "torchcodec")\n!pip install "{_torchcodec}"\nimport torch; torch._dynamo.config.recompile_limit = 64;\n')
# 
# 
# # In[2]:
# 
# 
# get_ipython().run_cell_magic('capture', '', '!pip install --no-deps --upgrade timm # For Gemma 4 vision/audio\n')
# 
# 
# # ### Unsloth
# 
# `FastModel` supports loading nearly any model now! This includes Vision and Text models!

# In[3]:


from unsloth import FastModel
import torch

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

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-4-E2B-it",
    dtype = None, # None for auto detection
    max_seq_length = 1024, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)


# # Gemma 4 can process Text, Vision and Audio!
# 
# Let's first experience how Gemma 4 can handle multimodal inputs. We use Gemma 4's recommended settings of `temperature = 1.0, top_p = 0.95, top_k = 64`

# In[4]:


from transformers import TextStreamer
# Helper function for inference
def do_gemma_4_inference(messages, max_new_tokens = 128):
    _ = model.generate(
        **tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True, # Must add for generation
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
        ).to("cuda"),
        max_new_tokens = max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 64,
        streamer = TextStreamer(tokenizer, skip_prompt = True)
    )


# # Gemma 4 can see images!
# 
# <img src="https://files.worldwildlife.org/wwfcmsprod/images/Sloth_Sitting_iStock_3_12_2014/story_full_width/8l7pbjmj29_iStock_000011145477Large_mini__1_.jpg" alt="Alt text" height="256">

# In[5]:


sloth_link = "https://files.worldwildlife.org/wwfcmsprod/images/Sloth_Sitting_iStock_3_12_2014/story_full_width/8l7pbjmj29_iStock_000011145477Large_mini__1_.jpg"

messages = [{
    "role" : "user",
    "content": [
        { "type": "image", "image" : sloth_link },
        { "type": "text",  "text" : "Which films does this animal feature in?" }
    ]
}]
# You might have to wait 1 minute for Unsloth's auto compiler
do_gemma_4_inference(messages, max_new_tokens = 256)


# Let's make a poem about sloths!

# In[6]:


messages = [{
    "role": "user",
    "content": [{ "type" : "text",
                  "text" : "Write a poem about sloths." }]
}]
do_gemma_4_inference(messages)


# # Gemma 4 can also hear!

# In[7]:


from IPython.display import Audio, display
Audio("https://www.nasa.gov/wp-content/uploads/2015/01/591240main_JFKmoonspeech.mp3")


# In[8]:


get_ipython().system('wget -qqq https://www.nasa.gov/wp-content/uploads/2015/01/591240main_JFKmoonspeech.mp3 -O audio.mp3')


# In[9]:


audio_file = "audio.mp3"

messages = [{
    "role" : "user",
    "content": [
        { "type": "audio", "audio" : audio_file },
        { "type": "text",  "text" : "What is this audio about?" }
    ]
}]
do_gemma_4_inference(messages, max_new_tokens = 256)


# # Let's combine all 3 modalities together!

# In[10]:


messages = [{
    "role" : "user",
    "content": [
        { "type": "audio", "audio" : audio_file },
        { "type": "image", "image" : sloth_link },
        { "type": "text",  "text" : "What is this audio and image about? "\
                                    "How are they related?" }
    ]
}]
do_gemma_4_inference(messages, max_new_tokens = 256)


# # Let's finetune Gemma 4!
# 
# You can finetune the vision and text parts for now through selection - the audio part can also be finetuned - we're working to make it selectable as well!

# We now add LoRA adapters so we only need to update a small amount of parameters!

# In[11]:


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # Should leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)


# <a name="Data"></a>
# ### Data Prep
# We now use the `Gemma-4` format for conversation style finetunes. We use [Maxime Labonne's FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset in ShareGPT style. Gemma-4 renders multi turn conversations like below:
# 
# ```
# <bos><|turn>user
# Hello<turn|>
# <|turn>model
# Hey there!<turn|>
# ```
# We use our `get_chat_template` function to get the correct chat template. We support `zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3, phi4, qwen2.5, gemma3, gemma-4` and more.

# In[12]:


from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-4",
)


# We get the first 3000 rows of the dataset

# In[13]:


from datasets import load_dataset
dataset = load_dataset("mlabonne/FineTome-100k", split = "train[:3000]")


# We now use `standardize_data_formats` to try converting datasets to the correct format for finetuning purposes!

# In[14]:


from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)


# Let's see how row 100 looks like!

# In[15]:


dataset[100]


# We now have to apply the chat template for `Gemma-4` onto the conversations, and save it to `text`. We remove the `<bos>` token using removeprefix(`'<bos>'`) since we're finetuning. The Processor will add this token before training and the model expects only one.

# In[16]:


def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
   return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)


# Let's see how the chat template did! Notice there is no `<bos>` token as the processor tokenizer will be adding one.

# In[17]:


dataset[100]["text"]


# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

# In[18]:


from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
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


# We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs. This helps increase accuracy of finetunes! Unsloth now auto-detects the instruction and response parts from the tokenizer's chat template, so we don't need to pass `instruction_part` and `response_part` anymore. You can still pass them explicitly if you use a custom chat template.

# In[19]:


from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(trainer)


# Let's verify masking the instruction part is done! Let's print the 100th row again.  Notice how the sample only has a single `<bos>` as expected!

# In[20]:


tokenizer.decode(trainer.train_dataset[100]["input_ids"])


# Now let's print the masked out example - you should see only the answer is present:

# In[21]:


tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")


# In[22]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# # Let's train the model!
# 
# To resume a training run, set `trainer.train(resume_from_checkpoint = True)`

# In[23]:


trainer_stats = trainer.train()


# In[24]:


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
# Let's run the model via Unsloth native inference! According to the `Gemma-4` team, the recommended settings for inference are `temperature = 1.0, top_p = 0.95, top_k = 64`

# In[25]:


from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-4",
)
messages = [{
    "role": "user",
    "content": [{
        "type" : "text",
        "text" : "Continue the sequence: 1, 1, 2, 3, 5, 8,",
    }]
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")
outputs = model.generate(
    **inputs,
    max_new_tokens = 64, # Increase for longer outputs!
    # Recommended Gemma-4 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
)
tokenizer.batch_decode(outputs)


#  You can also use a `TextStreamer` for continuous inference - so you can see the generation token by token, instead of waiting the whole time!

# In[26]:


messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Why is the sky blue?",}]
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")

from transformers import TextStreamer
_ = model.generate(
    **inputs,
    max_new_tokens = 64, # Increase for longer outputs!
    # Recommended Gemma-4 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[27]:


model.save_pretrained("gemma_4_lora")  # Local saving
tokenizer.save_pretrained("gemma_4_lora")
# model.push_to_hub("HF_ACCOUNT/gemma_4_lora", token = "YOUR_HF_TOKEN") # Online saving
# tokenizer.push_to_hub("HF_ACCOUNT/gemma_4_lora", token = "YOUR_HF_TOKEN") # Online saving


# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# In[28]:


if False:
    from unsloth import FastModel
    model, tokenizer = FastModel.from_pretrained(
        model_name = "gemma_4_lora", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = True,
    )

messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "What is Gemma-4?",}]
}]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
    return_dict = True,
).to("cuda")

from transformers import TextStreamer
_ = model.generate(
    **inputs,
    max_new_tokens = 128, # Increase for longer outputs!
    # Recommended Gemma-4 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)


# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly for deployment! We save it in the folder `gemma-4-finetune`. Set `if False` to `if True` to let it run!

# In[29]:


if False: # Change to True to save finetune!
    model.save_pretrained_merged("gemma-4-finetune", tokenizer)


# If you want to upload / push to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!

# In[30]:


if False: # Change to True to upload finetune
    model.push_to_hub_merged(
        "HF_ACCOUNT/gemma-4-finetune", tokenizer,
        token = "YOUR_HF_TOKEN"
    )


# ### LiteRT-LM: run the finetune on a phone
# 
# [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runs `.litertlm` files on Android phones and other devices. The cells below turn the merged 16-bit model from `save_pretrained_merged` (above) into `gemma-4-finetune.litertlm`: text only, int4 weights, a 2048-token context, about 2.6 GB.
# 
# They run on the CPU in a separate Python environment, so the training packages stay as they are. The export needs about **30 GB of RAM**, so run these cells on a runtime with at least 30 GB (a Kaggle "GPU T4 x2" session has 30 GiB). They only read the `gemma-4-finetune` folder: you can restart the runtime after saving it, to free the memory training used.
# 
# Set `export_litertlm = True` to run them.

# In[ ]:


export_litertlm = False # Change to True to make a .litertlm file
model_dir     = "gemma-4-finetune"          # The merged 16-bit model (save_pretrained_merged)
litertlm_file = "gemma-4-finetune.litertlm" # The file for LiteRT-LM
work_dir      = "/tmp/litertlm"             # The tools, their Python environment and the intermediate file

if export_litertlm:
    import os, subprocess, sys

    def run(command, show = True, env = None):
        # Run a command, print its output as it comes (show = True), return (exit code, its last lines)
        process = subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True, env = env)
        tail = []
        for line in process.stdout:
            if show: print(line, end = "")
            tail = (tail + [line])[-40:]
        return process.wait(), "".join(tail)

    def ram_gib():
        # The machine's memory, or this container's memory limit when that is lower
        with open("/proc/meminfo") as f:
            ram = int(next(line for line in f if line.startswith("MemTotal:")).split()[1]) * 1024
        for limit in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            if os.path.isfile(limit):
                value = open(limit).read().strip()
                if value.isdigit(): ram = min(ram, int(value))
        return ram / 2**30

    if ram_gib() < 20:
        raise RuntimeError(f"This runtime has {ram_gib():.1f} GiB of RAM. The export needs about 30 GB: "
                           "use a runtime with more RAM.")
    if ram_gib() < 29:
        print(f"This runtime has {ram_gib():.1f} GiB of RAM. The export peaked at about 28 GiB in our test, "
              "so it may run out of memory here.")
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        if "model" not in globals():
            raise RuntimeError(f"There is no merged model in {model_dir}. Train the model above, then run this "
                               "cell again: it saves the model with save_pretrained_merged.")
        model.save_pretrained_merged(model_dir, tokenizer)
    os.makedirs(work_dir, exist_ok = True)


# We fetch the export tools at a fixed commit, check them against their sha256, and install the versions they were tested with (litert-torch, LiteRT-LM and a CPU build of torch) into their own environment in `work_dir`. This takes a few minutes the first time.

# In[ ]:


if export_litertlm:
    import hashlib, urllib.request
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        raise RuntimeError(f"The export environment needs Python 3.12 or 3.13 (its pinned packages have wheels for "
                           f"those); this runtime has Python {sys.version.split()[0]}.")
    kit_url = "https://raw.githubusercontent.com/john-rocky/hf-to-litertlm/836ef8f6a9964f4136ac4773a8981b20223a256e/notebooks/unsloth_gemma4_e2b/" # The export tools at a fixed commit
    kit = {
        "requirements.txt"      : "1098d968b54a947060347b6830fbb5d693910f50255e32679e620ed55738ac55",
        "gemma4_to_litertlm.py" : "8f80f48e0ab726080b1cd7e9c93bdd2fa3a9f321761353e02d92c7e6ac318567",
        "bf16to32.py"           : "36a9e861eb290337fff8caf8f5ea42a35748f908731fdafb7d30eaa32cda0882",
    }
    kit_dir = os.path.join(work_dir, "kit")
    os.makedirs(kit_dir, exist_ok = True)
    file_sha256 = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
    for name, digest in kit.items():
        path = os.path.join(kit_dir, name)
        if not os.path.isfile(path) or file_sha256(path) != digest:
            urllib.request.urlretrieve(kit_url + name, path)
        if file_sha256(path) != digest:
            raise RuntimeError(f"{kit_url + name} does not have the sha256 this notebook expects.")
    tool = os.path.join(kit_dir, "gemma4_to_litertlm.py")
    requirements = os.path.join(kit_dir, "requirements.txt")

    # A separate environment, so the torch and transformers of the training stay as they are
    python = os.path.join(work_dir, "venv", "bin", "python")
    if not os.path.isfile(python):
        code, tail = run([sys.executable, "-m", "venv", "--without-pip", os.path.join(work_dir, "venv")], show = False)
        if code != 0:
            raise RuntimeError(f"Could not make a Python environment in {work_dir}:\n{tail}")
    if run([python, tool, "versions", requirements], show = False)[0] != 0:
        torch_line = next(line.strip() for line in open(requirements) if line.startswith("torch=="))
        pip = [sys.executable, "-m", "pip", "--python", python, "install", "--only-binary=:all:"]
        for args in (["--index-url", "https://download.pytorch.org/whl/cpu", torch_line], ["-r", requirements]):
            code, tail = run(pip + args, show = False)
            if code != 0:
                raise RuntimeError(f"pip could not install the export environment:\n{tail}")
    if run([python, tool, "versions", requirements])[0] != 0:
        raise RuntimeError("The export environment does not have the versions requirements.txt pins (see above).")


# Now the export itself. It first checks that LiteRT-LM renders the model's chat template as transformers does, and edits the template where it does not. Then litert-torch reads the merged model in bfloat16 and writes int4 weights, prefill signatures for 1024 and 128 tokens and a 2048-token KV cache. It takes about 13 minutes on Kaggle's 4 CPUs and most of the RAM.

# In[ ]:


if export_litertlm:
    import resource, time
    bf16_file = os.path.join(work_dir, "model_bf16.litertlm")
    env = dict(os.environ, PYTHONHASHSEED = "0", HF_HUB_OFFLINE = "1", TMPDIR = os.path.join(work_dir, "tmp"))
    os.makedirs(env["TMPDIR"], exist_ok = True)
    start = time.time()
    code, _ = run([python, tool, "export", model_dir, bf16_file, "--work", work_dir], env = env)
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20 # GiB (Linux reports KiB)
    if code == -9:
        raise RuntimeError(f"The export was killed after {time.time() - start:.0f} s at {peak:.1f} GiB, most likely "
                           f"because it ran out of memory ({ram_gib():.1f} GiB here). It needs about 30 GB of RAM.")
    if code != 0:
        raise RuntimeError(f"The export failed with exit code {code} (see the lines above).")
    print(f"Exported in {time.time() - start:.0f} s, peak memory {peak:.1f} GiB")


# Reading the model in bfloat16 leaves some BFLOAT16 tensors in the graph (the KV cache among them), and LiteRT-LM's GPU path does not take those. This step rewrites them as FLOAT32 and checks the result against the export. It takes about half a minute.

# In[ ]:


if export_litertlm:
    code, _ = run([python, tool, "fix", bf16_file, litertlm_file])
    if code != 0:
        raise RuntimeError(f"Rewriting {bf16_file} failed with exit code {code} (see the lines above).")


# Let's load `gemma-4-finetune.litertlm` with LiteRT-LM on the CPU here and ask it one question, to see that it answers before you copy it to a phone.

# In[ ]:


if export_litertlm:
    code, _ = run([python, tool, "ask", litertlm_file, "What is 17 + 25?"])
    if code != 0:
        raise RuntimeError(f"LiteRT-LM could not run {litertlm_file} (see the lines above).")


# Download `gemma-4-finetune.litertlm`, or set `if False` to `if True` below to upload it to your Hugging Face account. Then run it on the phone with LiteRT-LM: for example load it in the [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) app, which can load your own models, or from your own app with LiteRT-LM's [Kotlin API](https://ai.google.dev/edge/litert-lm/android).

# In[ ]:


if False: # Change to True to upload the .litertlm file
    from huggingface_hub import HfApi
    api = HfApi(token = "YOUR_HF_TOKEN")
    api.create_repo("HF_ACCOUNT/gemma-4-finetune-litertlm", exist_ok = True)
    api.upload_file(
        path_or_fileobj = litertlm_file,
        path_in_repo = litertlm_file,
        repo_id = "HF_ACCOUNT/gemma-4-finetune-litertlm",
    )


# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now for all models! For now, you can convert easily to `Q8_0, F16 or BF16` precision. `Q4_K_M` for 4bit will come later!

# In[31]:


if False: # Change to True to save to GGUF
    model.save_pretrained_gguf(
        "gemma_4_finetune",
        tokenizer,
        quantization_method = "Q8_0", # For now only Q8_0, BF16, F16 supported
    )


# Likewise, if you want to instead push to GGUF to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!

# In[32]:


if False: # Change to True to upload GGUF
    model.push_to_hub_gguf(
        "HF_ACCOUNT/gemma_4_finetune",
        tokenizer,
        quantization_method = "Q8_0", # Only Q8_0, BF16, F16 supported
        token = "YOUR_HF_TOKEN",
    )


# Now, use the `gemma-4-finetune.gguf` file or `gemma-4-finetune-Q4_K_M.gguf` file in llama.cpp.
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
