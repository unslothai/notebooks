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

# Introducing **Unsloth Studio** - a new open source, no-code web UI to train and run LLMs. [Blog](https://unsloth.ai/docs/new/studio) • [Notebook](https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb)
# 
# <table><tr>
# <td align="center"><a href="https://unsloth.ai/docs/new/studio"><img src="https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F~%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FxV1PO5DbF3ksB51nE2Tw%252Fmore%2520cropped%2520ui%2520for%2520homepage.png%3Falt%3Dmedia%26token%3Df75942c9-3d8d-4b59-8ba2-1a4a38de1b86&width=376&dpr=3&quality=100&sign=a663c397&sv=2" width="200" height="120" alt="Unsloth Studio Training UI"></a><br><sub><b>Train models</b> — no code needed</sub></td>
# <td align="center"><a href="https://unsloth.ai/docs/new/studio"><img src="https://unsloth.ai/docs/~gitbook/image?url=https%3A%2F%2F3215535692-files.gitbook.io%2F~%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FxhOjnexMCB3dmuQFQ2Zq%252Fuploads%252FRCnTAZ6Uh88DIlU3g0Ij%252Fmainpage%2520unsloth.png%3Falt%3Dmedia%26token%3D837c96b6-bd09-4e81-bc76-fa50421e9bfb&width=376&dpr=3&quality=100&sign=c1a39da1&sv=2" width="200" height="120" alt="Unsloth Studio Chat UI"></a><br><sub><b>Run GGUF models</b> on Mac, Windows & Linux</sub></td>
# </tr></table>
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
# get_ipython().run_cell_magic('capture', '', 'import os, re\nif "COLAB_" not in "".join(os.environ.keys()):\n    !pip install unsloth  # Do this in local & cloud setups\nelse:\n    import torch; v = re.match(r\'[\\d]{1,}\\.[\\d]{1,}\', str(torch.__version__)).group(0)\n    xformers = \'xformers==\' + {\'2.10\':\'0.0.34\',\'2.9\':\'0.0.33.post1\',\'2.8\':\'0.0.32.post2\'}.get(v, "0.0.34")\n    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer\n    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth\n!pip install --no-deps transformers==5.5.0\n!pip install torchcodec\nimport torch; torch._dynamo.config.recompile_limit = 64;\n')
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
from huggingface_hub import snapshot_download

fourbit_models = [
    # Gemma 4 models
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E2B",
    "unsloth/gemma-4-E2B-it",
    "unsloth/gemma-4-E4B",
    "unsloth/gemma-4-31B-it",
    "unsloth/gemma-4-31B",
    "unsloth/gemma-4-26B-A4B-it",
    "unsloth/gemma-4-26B-A4B",
] # More models at https://huggingface.co/unsloth

model, processor = FastModel.from_pretrained(
    model_name = "unsloth/gemma-4-E2B-it",
    dtype = None, # None for auto detection
    max_seq_length = 8192, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)


# # Gemma 4 can process Text, Vision and Audio!
# 
# Let's first experience how Gemma 4 can handle multimodal inputs. We use Gemma 4's recommended settings of `temperature = 1.0, top_p = 0.95, top_k = 64` but for this example we use `do_sample=False` for ASR.

# In[4]:


from transformers import TextStreamer
# Helper function for inference
def do_gemma_4_inference(messages, max_new_tokens = 128):
    _ = model.generate(
        **processor.apply_chat_template(
            messages,
            add_generation_prompt = True, # Must add for generation
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
        ).to("cuda"),
        max_new_tokens = max_new_tokens,
        do_sample = False,
        streamer = TextStreamer(processor, skip_prompt = True),
    )


# <h3>Let's Evaluate Gemma 4 Baseline Performance on German Transcription</h2>

# In[5]:


from datasets import load_dataset,Audio,concatenate_datasets

dataset = load_dataset("kadirnar/Emilia-DE-B000000", split = "train")

# Select a single audio sample to reserve for testing.
# This index is chosen from the full dataset before we create the smaller training split.
test_audio = dataset[7546]

dataset = dataset.select(range(3000))

dataset = dataset.cast_column("audio", Audio(sampling_rate = 16000))


# In[6]:


from IPython.display import Audio, display
print(test_audio['text'])
Audio(test_audio['audio']['array'],rate = test_audio['audio']['sampling_rate'])


# And the translation of the audio from German to English is:
# 
# > I—I hold myself directly accountable. That much is, of course, clear: namely, that there are political interests involved in trade—in the exchange of goods—and that political influences are at play. The question is: that should not be the alternative.

# In[7]:


messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an assistant that transcribes speech accurately.",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": test_audio['audio']['array']},
            {"type": "text", "text": "Please transcribe this audio."}
        ]
    }
]

do_gemma_4_inference(messages, max_new_tokens = 256)


# <h3>Baseline Model Performance: 32.43% Word Error Rate (WER) for this sample !</h3>

# # Let's finetune Gemma 4!
# 
# You can finetune the vision and text and audio parts

# We now add LoRA adapters so we only need to update a small amount of parameters!

# In[8]:


model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # False if not finetuning vision layers
    finetune_language_layers   = True,  # False if not finetuning language layers
    finetune_attention_modules = True,  # False if not finetuning attention layers
    finetune_mlp_modules       = True,  # False if not finetuning MLP layers

    r = 8,                              # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,                    # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,                 # We support rank stabilized LoRA
    loftq_config = None,                # And LoftQ
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",

        # Audio layers
        "post", "linear_start", "linear_end",
        "embedding_projection",
        "ffw_layer_1", "ffw_layer_2",
        "output_proj",
    ]
)


# <a name="Data"></a>
# ### Data Prep
# We adapt the `kadirnar/Emilia-DE-B000000` dataset for our German ASR task using Gemma 4 multi-modal chat format. Each audio-text pair is structured into a conversation with `system`, `user`, and `assistant` roles. The processor then converts this into the final training format:
# 
# ```
# <bos><|turn>system
# You are an assistant that transcribes speech accurately.<turn|>
# <|turn>user
# <|audio|>Please transcribe this audio.<turn|>
# <|turn>model
# Ich, ich rechne direkt mich an.<turn|>

# In[9]:


def format_intersection_data(samples: dict) -> dict[str, list]:
    """Format intersection dataset to match expected message format"""
    formatted_samples = {"messages": []}
    for idx in range(len(samples["audio"])):
        audio = samples["audio"][idx]["array"]
        label = str(samples["text"][idx])

        message = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an assistant that transcribes speech accurately.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": "Please transcribe this audio."}
                ]
            },
            {
                "role": "assistant",
                "content":[{"type": "text", "text": label}]
            }
        ]
        formatted_samples["messages"].append(message)
    return formatted_samples


# In[10]:


dataset = dataset.map(format_intersection_data, batched = True, batch_size = 4, num_proc = 4)


# <a name="Train"></a>
# ### Train the model
# Now let's train our model. We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.

# In[11]:


# Use UnslothVisionDataCollator which handles audio token alignment correctly
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    processing_class = processor.tokenizer,
    data_collator = UnslothVisionDataCollator(model, processor),
    args = SFTConfig(
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 1,
        warmup_ratio = 0.03,
        # num_train_epochs = 1, # Use for full training runs
        max_steps = 60,
        learning_rate = 5e-5,
        logging_steps = 1,
        save_strategy = "steps",
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
        remove_unused_columns = False,

        # The below are a must for audio finetuning:
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 8192,
    )
)


# In[12]:


# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")


# # Let's train the model!
# 
# To resume a training run, set `trainer.train(resume_from_checkpoint = True)`

# In[13]:


trainer_stats = trainer.train()


# In[14]:


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
# Let's run the model via Unsloth native inference! According to the `Gemma-4` team, the recommended settings for inference are `temperature = 1.0, top_p = 0.95, top_k = 64` but for this example we use `do_sample=False` for ASR.

# In[15]:


messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an assistant that transcribes speech accurately.",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": test_audio['audio']['array']},
            {"type": "text", "text": "Please transcribe this audio."}
        ]
    }
]

do_gemma_4_inference(messages, max_new_tokens = 256)


# <a name="Save"></a>
# ### Saving, loading finetuned models
# To save the final model as LoRA adapters, either use Hugging Face's `push_to_hub` for an online save or `save_pretrained` for a local save.
# 
# **[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!

# In[16]:


model.save_pretrained("gemma_4_lora")  # Local saving
processor.save_pretrained("gemma_4_lora")
# model.push_to_hub("HF_ACCOUNT/gemma_4_lora", token = "YOUR_HF_TOKEN") # Online saving
# processor.push_to_hub("HF_ACCOUNT/gemma_4_lora", token = "YOUR_HF_TOKEN") # Online saving


# Now if you want to load the LoRA adapters we just saved for inference, set `False` to `True`:

# In[17]:


if False:
    from unsloth import FastModel
    model, processor = FastModel.from_pretrained(
        model_name = "gemma_4_lora", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = 2048,
        load_in_4bit = True,
    )

messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "What is Gemma-4?",}]
}]
inputs = processor.apply_chat_template(
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
    streamer = TextStreamer(processor, skip_prompt = True),
)


# ### Saving to float16 for VLLM
# 
# We also support saving to `float16` directly for deployment! We save it in the folder `gemma-4-finetune`. Set `if False` to `if True` to let it run!

# In[18]:


if False: # Change to True to save finetune!
    model.save_pretrained_merged("gemma-4", processor)


# If you want to upload / push to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!

# In[19]:


if False: # Change to True to upload finetune
    model.push_to_hub_merged(
        "HF_ACCOUNT/gemma-4-finetune", processor,
        token = "YOUR_HF_TOKEN"
    )


# ### GGUF / llama.cpp Conversion
# To save to `GGUF` / `llama.cpp`, we support it natively now for all models! For now, you can convert easily to `Q8_0, F16 or BF16` precision. `Q4_K_M` for 4bit will come later!

# In[20]:


if False: # Change to True to save to GGUF
    model.save_pretrained_gguf(
        "gemma_4_finetune",
        processor,
        quantization_method = "Q8_0", # For now only Q8_0, BF16, F16 supported
    )


# Likewise, if you want to instead push to GGUF to your Hugging Face account, set `if False` to `if True` and add your Hugging Face token and upload location!

# In[21]:


if False: # Change to True to upload GGUF
    model.push_to_hub_gguf(
        "HF_ACCOUNT/gemma_4_finetune",
        processor,
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
