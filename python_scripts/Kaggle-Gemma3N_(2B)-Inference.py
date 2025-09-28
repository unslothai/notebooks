#!/usr/bin/env python
# coding: utf-8

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://docs.unsloth.ai/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth on your own computer, follow the installation instructions on our Github page [here](https://docs.unsloth.ai/get-started/installing-+-updating).
# 
# You will learn how to do [data prep](#Data), how to [train](#Train), how to [run the model](#Inference), & [how to save it](#Save)
# 

# ### News

# 
# Unsloth now supports [gpt-oss RL](https://docs.unsloth.ai/new/gpt-oss-reinforcement-learning) with the fastest inference & lowest VRAM. Try our [new notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb) which automatically creates kernels!
# 
# [Vision RL](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl) is now supported! Train Qwen2.5-VL, Gemma 3 etc. with GSPO or GRPO.
# 
# Introducing Unsloth [Standby for RL](https://docs.unsloth.ai/basics/memory-efficient-rl): GRPO is now faster, uses 30% less memory with 2x longer context.
# 
# Unsloth now supports Text-to-Speech (TTS) models. Read our [guide here](https://docs.unsloth.ai/basics/text-to-speech-tts-fine-tuning).
# 
# Visit our docs for all our [model uploads](https://docs.unsloth.ai/get-started/all-our-models) and [notebooks](https://docs.unsloth.ai/get-started/unsloth-notebooks).
# 

# # ### Installation
# 
# # In[1]:
# 
# 
# get_ipython().run_cell_magic('capture', '', 'import os\n\n!pip install pip3-autoremove\n!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128\n!pip install unsloth\n!pip install transformers==4.55.4\n!pip install --no-deps trl==0.22.2\nimport torch; torch._dynamo.config.recompile_limit = 64;\n')
# 
# 
# # ### Unsloth

# ### Launch sglang inference for unsloth/gemma-3n-E2B-it (https://huggingface.co/unsloth/gemma-3n-E2B-it)

# In[2]:


# Load and run the model using sglang
get_ipython().system('nohup python -m sglang.launch_server --model-path unsloth/gemma-3n-E2B-it --attention-backend fa3 --port 8000 > sglang.log &')

# tail vllm logs. Check server has been started correctly
get_ipython().system('while ! grep -q "The server is fired up and ready to roll" sglang.log; do tail -n 1 sglang.log; sleep 5; done')


# ### Image helper functions

# In[3]:


from PIL.ImageFile import ImageFile
from PIL import Image
import numpy as np
import io
import base64
import requests
from io import BytesIO

def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

def process_image(image: ImageFile) -> str:
    """Process image for sglang gemma3n and return base64 string."""
    assert isinstance(image, ImageFile), "please pass an image object"

    # Resize the image
    resized_image = image.resize((384, 384))

    # Convert to numpy array and transpose
    image_array = np.array(resized_image)
    array_reordered = np.transpose(image_array, (1, 0, 2))

    # Convert back to PIL Image
    processed_image = Image.fromarray(array_reordered)

    # Convert to base64 string
    image_bytes = io.BytesIO()
    processed_image.save(image_bytes, format=image.format)
    base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")


    # Return data URL string
    format_name = image.format.lower() if image.format else 'png'
    return f"data:image/{format_name};base64,{base64_image}"


# ## Gemma3n Inference using sglang (source model: https://huggingface.co/unsloth/gemma-3n-E2B-it)
# 
# 

# ## Inference 1
# Image source file "https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/main/test/lang/example_image.png"

# load image from url source

# In[4]:


from IPython.display import display
image = load_image_from_url("https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/main/test/lang/example_image.png")
display(image)


# Let's run the model!

# In[6]:


import requests
from sglang.utils import wait_for_server, print_highlight, terminate_process

processed_image = process_image(image)
url = f"http://localhost:8000/v1/chat/completions"

processed_image = process_image(image)
data = {
    "model": "unsloth/gemma-3n-E2B-it",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": processed_image
                    },
                },
            ],
        }
    ],
    "max_tokens": 300,
}

response = requests.post(url, json=data)
print_highlight(response.text)


# ## Inference 2
# Image source file "https://i.ibb.co/1tw5whfz/ocr.png"

# load image from url source

# In[7]:


image = load_image_from_url("https://i.ibb.co/1tw5whfz/ocr.png")
display(image)


# Let's run the model!

# In[8]:


import requests

url = f"http://localhost:8000/v1/chat/completions"
processed_image = process_image(image)
data = {
    "model": "unsloth/gemma-3n-E2B-it",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Read the text in the image"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": processed_image
                    },
                },
            ],
        }
    ],
    "max_tokens": 300,
}

response = requests.post(url, json=data)
print_highlight(response.text)


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
