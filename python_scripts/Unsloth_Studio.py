#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# To run this, press "*Runtime*" and press "*Run all*" on a **free** Tesla T4 Google Colab instance!
# <div class="align-center">
# <a href="https://unsloth.ai/"><img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="115"></a>
# <a href="https://discord.gg/unsloth"><img src="https://github.com/unslothai/unsloth/raw/main/images/Discord button.png" width="145"></a>
# <a href="https://unsloth.ai/docs/"><img src="https://github.com/unslothai/unsloth/blob/main/images/documentation%20green%20button.png?raw=true" width="125"></a> Join Discord if you need help + ⭐ <i>Star us on <a href="https://github.com/unslothai/unsloth">Github</a> </i> ⭐
# </div>
# 
# To install Unsloth Studio on your local device, follow [our guide](https://unsloth.ai/docs/new/unsloth-studio/install). Unsloth Studio is licensed [AGPL-3.0](https://github.com/unslothai/unsloth/blob/main/studio/LICENSE.AGPL-3.0).
# 
# ### Unsloth Studio
# 
# Train and run open models with [**Unsloth Studio**](https://unsloth.ai/docs/new/unsloth-studio/start). NEW! Installation should now only take 2 mins!
# 
# 
# We are actively working on making Unsloth Studio install on Colab T4 GPUs faster.
# 
# [Features](https://unsloth.ai/docs/new/unsloth-studio#features) • [Quickstart](https://unsloth.ai/docs/new/unsloth-studio/start) • [Data Recipes](https://unsloth.ai/docs/new/unsloth-studio/data-recipe) • [Studio Chat](https://unsloth.ai/docs/new/unsloth-studio/chat) • [Export](https://unsloth.ai/docs/new/unsloth-studio/export)

# <p align="left"><img src="https://github.com/unslothai/unsloth/raw/main/studio/frontend/public/studio%20github%20landscape%20colab%20display.png" width="600"></p>

# ### Setup: Clone repo and run setup

# In[ ]:


get_ipython().system('git clone --depth 1 --branch main https://github.com/unslothai/unsloth.git')
get_ipython().run_line_magic('cd', '/content/unsloth')
get_ipython().system('chmod +x studio/setup.sh && ./studio/setup.sh --local')


# ### Start Unsloth Studio

# In[ ]:


import sys
sys.path.insert(0, "/content/unsloth/studio/backend")
from colab import start
start()


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
#   <b>This notebook is licensed <a href="https://github.com/unslothai/unsloth/blob/main/studio/LICENSE.AGPL-3.0">AGPL-3.0</a></b>
# </div>
