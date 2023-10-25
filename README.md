# PyChatGPT: Python Module
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/johndef64/pychatgpt/blob/main/pychatgpt_trial.ipynb) 

`pychatgpt` is a mall and useful Python module that provides functions for interacting with OpenAI's GPT models to create conversational agents. This module allows users to have interactive conversations with the GPT models and keeps track of the conversation history in your Python Projects and Jupyter Notebooks.



## Installation
In a Conda envoiroment, add `pychatgpt.py` to the folder `C:\Users\\*your_user_name*\\.conda\envs\\*your_env*\\Lib\site-packages`; else simply add it in your working directory.

To use this module, you need to have an **OpenAI API key**. You have to provide your API key when requested once and it will be stored automatically in a file called `openai_api_key.txt` in your working directory.

## Usage
`import pychatgpt as op`

The module provides the following main functions:

1. `op.ask_gpt(prompt, *parameters*)`:  
This function takes a prompt as input and generates a response from the GPT chosen model. It returns the generated response and logs the conversation in the `conversation_log.txt` file.
You can simply use `op.ask_gpt(prompt)` and keep the default parameters.

2. `op.send_message(message,*parameters*)`:  
This function allows for a more interactive conversation with the GPTchosen model. It takes a message as input, generates a response from the model, and updates the conversation history. It also logs the conversation in the `conversation_log.txt` file.

Use `op.send_message(message)` keeping the default *parameters* or change them as function operators:

        op.send_message(message,*parameters*)
        - model= 'gpt-3.5-turbo-16k'
        - maxtoken = 800,
        - temperature = 1,
        - system='',
        - persona='',
        - printreply = True,
        - printuser = False,
        - printtoken = True



The module also provides additional utility functions for managing the conversation, such as clearing the chat history, setting a persona, and setting system instructions, save/load conversations.

3. `choose_model()`
4. `clear_chat()`
5. `expand_conversation()`
6. `save_conversation()`
7. `load_conversation()`
8. `load_file()`

To set-up multiple conversations or change the API-key, follow the example proposed in [pychatgpt_trial.ipynb](https://github.com/johndef64/pychatgpt/blob/main/pychatgpt_trial.ipynb)

## Notes
The code in this module assumes that the conversation history is stored in a global variable named `conversation_gpt`. Use `print(op.conversation_gpt)` to show conversation history and `op.conversation_gpt.pop()` to remove last interacition. `op.send_message('clearchat')` to start a new conversation.

Using `op.send_message`, the code checks if the total number of tokens exceeds the model's maximum context length (gpt 3.5 turbo-16k: 16,384 tokens). If it does, a warning message indicates that the token limit is being reached and then then the first part of the conversation will automatically be deleted to make room for the next interaction.

## Openai-based applications
Some other python applications executable in Terminal that take advantage of openai modulo features
- auto-gpt
- gpt-cli
- rec-whisper-paste

## Author
Written by: JohnDef64 

## Acknowledgment
This module only facilitates interaction with the OpenAI API and keeps track of it. OpenAI holds all rights to ChatGPT.
