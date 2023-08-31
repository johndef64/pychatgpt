# PyChatGPT Python Module

`pychatgpt` is small and useful Python module that provides functions for interacting with OpenAI's GPT-3.5 Turbo model to create conversational agents. This module allows users to have interactive conversations with the GPT-3.5 Turbo model and keeps track of the conversation history in your Python Projects or working on a Jupyer Notebook.

## Installation
In a Conda envoiroment, add `pychatgpt.py` to the folder `C:\Users\\*your_user_name*\\.conda\envs\\*your_env*\\Lib\site-packages`; else simply add it in `/lib/site-packages` in your Python folder.

To use this module, you need to have an OpenAI API key. You have to provide your API key when requested once and it will be stored automatically in a file called `openai_api_key.txt` in your working directory.

## Usage
`import pychatgpt as op`

The module provides the following main functions:

1. `op.ask_gpt(prompt, model = "gpt-3.5-turbo", system='you are a helpful assistant', printuser=False)`: This function takes a prompt as input and generates a response from the GPT-3.5 Turbo model. It returns the generated response and logs the conversation in the `conversation_log.txt` file.
You can simply use `op.ask_gpt(prompt)` and keep the default parameters.

2. `op.send_message_gpt(message, model='gpt-3.5-turbo-16k', language='eng', maxtoken=800, persona='', system='', printuser=False)`: This function allows for a more interactive conversation with the GPT-3.5 Turbo model. It takes a message as input, generates a response from the model, and updates the conversation history. It also logs the conversation in the `conversation_log.txt` file.
You can simply `op.send_message_gpt(message)` and keep the default parameters.

The module also provides additional utility functions for managing the conversation, such as clearing the chat history, setting a persona, and setting system instructions.

## Trial
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/johndef64/pychatgpt/blob/main/pychatgpt_trial.ipynb) 

## Notes
The code in this module assumes that the conversation history is stored in a global variable named `conversation_gpt`. Use `print(op.conversation_gpt)` to show conversation history and `op.conversation_gpt.pop()` to remove last interacition. `op.send_message_gpt('clearchat')` to start a new conversation.

Using `op.send_message_gpt`, the code checks if the total number of tokens exceeds the model's maximum context length (gpt 3.5 turbo-16k: 16,384 tokens). If it does, a warning message indicates that the token limit is being reached and then then the first part of the conversation will automatically be deleted to make room for the next interaction.

## Author
Written by: JohnDef64 

## Acknowledgment
This module only facilitates interaction with the OpenAI API and keeps track of it. OpenAI holds all rights to ChatGPT.