# PyChatGPT Python Module

`pychatgpt` is small and useful Python module that provides functions for interacting with OpenAI's GPT-3.5 Turbo model to create conversational agents. This module allows users to have interactive conversations with the GPT-3.5 Turbo model and keeps track of the conversation history in your Python Projects or working on a Jupyter Notebook.

## Installation
Add the `pychatgpt.py` file to the `site-packages` directory in your current Python environment:
- in a Conda envoiroment `C:\Users\*your_user_name*\.conda\envs\*your_env*\Lib\site-packages`;
- in a Python enviroment `/lib/site-packages`.

Or simply add it to your Project Workspace

## Requirements

- The `openai` module is required. It will prompt automatic installation if this requirement is not met.
- To use this module, you need to have an OpenAI API key. You have to provide your API key when requested once and it will be stored automatically in a file called `openai_api_key.txt` in your working directory. Get your API key here [API keys - OpenAI API](https://platform.openai.com/account/api-keys)

## Usage
`import pychatgpt as op`

The module provides the following main functions:

1. `op.ask_gpt(prompt, model = "gpt-3.5-turbo" system='you are a helpful assistant', printuser=False)`: This function takes a prompt as input and generates a single response from the GPT-3.5 Turbo model. It returns the generated response and logs the interaction in the `conversation_log.txt` file.
You can simply use `op.ask_gpt(prompt)` and keep the default parameters.

2. `op.send_message_gpt(message, model='gpt-3.5-turbo-16k', language='eng', maxtoken=1000, persona='', system='', printuser=False)`: This function allows for a more interactive conversation with the GPT-3.5 Turbo model. It takes a message as input, generates a response from the model, and updates the conversation history. It also logs the conversation in the `conversation_log.txt` file.
You can simply `op.send_message_gpt(message)` and keep the default parameters.

The module also provides additional utility functions for managing the conversation, such as clearing the chat history, setting a persona, and setting a system message.

## Notes
The code in this module assumes that the conversation history is stored in a global variable named `conversation_gpt`. Use `print(op.conversation_gpt)` to show the running conversation and `op.conversation_gpt.pop()` to remove last interacition.

Using `op.send_message_gpt`, the code checks if the total number of tokens exceeds the model's maximum context length . If it does, a warning message indicates that the token limit is being reached and then then the first third of the conversation will automatically be deleted to make room for the next interaction.
- gpt-3.5-turbo:      4,097 tokens
- gpt-3.5-turbo-16k:  16,384 tokens

## Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/johndef64/pychatgpt/blob/main/pychatgpt_trial.ipynb)

## Author
Written by: JohnDef64 

## Credits
This project utilizes technology provided by OpenAI. For more information, please visit: [https://openai.com](https://openai.com)
