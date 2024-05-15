# PyChatGPT: Python Module
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/johndef64/pychatgpt/blob/main/pychatgpt_trial.ipynb) 

`pychatgpt` is a small and useful Python module that provides functions for interacting with OpenAI's GPT models to create conversational agents. This module allows users to have interactive conversations with the GPT models and keeps track of the conversation history in your Python Projects and Jupyter Notebooks.

## Installation

Simply add `pychatgpt.py` to your working directory. Otherwise, to implement your Conda envoiroment, add it in the folder `C:\Users\\*your_user_name*\\.conda\envs\\*your_env*\\Lib\site-packages`.

To use this module, you need to have an **OpenAI API key**. You have to provide your API key when requested once and it will be stored automatically in a file called `openai_api_key.txt` in your working directory.

## Usage
`import pychatgpt as op`

The module provides the following main functions:

1. `op.ask_gpt(prompt, *parameters*)`:  
This function takes a prompt as input and generates a response from the GPT chosen model. It returns the generated response and logs the conversation in the `chat_log.txt` file.
You can simply use `op.ask_gpt(prompt)` and keep the default parameters.

2. `op.send_message(message,*parameters*)`:  
This function allows for a more interactive conversation with the GPTchosen model. It takes a message as input, generates a response from the model, and updates the conversation history. It also logs the conversation in the `chat_log.txt` file.  
Use `op.send_message(message)` keeping the default *parameters* or change them as function operators:

```python
import pychatgpt as op

op.send_message('Your request',
                model='gpt-3.5-turbo', # choose openai model 
                system='',        # 'system' instruction
                maxtoken=800,     # max tokens in reply
                temperature=1,    # output randomness [0-2]
                lag=0.00,         # word streaming lag

                play= False,      # play audio response
                save_chat=True,   # update chat_log.txt
                to_clip=False,    # send reply to clipboard
                
                print_reply=True, print_user=False, print_token=True,
                )
```
        
3. `op.send_image(url,*parameters*)` insert in your chat context gpt-vision, activate  a multimodal chat  
```python
op.send_image(image="https://repo.com/image.jpg",
              message="What’s in this image?",
              model= "gpt-4o", #"gpt-4-vision-preview"
              maxtoken=1000, 
              lag=0.00, printreply=True, to_clip=True)
```
4. `op.create_image(prompt,*parameters*)`
```python
op.create_image(prompt= "a cute kitten",
                model="dall-e-2",
                size='512x512',
                response_format='b64_json',
                quality="standard",
                time_flag=True, show_image=True)
```
5. Whisper and Text-to-Speech
```python
op.whisper(filepath, # audio.mp3, audio.wav
           translate = False,
           response_format = "text",
           print_transcriprion = True)

voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
response_formats = ["mp3", "flac", "aac", "opus"]
op.text2speech(text,
               voice="alloy",
               filename="speech.mp3",
               model="tts-1",
               speed=1,
               play=False)
```

6. Chat With...
```python
op.chat_with(who='',  
             message='', 
             system='',  
             voice='nova', 
             language='eng', 
             gpt='gpt-4-turbo', 
             tts= 'tts-1', 
             max=1000, 
             printall=False)

# Use an in-build assistant or any character of your choice, example:
op.chat_with('Scorates', 'Tell me about the Truth.', 'onyx')

# Endless chat, keyboard controlled
op.chat_with_loop(who='', system='', voice='nova',  gpt='gpt-4-turbo', tts= 'tts-1', max=1000, language='eng', printall=False, exit_chat='stop')
```
7. Talk With...
```python
op.talk_with(who, voice='nova', language='eng', gpt='gpt-4-turbo', tts= 'tts-1', max=1000, printall=False)

# Endless talk, keyboard controlled
op.talk_with_loop(who, voice='nova', language='eng', gpt='gpt-4-turbo', tts= 'tts-1', max=1000, printall=False, chat='alt' , exit='shift')
```

The module also provides additional utility functions for managing the conversation, such as clearing the chat history, setting a persona, and setting system instructions, save/load chats.

1. `choose_model()`
2. `clear_chat()`
3. `expand_chat()`
4. `save_chat()`
5. `load_chat()`
6. `load_file()`

To set-up multiple conversations or change the API-key, follow the example proposed in [pychatgpt_trial.ipynb](https://github.com/johndef64/pychatgpt/blob/main/pychatgpt_trial.ipynb)

## In-Build Assistants
```python
op.display_assistants()

# Call an assistant simply by name
op.delamain('your message',
            gpt='gpt-4o', 
            max = 1000, 
            clip=True)  
#n.b. assistants sends reply to clipbord by default
```
An extract of the Assistants provided:

| Role            | Assistant Name | Reply type    |
|-----------------|----------------|---------------|
| Copilots        | delamain       | python (main) |
|                 | oracle         | python        |
|                 | roger          | R             |
| Formatters      | schematizer    | bulletpoint   |
|                 | prompt_maker   | promt         |
| Scientific Assistants  | galileo        | markdown      |
|                 | newton         | python        |
|                 | leonardo       | text          |
|                 | turing         | python        |
|                 | penrose        | text          |
| Characters      | bestie         | text          |
|                 | julia          | text          |
| Translators     | english        | text          |
|                 | italian        | text          |
|                 | portuguese     | text          |
|                 | japanese       | text          |



## Notes
The code in this module assumes that the conversation history is stored in a global variable named `chat_thread`. Use `print(op.chat_thread)` to show conversation history and `op.chat_thread.pop()` to remove last interacition. `op.send_message('clearchat')` to start a new conversation.

Using `op.send_message()`, the code checks if the total number of tokens exceeds the model's maximum context length (gpt 3.5 turbo-16k: 16,384 tokens). If it does, a warning message indicates that the token limit is being reached and then then the first part of the conversation will automatically be deleted to make room for the next interaction.

## 




## Openai-based applications 
Some other python applications executable in Terminal that take advantage of openai modulo features:
- auto-gpt
- gpt-cli 
- rec-whisper-paste 


## Author
Written by: JohnDef64 

## Acknowledgment
This module only facilitates interaction with the OpenAI API and keeps track of it. OpenAI holds all rights to ChatGPT.
