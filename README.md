# MyChatGPT: Python Module
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/johndef64/mychatgpt/blob/main/notebooks/mychatgpt_trial.ipynb) [![PyPI Latest Release](https://img.shields.io/pypi/v/mychatgpt.svg)](https://pypi.org/project/mychatgpt/) 

`mychatgpt` is a small and useful Python module that provides functions for interacting with OpenAI's GPT models to create conversational agents. This module allows users to have interactive conversations with the GPT models and keeps track of the conversation history in your Python Projects and Jupyter Notebooks.

Now implemented with vision, hearing and drawing functions.

## Installation

Install package with pip: 
```bash
pip install mychatgpt
```
Install latest version through git
```bash
pip install git+https://github.com/johndef64/mychatgpt.git
```

To use this module, you need an **OpenAI API key**. You have to provide your API key when requested once and it will be stored as `openai_api_key.txt` in your working directory.

## Usage

The module provides the following main functions:

1. `GPT().ask_gpt(prompt, *args)`:  
This basic function takes a prompt as input and generates a single response from the GPT chosen model. It returns the generated response and update `chat_log.json`.
You can simply use `op.ask_gpt(prompt)` and keep the default parameters.

2. `GPT().send_message(message,*args)`:  
This main function allows for a more interactive conversation with the GPT chosen model. It takes a message as input, generates a response from the model, and updates the conversation history. It also logs the conversation in the `chat_log.json` file.  
This function is implemented with GPT vision, Text2Speech and Dall-E functionalities.
Use `op.send_message(message)` keeping the default *parameters* or change them as function operators:

```python
from mychatgpt import GPT

op = GPT(assistant='',            # in-build assistant name 
         persona='',              # any known character
         format='',               # output format (latex,python,markdown)
         translate=False,         # translate outputs
         translate_jap=False,     # translate in jap outputs
         save_log=True,           # save log file
         to_clip=True,            # send reply t clipboard
         print_token=True,        # print token count
         model='gpt-4o-mini',     # set openai main model
         talk_model='gpt-4o-2024-08-06', # set openai speak model
         dalle="dall-e-2",        # set dall-e model
         image_size='512x512',    # set generated image size
         )

op.chat('Your message goes here', 
        max=1000,          # max tokens in reply
        img='',            # insert an image path to activate gpt vision
        paste=False,       # append clipboard to message
        create=False       # create an image
        )

```
```python
op.add_persona('Elon Musk')
op.chat("""What do you think about OpenAI?""")
```

```python
elon = GPT('Elon Musk')
elon.chat("""What do you think about OpenAI?""")
```

```python
vincent = GPT('Vincent Van Gogh')
vincent.chat("""Tell me what you see. Can you paint it?""", img=vincent.dummy_img)
```
        
3. `op.send_image(url,*parameters*)` insert in your chat context gpt-vision, activate  a multimodal chat  
```python
op.send_image(image="https://repo.com/image.jpg",
              message="What’s in this image?",
              system = '',     # add 'system' instruction
              model= "gpt-4o", #"gpt-4-turbo", "gpt-4-vision-preview"
              maxtoken=1000, 
              lag=0.00, printreply=True)
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

op.text2speech(text,
               voice="alloy",
               filename="speech.mp3",
               model="tts-1",
               speed=1,
               play=False)

voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
response_formats = ["mp3", "flac", "aac", "opus"]
```

6. Speak With...
```python
op.speak(message='',
         system='',
         voice='nova', 
         language='eng', 
         tts= 'tts-1', max=1000, printall=False)

# Use an in-build assistant or any character of your choice, example:
socrates = GPT('Socrates')
socrates.speak('Tell me about the Truth.', 'onyx')

# Endless chat, keyboard controlled
socrates.speak_loop(system='',
                    voice='nova', tts= 'tts-1', max=1000, language='eng', printall=False, exit_chat='stop')
```
7. Talk With...
```python
dua = GPT('Dua Lipa')
dua.talk(voice='nova', language='eng', gpt='gpt-4-turbo', tts= 'tts-1', max=1000, printall=False)

# Endless talk, keyboard controlled
dua.talk_loop(voice='nova', language='eng', gpt='gpt-4-turbo', tts= 'tts-1', max=1000, printall=False, chat='alt' , exit='shift')
```
```python
nietzsche = GPT('Friedrich Nietzsche')
nietzsche.talk_loop('onyx')
```

The module also provides additional utility functions for managing the conversation, such as clearing the chat history, setting a persona, and setting system instructions, save/load chats.

1. `choose_model()`
2. `clear_chat()`
3. `expand_chat()`
4. `save_chat()`
5. `load_chat()`
6. `load_file()`

To set-up multiple conversations or change the API-key, follow the example proposed in [mychatgpt_trial.ipynb](https://github.com/johndef64/mychatgpt/blob/main/mychatgpt_trial.ipynb)

## In-Build Assistants
```python
op.display_assistants()
```
```python
from mychatgpt import delamain
# Call an assistant simply by name
delamain.chat('your message',
              gpt='gpt-4o', 
              max = 1000, 
              clip=True)  

#n.b. assistants sends reply to clipboard by default
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
