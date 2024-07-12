import os
import re
import time
import requests
import pandas as pd



# check requirements
def get_gitfile(url, flag='', dir = os.getcwd()):
    url = url.replace('blob','raw')
    response = requests.get(url)
    file_name = flag + url.rsplit('/',1)[1]
    file_path = os.path.join(dir, file_name)
    with open(file_path, 'wb') as file:
        file.write(response.content)

if not os.getcwd().endswith('pychatgpt.py'):
    handle="https://raw.githubusercontent.com/johndef64/pychatgpt/main/"
    files = ["pychatgpt.py"]
    for file in files:
        url = handle+file
        get_gitfile(url)

import pychatgpt as op
op.check_and_install_module('pyperclip')
import pyperclip

def get_boolean_input(prompt):
    while True:
        try:
            return {"y": True, "n": False}[input(prompt).lower()]
        except KeyError:
            print("Invalid input. Please enter 'y' or 'n'.")

def starts_with_japanese_character(s):
    # Check if the string starts with a Japanese character (Hiragana, Katakana, CJK Unified Ideographs)
    return re.match(r'^[\u3040-\u30FF\u4E00-\u9FFF]', s) is not None

def simple_bool(message, y='y', n ='n'):
    choose = input(message+" ("+y+"/"+n+"): ").lower()
    your_bool = choose in [y]
    return your_bool
#print('--------------------------------------')

current_dir = ''
# to run in python terminal you need this:
if os.getcwd() == 'C:\WINDOWS\System32':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
else:
    current_dir = os.getcwd()

# check Usage:
# https://platform.openai.com/account/usage

# Set API key:
# https://platform.openai.com/account/api-keys


# Application dictionary
dict = {
    'title': "---------------------\nWelcome to Clipboard2 Speech!\n\nOpenai will read aloud everything you will send to clipboard.\n\nSend to clipboard:\n- 'restartnow' to start over the application.\n- 'exitnow' to shut down the application.\n- 'maxtoken' to set up max token in response (chat mode).\n- 'rest'/'wake' To temporarily suspend and wakeup the application.\n\nwritten by JohnDef64\n---------------------\n",

}
####### text-to-speech #######
voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
models = ["tts-1","tts-1-hd"]

#-----------------------------------------------------
# Run script:
while True:  # external cycle
    safe_word = ''
    pyperclip.copy('')
    print(dict['title'])
    voice_id = 4#int(input('Choose Voice:\n'+str(pd.Series(voices))+':\n'))
    voice = voices[voice_id]
    print('Using',voice,'voice')

    model_id = 0#harint(input('Choose TTS Model:\n'+str(pd.Series(models))+':\n'))
    model = models[model_id]
    print('Using',model,'model')

    translate = simple_bool('Translate?')

    previous_content = ''  # Initializes the previous contents of the clipboard

    while safe_word != 'restartnow' or 'exitnow' or 'maxtoken' or 'wake':
        safe_word = pyperclip.paste()
        clipboard_content = pyperclip.paste()  # Copy content from the clipboard

        x = True
        while safe_word == 'rest':
            if x : print('<sleep>')
            x = False
            clipboard_content = pyperclip.paste()
            time.sleep(1)
            if clipboard_content == 'wake':
                print('<Go!>')
                break

        if starts_with_japanese_character(clipboard_content):
            if clipboard_content != previous_content and clipboard_content.startswith('sys:'):
                add = clipboard_content.replace('sys:','')
                previous_content = clipboard_content
                print('\n<system setting changed>\n')

            if clipboard_content != previous_content and clipboard_content != 'restartnow':  # Check if the content has changed
                #print(clipboard_content)  # Print the content if it has changed
                print('\n\n--------------------------------\n')
                previous_content = clipboard_content  # Update previous content

                # request to openai
                print(clipboard_content)
                op.text2speech_stream(clipboard_content, voice=voice, model=model)
                if translate:
                    #op.clearchat(False)
                    op.english(clipboard_content, clip=False)
                    op.text2speech_stream(op.reply, voice=voice, model=model)


        time.sleep(1)  # Wait 1 second before checking again
        if safe_word == 'restartnow':
            break

        if safe_word == 'exitnow':
            exit()

        if safe_word == 'maxtoken':
            #global maxtoken
            #custom = get_boolean_input('\ndo you want to cutomize token options? (yes:1/no:0):')
            #if custom:
            op.maxtoken = int(input('set max response tokens (800 default):'))
            break

# openai.error.InvalidRequestError: This model's maximum context length is 4097 tokens.  However, you requested 4116 tokens (2116 in the messages, 2000 in the completion).  Please reduce the length of the messages or completion.
#%%
