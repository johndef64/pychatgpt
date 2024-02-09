import os
import time
import importlib
import requests
from datetime import datetime

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

def get_boolean_input(prompt):
    while True:
        try:
            return {"y": True, "n": False}[input(prompt).lower()]
        except KeyError:
            print("Invalid input. Please enter 'y' or 'n'.")

def check_and_install_module(module_name):
    try:
        # Check if the module is already installed
        importlib.import_module(module_name)
        #import module_name
        #print(f"The module '{module_name}' is already installed.")
    except ImportError:
        # If the module is not installed, try installing it
        x = get_boolean_input(
            "\n" + module_name + "  module is not installed.\nwould you like to install it? (y/n):")
        if x:
            import subprocess
            subprocess.check_call(["pip", "install", module_name])
            print(f"The module '{module_name}' was installed correctly.")
        else:
            exit()


# check requirements
try:
    import pandas as pd
    import openai
    import tiktoken
except ImportError:
    print('Check requirements:')

check_and_install_module("pandas")
check_and_install_module("openai")
check_and_install_module("tiktoken")
check_and_install_module('pyperclip')
#print('--------------------------------------')

current_dir = ''
# to run in python terminal you need this:
if os.getcwd() == 'C:\WINDOWS\System32':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
else:
    current_dir = os.getcwd()

# If  import openai
import pychatgpt as op
import pyperclip
import pandas as pd

# check Usage:
# https://platform.openai.com/account/usage

# Set API key:
# https://platform.openai.com/account/api-keys




# Parameters--------------------------
#The Max Token Limit In ChatGPT: gpt-4 (8,192 tokens), gpt-4-0613 (8,192 tokens), gpt-4-32k (32,768 tokens), gpt-4-32k-0613 (32,768 tokens). maximum limit (4097 tokens for gpt-3.5-turbo )


#================================================================

#================================================================

# Run script:
while True:  # external cycle
    safe_word = ''

    print(
        '''---------------------\nWelcome to GPT-CLI!\n\nChatGPT will answer every question.\n\nReply with:\n- 'restartnow' to start over the application.\n- 'exitnow' to shut down the application.\n- 'maxtoken' to set up max token in response (chat mode).\n- 'system' to set new system instructions' to change system instructions (instruct mode)'\n\nwritten by JohnDef64\n---------------------\n''')
    language = '1'  # setting Italian default temporarly

    '''while language not in ['1', '2']:
        language = input('\nPlease choose language: \n1. English\n2. Italian\n\nLanguage number:')
        if language not in ['1', '2']:
            print("Invalid choice.")'''

    #----------
    op.choose_model()
    print('-------------')
    op.select_assistant()
    print('Start Chat')

    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    while safe_word != 'restartnow' or 'exitnow' or 'maxtoken':
        timea = datetime.now()
        print('--------------------------------')

        message = input()
        #print(message)

        safe_word = message
        if safe_word == 'restartnow':
            conversation_gpt = []
            break
        if safe_word == 'exitnow':
            exit()
        if safe_word == 'maxtoken':
            maxtoken = int(input('set max response tokens (1000 default):'))
            break
        if safe_word == 'system':
            conversation_gpt = []
            system = input('define custum system instructions:')
            print('*system instruction changed*')
            pass
        else:
            op.send_message(message)

        timed = datetime.now() - timea

        #Rate-Limit gpt-4 = 200
        #https://medium.com/@pankaj_pandey/understanding-the-chatgpt-api-key-information-and-frequently-asked-questions-4a0e963fb138#:~:text=The%20ChatGPT%20API%20has%20different,90000%20TPM%20after%2048%20hours.
        #https://platform.openai.com/docs/guides/rate-limits/overview
        #https://platform.openai.com/account/rate-limits
        time.sleep(0.5)  # Wait 1 second before checking again


#%%
