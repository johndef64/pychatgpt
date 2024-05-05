#%%

#%%
import os
import time
import importlib
import requests
import subprocess
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
    files = ["pychatgpt_lite.py", 'pychatgpt.py']
    for file in files:
        url = handle+file
        get_gitfile(url)
        time.sleep(0.2)

def simple_bool(message, y='y', n ='n'):
    choose = input(message+" ("+y+"/"+n+"): ").lower()
    your_bool = choose in [y]
    return your_bool

def get_boolean_input(prompt):
    while True:
        try:
            return {"y": True, "n": False}[input(prompt).lower()]
        except KeyError:
            print("Invalid input. Please enter 'y' or 'n'.")

def check_and_install_requirements(requirements):
    missing_requirements = []
    for module in requirements:
        try:
            # Check if the module is already installed
            importlib.import_module(module)
        except ImportError:
            missing_requirements.append(module)
    if len(missing_requirements) == 0:
        pass
    else:
        x = simple_bool(str(missing_requirements)+" are missing.\nWould you like to install them all?")
        if x:
            for module in missing_requirements:
                subprocess.check_call(["pip", "install", module])
                print(f"{module}' was installed correctly.")
        else:
            exit()


# check requirements
try:
    import pandas as pd
    import openai
    import tiktoken
except ImportError:
    print('Check requirements:')

requirements = ["openai", "tiktoken", "pandas", "pyperclip"]
check_and_install_requirements(requirements)

#print('--------------------------------------')

# to run in python terminal you need this:
if os.getcwd() == 'C:\WINDOWS\System32':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
else:
    current_dir = os.getcwd()

# If  import openai
#import pychatgpt_lite as op
import pychatgpt as op

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

        message = input('--------------------------------\nuser:')
        #print(message)

        safe_word = message
        if safe_word == 'restartnow':
            op.clearchat()
            break
        if safe_word == 'exitnow':
            exit()
        if safe_word == 'maxtoken':
            maxtoken = int(input('\nSet max response tokens (1000 default):'))
            break
        if safe_word == 'system':
            op.clearchat()
            system = input('\nDefine custom system instructions:')
            print('*system instruction changed*')
            pass
        else:
            op.send_message(message, to_clip=True)

        timed = datetime.now() - timea

        #Rate-Limit gpt-4 = 200
        #https://medium.com/@pankaj_pandey/understanding-the-chatgpt-api-key-information-and-frequently-asked-questions-4a0e963fb138#:~:text=The%20ChatGPT%20API%20has%20different,90000%20TPM%20after%2048%20hours.
        #https://platform.openai.com/docs/guides/rate-limits/overview
        #https://platform.openai.com/account/rate-limits
        time.sleep(0.5)  # Wait 1 second before checking again

