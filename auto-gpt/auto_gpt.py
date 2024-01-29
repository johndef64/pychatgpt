import os
import time
from datetime import datetime
import requests


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


# Parameters--------------------------
persona = ''
#model = 'gpt-4'

add = "You are a helpful assistant."

'''InvalidRequestError: This model's maximum context length is 4097 tokens. However, you requested 4201 tokens (3201 in the messages, 1000 in the completion). Please reduce the length of the messages or completion.
last token report 2568 (not cosidering response tokens that could be up to 500-700)
'''


# Applications dictionaries
dicteng = {
    'title': "---------------------\nWelcome to Auto-GPT!\n\nGPT will answer everything you will send to clipboard.\n\nSend to clipboard:\n- 'restartnow' to start over the application.\n- 'exitnow' to shut down the application.\n- 'maxtoken' to set up max token in response (chat mode).\n- 'rest'/'wake' To temporarily suspend and wakeup the application.\n- 'sys: (new system instructions)' to change system instructions (instruct mode)'\n\nwritten by JohnDef64\n---------------------\n",

    'language': '\nPlease choose language: \n1. English\n2. Italian\n\nLanguage number:',

    'chat': "\nChoose chat-mode: \n1. Ask to gpt-4\n2. Ask to gpt-4 (instruct)\n3. Ask to someone\n4. Chat with gpt-4\n\nNumber:",

    'instruct': '''\nEnter system instructions.\nexamples:\n - Answer the question and give only the correct answer:\n - Correct this code:\n  Instruction:''',

    'instruction': "\nWho do you want to converse with?: \n1. the assistant\n2. the poet \n3. the scientist\n4. someone else\n\nNumber: ",

    'instructions': {'assistant': "Today, you are the 'expert assistant in everything. Answer in accordance with this.",
                     'poet': "Today, you are the greatest poet ever, inspired, profound, sensitive, visionary and creative. Answer in accordance with this.",
                     'scientist': "Today, you are the greatest and most experienced scientist ever, analytical, precise and rational. Answer in accordance with this.",
                     'tell me': 'Tell me who you want to talk to:'
                     }
}

dictita = {
    'title': "---------------------\nBenvenuto in Auto-GPT!\n\nGPT risponderà a tutto ciò che invierai nella clipboard.\n\nInvia alla clipboard:\n- 'restartnow' per riavviare l'applicazione.\n- 'exitnow' per spegnere l'applicazione.\n- 'maxtoken' per impostare il massimo numero di token nella risposta (modalità chat).\n- 'rest'/'wake' Per sospendere temporaneamente e risvegliare l'applicazione.\n- 'sys: (nuove istruzioni di sistema)' per cambiare le istruzioni di sistema (modalità istruzione)'\n\nscritto da JohnDef64\n---------------------\n",

    'language': '\nPer favore scegli la lingua: \n1. Inglese\n2. Italiano\n\nNumero della lingua:',

    'chat': "\nScegli la modalità chat: \n1. Chiedi a gpt-4\n2. Chiedi a gpt-4 (istruzione)\n3. Chiedi a qualcuno\n4. Chatta con gpt-4\n\nNumero:",

    'instruct': '''\nInserisci le istruzioni di sistema.\nesempi:\n - Rispondi alla domanda e dai solo la risposta corretta:\n - Correggi questo codice:\n  Istruzione:''',

    'instruction': "\nCon chi vuoi conversare?: \n1. l'assistente\n2. il poeta \n3. lo scienziato\n4. qualcun altro\n\nNumero: ",

    'instructions': {'assistant': "Oggi, tu sei l' 'esperto assistente in tutto. Rispondi in conformità con questo.",
                     'poet': "Oggi, tu sei il più grande poeta di sempre, ispirato, profondo, sensibile, visionario e creativo. Rispondi in conformità con questo.",
                     'scientist': "Oggi, tu sei il più grande e più esperto scienziato di sempre, analitico, preciso e razionale. Rispondi in conformità con questo.",
                     'tell me': 'Dimmi con chi vuoi parlare:'
                     }
}

language = ''
dict = dicteng

#-----------------------------------------------------
# Run script:
while True:  # external cycle
    safe_word = ''
    pyperclip.copy('')
    print(dict['title'])

    while language not in ['1', '2']:
        language = input(dict['language'])
        if language not in ['1', '2']:
            print("Invalid choice.")
    if language == '1':
        dict = dicteng
    else:
        dict = dictita

    # choose model
    op.choose_model()
    #----------
    #if language == '1':
    choose = ""
    while choose not in ['1', '2', '3', '4']:
        # Prompt the user to make a choice between three predefined strings
        choose = input(dict['chat'])

        # Verify user's choice
        if choose == '1':
            with open('chat_log.txt', 'a') as file:
                file.write('\n' + str(datetime.now()) + ": <You are asking the virtual assistant>\n")
            print("You chose option 1.")

        elif choose == '2':
            add = input(dict['instruct'])
            with open('chat_log.txt', 'a') as file:
                file.write('\n' + str(datetime.now()) + ": <You are asking the virtual assistant, system: "+add+"> \n")
            print("You chose option 2.")

        elif choose == '3':
            op.persona = input('Tell me who you want to talk to:')
            with open('chat_log.txt', 'a') as file:
                file.write('\n' + str(datetime.now()) + ': <You asked to ' + op.persona + '>\n')
            print("\nI'm connecting you with " + op.persona)

        elif choose == '4':
            print("You chose to have some conversation.")
            assistant = input(dict['instruction'])
            if assistant == '1':
                op.chat_gpt.append({"role": "system",
                                         "content": dict['instructions']['assistant']})
                with open('chat_log.txt', 'a') as file:
                    file.write('\n' + str(datetime.now()) + ': <You are chatting with the general assistant>\n')
            elif assistant == '2':
                op.chat_gpt.append({"role": "system",
                                         "content": dict['instructions']['poet']})
                with open('chat_log.txt', 'a') as file:
                    file.write('\n' + str(datetime.now()) + ': <You are chatting with the greatest poet ever>\n')
            elif assistant == '3':
                op.chat_gpt.append({"role": "system",
                                         "content": dict['instructions']['scientist']})
                with open('chat_log.txt', 'a') as file:
                    file.write(
                        '\n' + str(datetime.now()) + ': <You are chatting with the greatest  scientist ever>\n')
            elif assistant == '4':
                #persona = input(dict['instructions']['tell me'])
                op.persona = input('Tell me who you want to talk to:')

                with open('chat_log.txt', 'a') as file:
                    file.write('\n' + str(datetime.now()) + ': <You are chatting with ' + op.persona + '>\n')
                print("\nI'm connecting you with " + op.persona)
        else:
            print("Invalid choice.")

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

        if clipboard_content != previous_content and clipboard_content.startswith('sys:'):
            add = clipboard_content.replace('sys:','')
            previous_content = clipboard_content
            print('\n<system setting changed>\n')

        if clipboard_content != previous_content and clipboard_content != 'restartnow':  # Check if the content has changed
            #print(clipboard_content)  # Print the content if it has changed
            print('\n\n--------------------------------\n')
            previous_content = clipboard_content  # Update previous content

            if choose == '4':
                if language == 1:
                    op.send_message(clipboard_content, persona=op.persona, language='eng')
                else:
                    op.send_message(clipboard_content, persona=op.persona, language='ita')

            else:
                if language == 1:
                    op.ask_gpt(clipboard_content, persona=op.persona, language='eng')
                else:
                    op.ask_gpt(clipboard_content, persona=op.persona, language='ita')

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
