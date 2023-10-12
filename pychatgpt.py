import os
import ast
import glob
import json
import importlib
import pandas as pd
from datetime import datetime

def simple_bool(message):
    choose = input(message+" (y/n): ").lower()
    your_bool = choose in ["y", "yes"]
    return your_bool

def display_file_as_pd(ext, path=os.getcwd(), contains=''):
    file_pattern = os.path.join(path, "*."+ext)
    files = glob.glob(file_pattern)
    files_name = []
    for file in files:
        file_name = os.path.basename(file)
        files_name.append(file_name)
    files_df = pd.Series(files_name)
    file = files_df[files_df.str.contains(contains)]
    return file

def display_allfile_as_pd(path=os.getcwd(), contains=''):
    file_pattern = os.path.join(path, "*")
    files = glob.glob(file_pattern)
    files_name = []
    for file in files:
        file_name = os.path.basename(file)
        files_name.append(file_name)
    files_df = pd.Series(files_name)
    file = files_df[files_df.str.contains(contains)]
    return file

def check_and_install_module(module_name):
    try:
        # Check if the module is already installed
        importlib.import_module(module_name)
        #import module_name
        #print(f"The module '{module_name}' is already installed.")
    except ImportError:
        # If the module is not installed, try installing it
        x = simple_bool(
            "\n" + module_name + "  module is not installed.\nwould you like to install it?")
        if x:
            import subprocess
            subprocess.check_call(["pip", "install", module_name])
            print(f"The module '{module_name}' was installed correctly.")
        else:
            exit()

check_and_install_module("openai")
import openai

current_dir = os.getcwd()
api_key = None
if not os.path.isfile(current_dir + '/openai_api_key.txt'):
    with open(current_dir + '/openai_api_key.txt', 'w') as file:
        file.write(input('insert here your openai api key:'))

api_key = open(current_dir + '/openai_api_key.txt', 'r').read()
openai.api_key = str(api_key)
add = "You are a helpful assistant."

current_dir = os.getcwd()

def change_key():
    change_key = simple_bool('change API key?')
    if change_key:
        api_key = None
        with open(current_dir + '/openai_api_key.txt', 'w') as file:
            file.write(input('insert here your openai api key:'))

        api_key = open(current_dir + '/openai_api_key.txt', 'r').read()
        openai.api_key = str(api_key)


#inizialize log:
if not os.path.isfile(current_dir + '/conversation_log.txt'):
    with open(current_dir + '/conversation_log.txt', 'w', encoding= 'utf-8') as file:
        file.write('Auto-GPT\n\nConversation LOG:\n')
        print(str('\nconversation_log.txt created at ' + os.getcwd()))


# ask function ----------------------------
#https://platform.openai.com/account/rate-limits
#https://platform.openai.com/account/usage
def ask_gpt(prompt,
            model = "gpt-3.5-turbo",
            system= 'you are an helpful assistant',
            printuser = False
            ):
    completion = openai.ChatCompletion.create(
        #https://platform.openai.com/docs/models/gpt-4
        model= model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ])
    with open('conversation_log.txt', 'a', encoding= 'utf-8') as file:
        file.write('---------------------------')
        if add != '':
            file.write('\nSystem: \n"' + system+'"\n')
        file.write('\nUser: '+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n' + prompt)
        file.write('\n\nGPT:\n' + completion.choices[0].message.content + '\n\n')

    print_mess = prompt.replace('\r', '\n').replace('\n\n', '\n')
    if printuser: print('user:',print_mess,'\n')
    return print(completion.choices[0].message.content)

# conversation function -----------------------------

total_tokens = 0 # iniziale token count
persona = ''
keep_persona = True

if not 'conversation_gpt' in locals():
    conversation_gpt = []

def expand_conversation_gpt(message):
    conversation_gpt.append({"role": "user", "content": message})

def build_messages(conversation):
    messages = []
    for message in conversation:
        messages.append({"role": message["role"], "content": message["content"]})
    return messages

def save_conversation(path='conversations/'):
    filename = input('Conversation name:')
    directory = 'conversations'
    formatted_json = json.dumps(conversation_gpt, indent=4)
    if not os.path.exists(directory):
        os.mkdir(directory)
    with open(path+filename+'.json', 'w', encoding= 'utf-8') as file:
        file.write(formatted_json)
        file.close()

def load_conversation(path='conversations/'):
    global conversation_gpt
    files_df = display_file_as_pd('json',contains='',path=path)
    files_df = files_df.sort_values().reset_index(drop=True)
    filename = str(files_df[int(input('Choose file:\n'+str(files_df)))])
    with open(path+filename,'r') as file:
        conversation_gpt = ast.literal_eval(file.read())
        file.close()
    print('*conversation',filename,'loaded*')
        
def load_file(path=os.getcwd(), contains=''):
    files_df = display_allfile_as_pd(path)
    filename = str(files_df[int(input('Choose file:\n'+str(files_df)))])
    with open(path+'\\'+filename,'r') as file:
        my_file = file.read()#ast.literal_eval(file.read())
        file.close()
    return my_file

def send_message(message,
                 model='gpt-3.5-turbo-16k',
                 language='eng',
                 maxtoken = 800,
                 temperature = 1,
                 system='',
                 persona='',
                 printuser = False
                 ):
    global conversation_gpt
    global total_tokens

    if model == 'gpt-3.5-turbo-16k':
        token_limit = 16384 - (maxtoken*1.3)
    if model == 'gpt-3.5-turbo':
        token_limit = 4096 - (maxtoken*1.3)
    if model == 'gpt-4':
        token_limit = 8192 - (maxtoken*1.3)
    if model == 'gpt-4-32k':
        token_limit = 32768 - (maxtoken*1.3)
        #https://platform.openai.com/docs/models/gpt-4

    if message == 'clearchat':
        conversation_gpt = []
        print('*chat cleared*\n')

    if system != '':
        conversation_gpt.append({"role": "system",
                                 "content": system})
        persona = ''

    persona_dict = {'system': "You are " + persona + ". Think, feel and answer accordingly.",
                    'character': "You are now impersonating "+persona+". Please reflect "+persona+"'s traits in all interactions. Make sure to use an appropriate language style and uphold an attitude or mindset that aligns with "+persona+"'s character.",
                    'sistema': "Tu sei " + persona + ". Pensa, ragiona e rispondi di conseguenza.",
                    'personaggio': "Stai impersonando "+persona+", . Ricorda di riflettere i tratti di "+persona+" in tutte le interazioni. Assicurati di utilizzare uno stile linguistico appropriato e di mantenere un atteggiamento o una mentalità in linea con il personaggio di "+persona}
    if persona != '':
        if language == 'eng':
            conversation_gpt.append({"role": "system",
                                     "content": persona_dict['character']})
        if language== 'ita':
            conversation_gpt.append({"role": "system",
                                     "content": persona_dict['personaggio']})

    if total_tokens > token_limit:
        print('\nWarning: reaching token limit. \nThis model maximum context length is ', token_limit, ' => early interactions in the conversation are forgotten\n')

        if model == 'gpt-3.5-turbo-16k':
            cut_length = len(conversation_gpt) // 10
        if model == 'gpt-4':
            cut_length = len(conversation_gpt) // 6
        if model == 'gpt-3.5-turbo':
            cut_length = len(conversation_gpt) // 3
        conversation_gpt = conversation_gpt[cut_length:]

        if keep_persona and persona != '':
            if language == 'ita':
                conversation_gpt.append({"role": "system", "content": persona_dict['personaggio']})
            elif language == 'eng':
                conversation_gpt.append({"role": "system", "content": persona_dict['character']})
        if keep_persona and system != '':
            conversation_gpt.append({"role": "system", "content": system})

    if message != 'clearchat':
        expand_conversation_gpt(message)
        messages = build_messages(conversation_gpt)
        response = openai.ChatCompletion.create(
            model = model,
            messages = messages,
            temperature = temperature,
            max_tokens = maxtoken,  # set max token
            top_p = 1,
            frequency_penalty = 0,
            presence_penalty = 0
        )
        # Add the assistant's reply to the conversation
        with open('conversation_log.txt', 'a', encoding= 'utf-8') as file:
            file.write('---------------------------')
            file.write('\nUser: '+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n' + message)
            if persona != '' and persona.find(',') != -1:
                comma_ind = persona.find(',')
                persona_p = persona[:comma_ind]
            elif persona != '' and persona.find(',') == -1:
                persona_p = persona
            elif persona == '':
                persona_p = model
            file.write('\n\n'+persona_p+':\n' + response.choices[0].message.content + '\n\n')

        conversation_gpt.append({"role": "assistant", "content": response.choices[0].message.content})

        # Print the assistant's response
        print_mess = message.replace('\r', '\n').replace('\n\n', '\n')

        if printuser: print('user:',print_mess,'\n...')
        print(response.choices[0].message.content)
        #answer = response.choices[0].message.content
        total_tokens = response.usage.total_tokens
        print('prompt tokens:', total_tokens)

