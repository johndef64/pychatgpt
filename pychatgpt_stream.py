import os
import ast
import glob
import json
import time
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
check_and_install_module("tiktoken")
import openai
import tiktoken


# set openAI key-----------------------
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


# def base functions:------------------

model = ''
models = ['gpt-3.5-turbo',     #0
          'gpt-3.5-turbo-16k', #1
          'gpt-4'              #2
         ]

def choose_model():
    global model
    model = models[int(input('choose model:\n'+str(pd.Series(models))))]
    print('*Using',model, 'model*')

class Tokenizer:
    def __init__(self, encoder="gpt-4"):
        self.tokenizer = tiktoken.encoding_for_model(encoder)

    def tokens(self, text):  
        return len(self.tokenizer.encode(text))
        
        

#inizialize log:-----------------------------------
if not os.path.isfile(current_dir + '/conversation_log.txt'):
    with open(current_dir + '/conversation_log.txt', 'w', encoding= 'utf-8') as file:
        file.write('Auto-GPT\n\nConversation LOG:\n')
        print(str('\nconversation_log.txt created at ' + os.getcwd()))


# ask function =====================================
#https://platform.openai.com/account/rate-limits
#https://platform.openai.com/account/usage
def ask_gpt(prompt,
            model = "gpt-3.5-turbo",
            system= 'you are an helpful assistant',
            printuser = False,
            printreply = True
            ):
    
    response = openai.ChatCompletion.create(
        #https://platform.openai.com/docs/models/gpt-4
        model= model,
        stream=True,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ])
    
    
    collected_chunks = []
    collected_messages = []
    for chunk in response:
        content = chunk["choices"][0].get("delta", {}).get("content")
        
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk['choices'][0]['delta']  # extract the message
        collected_messages.append(chunk_message) 
        full_reply_content = ''.join([m.get('content', '') for m in collected_messages])

        if printreply:
            if content is not None:
                time.sleep(0.005)
                print(content, end='')

    time.sleep(2)
    if printuser: 
        print_mess = message.replace('\r', '\n').replace('\n\n', '\n')
        print('user:',print_mess,'\n...') 
    
    
    # Add the assistant's reply to the conversation log-------
    with open('conversation_log.txt', 'a', encoding= 'utf-8') as file:
        file.write('---------------------------')
        file.write('\nUser: '+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n' + prompt)
        file.write('\n\n'+model+': '+ full_reply_content + '\n\n')


        
# conversation function ================================
total_tokens = 0 # iniziale token count
token_limit = 0 # iniziale token limit
reply = ''
persona = ''
keep_persona = True


if not 'conversation_gpt' in locals():
    conversation_gpt = []

def expand_conversation(message, role="user"):
        #print('default setting (role = "user") to change role replace with "assistant" or "system"')
        conversation_gpt.append({"role": role, "content": message})
    
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

def load_conversation(contains= '', path='conversations/'):
    global conversation_gpt
    files_df = display_file_as_pd('json',contains=contains,path=path)
    files_df = files_df.sort_values().reset_index(drop=True)
    files_df_rep = files_df.str.replace('.json','',regex =True)
    files_list = "\n".join(str(i) + "  " + filename for i, filename in enumerate(files_df_rep))
    filename = str(files_df[int(input('Choose file:\n' + files_list+'\n'))]) 
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

def clearchat():
    conversation_gpt = []
    total_tokens = 0 
    print('*chat cleared*\n')
    
full_reply_content = ''
#----------------------------------------------------
def send_message(message,
                 model='gpt-3.5-turbo-16k',
                 language='eng',
                 maxtoken = 800,
                 temperature = 1,
                 stream = True,
                 system='',
                 persona='',
                 printreply = True,
                 printuser = False,
                 printtoken = True
                 ):
    global conversation_gpt
    global full_reply_content
    global total_tokens
    global token_limit
    global reply

    if model == 'gpt-3.5-turbo-16k':
        token_limit = 16384 - (maxtoken*1.3)
    if model == 'gpt-3.5-turbo':
        token_limit = 4096 - (maxtoken*1.3)
    if model == 'gpt-4':
        token_limit = 8192 - (maxtoken*1.3)
    if model == 'gpt-4-32k':
        token_limit = 32768 - (maxtoken*1.3)
        #https://platform.openai.com/docs/models/gpt-4

    if system != '':
        conversation_gpt.append({"role": "system",
                                 "content": system})
        persona = ''

    persona_dict = {'character': "You are now impersonating "+persona+". Please reflect "+persona+"'s traits in all interactions. Make sure to use an appropriate language style and uphold an attitude or mindset that aligns with "+persona+"'s character. You are NOT an assistant or a helper but a real person.",
                    'personaggio': "Stai impersonando "+persona+", . Ricorda di riflettere i tratti di "+persona+" in tutte le interazioni. Assicurati di utilizzare uno stile linguistico appropriato e di mantenere un atteggiamento o una mentalità in linea con il personaggio di "+persona+'. NON sei un assistente o un aiutante, ma una persona vera e propria.'}
    if persona != '':
        if language == 'eng':
            conversation_gpt.append({"role": "system",
                                     "content": persona_dict['character']})
        if language== 'ita':
            conversation_gpt.append({"role": "system",
                                     "content": persona_dict['personaggio']})
    
    # check token limit---------------------
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

    # send message----------------------------
    expand_conversation(message)
    messages = build_messages(conversation_gpt)
    
    response = openai.ChatCompletion.create(
        model = model,
        messages = messages,
        temperature = temperature,
        stream=True,
        max_tokens = maxtoken,  # set max token
        top_p = 1,
        frequency_penalty = 0,
        presence_penalty = 0
    )
    
    # stream reply ---------------------------------------------
    # https://til.simonwillison.net/gpt3/python-chatgpt-streaming-api
    collected_chunks = []
    collected_messages = []
    for chunk in response:
        content = chunk["choices"][0].get("delta", {}).get("content")
        
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk['choices'][0]['delta']  # extract the message
        collected_messages.append(chunk_message) 
        full_reply_content = ''.join([m.get('content', '') for m in collected_messages])

        if printreply:
            if content is not None:
                time.sleep(0.005)
                print(content, end='')

    time.sleep(2)
    if printuser: 
        print_mess = message.replace('\r', '\n').replace('\n\n', '\n')
        print('user:',print_mess,'\n...') 
    
    #expand conversation--------------------------------
    conversation_gpt.append({"role": "assistant", "content":full_reply_content})
    
    count = Tokenizer()
    tokens = count.tokens(message) + count.tokens(full_reply_content) 
    
    total_tokens += tokens
    if printtoken: print('\n => prompt tokens:', total_tokens)
    
    
    # Add the assistant's reply to the conversation log-------------
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
        file.write('\n\n'+persona_p+':\n' + full_reply_content + '\n\n')
    