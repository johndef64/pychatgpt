import os
from datetime import datetime

def get_boolean_input(prompt):
    while True:
        try:
            return {"1": True, "0": False}[input(prompt).lower()]
        except KeyError:
            print("Invalid input. Please enter '1' or '0'.")

def check_and_install_module(module_name):
    try:
        # Check if the module is already installed
        importlib.import_module(module_name)
        #import module_name
        #print(f"The module '{module_name}' is already installed.")
    except ImportError:
        # If the module is not installed, try installing it
        x = get_boolean_input(
            "\n" + module_name + "  module is not installed.\nwould you like to install it? (yes:1/no:0):")
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

#inizialize log:
if not os.path.isfile(current_dir + '/conversation_log.txt'):
    with open(current_dir + '/conversation_log.txt', 'w', encoding= 'utf-8') as file:
        file.write('Auto-GPT\n\nConversation LOG:\n')
        print(str('\nconversation_log.txt created at ' + os.getcwd()))
        

# chat functions ----------------------------
#https://platform.openai.com/account/rate-limits
def chatWithGPT(prompt, system= 'you are an helpful assistant', printuser = False):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
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

if not 'conversation_gpt' in locals():
    conversation_gpt = []

def expand_conversation_gpt(message):
    conversation_gpt.append({"role": "user", "content": message})

def build_messages(conversation):
    messages = []
    for message in conversation:
        messages.append({"role": message["role"], "content": message["content"]})
    return messages

total_tokens = 0
persona = ''
keep_persona = True
'''InvalidRequestError: This model's maximum context length is 4097 tokens. However, you requested 4201 tokens (3201 in the messages, 1000 in the completion). Please reduce the length of the messages or completion.
last token report 2568 (not cosidering response tokens that could be up to 500-700)
'''

def send_message_gpt(message, language='eng', maxtoken = 800, persona='', system='', printuser = False):
    global conversation_gpt
    global total_tokens
    token_limit = 4096 - (maxtoken*1.2)

    if message == 'clearchat':
        conversation_gpt = []
        print('*chat cleared*\n')

    if system != '':
        conversation_gpt.append({"role": "system",
                                 "content": system})
        persona = ''

    if persona != '':
        if language == 'eng':
            conversation_gpt.append({"role": "system",
                                     "content": "You are " + persona + ". Think, feel answer accordingly."})
        if language== 'ita':
            conversation_gpt.append({"role": "system",
                                     "content": "Tu sei " + persona + ". Pensa, ragiona e senti in accordo."})

    if total_tokens > token_limit:
        print('\nWarning: reaching token limit. \nThis model maximum context length is ', token_limit,
              ' in the messages, 4100 in total.')
        #print('\n Inizializing new conversation.')
        #conversation_gpt.clear()
        if language == 'eng':
            print('\n the first third of the conversation was forgotten\n')
        elif language == 'ita':
            print("il primo terzo della conversazione è stato dimenticato\n")

        quarter_length = len(conversation_gpt) // 3
        conversation_gpt = conversation_gpt[quarter_length:]
        if keep_persona and persona != '':
            if language == 'ita':
                conversation_gpt.append({"role": "system", "content": "Tu sei " + persona + ". Pensa, senti e rispondi di conseguenza."})
            elif language == 'eng':
                conversation_gpt.append({"role": "system", "content": "You are " + persona + ". Think, feel and respond in accordance with this."})
        if keep_persona and system != '':
            conversation_gpt.append({"role": "system", "content": system})

    if message != 'clearchat':
        expand_conversation_gpt(message)
        messages = build_messages(conversation_gpt)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=maxtoken  # set max token
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
                persona_p = 'GPT'
            file.write('\n\n'+persona_p+':\n' + response.choices[0].message.content + '\n\n')

        conversation_gpt.append({"role": "assistant", "content": response.choices[0].message.content})

        # Print the assistant's response
        print_mess = message.replace('\r', '\n').replace('\n\n', '\n')

        if printuser: print('user:',print_mess,'\n...')
        print(response.choices[0].message.content)
        #answer = response.choices[0].message.content
        total_tokens = response.usage.total_tokens
        print('prompt tokens:', total_tokens)
