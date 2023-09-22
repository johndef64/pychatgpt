import os
import time
import importlib
from datetime import datetime


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
    import pyperclip
    import openai
except ImportError:
    print('Check requirements:')
check_and_install_module("pyperclip")
check_and_install_module("openai")
#print('--------------------------------------')

current_dir = ''
# to run in python terminal you need this:
if os.getcwd() == 'C:\WINDOWS\System32':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
else:
    current_dir = os.getcwd()

# If  import openai
import openai
import pyperclip

# check Usage:
# https://platform.openai.com/account/usage

# Set API key:
# https://platform.openai.com/account/api-keys
api_key = None
if not os.path.isfile(current_dir + '/openai_api_key.txt'):
    with open(current_dir + '/openai_api_key.txt', 'w') as file:
        file.write(input('insert here your openai api key:'))

api_key = open(current_dir + '/openai_api_key.txt', 'r').read()
openai.api_key = str(api_key)

add = "You are a helpful assistant."
model = "gpt-4"

#inizialize log:
if not os.path.isfile(current_dir + '/conversation_log.txt'):
    with open(current_dir + '/conversation_log.txt', 'w', encoding='utf-8') as file:
        file.write('Auto-GPT\n\nConversation LOG:\n')
        print(str('\nconversation_log.txt created at ' + os.getcwd()))


# chat functions ----------------------------
#https://platform.openai.com/account/rate-limits
def ask_gpt(prompt):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": add},
            {"role": "user", "content": prompt}
        ])
    with open('conversation_log.txt', 'a', encoding='utf-8') as file:
        file.write('---------------------------')
        if add != '':
            file.write('\nSystem: \n"' + add + '"\n')
        file.write('\nUser: ' + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '\n' + prompt)
        file.write('\n\nGPT:\n' + completion.choices[0].message.content + '\n\n')

    #print('user:',prompt,'\n')
    return print(completion.choices[0].message.content)


# Initialize the conversation (gpt-4)----------------------------

# Conversation history
conversation_gpt = [
    {"role": "system", "content": "You are a helpful assistant expert in science and informatics."},
]
conversation_gpt = []


def expand_conversation_gpt(message):
    conversation_gpt.append({"role": "user", "content": message})


def build_messages(conversation):
    messages = []
    for message in conversation:
        messages.append({"role": message["role"], "content": message["content"]})
    return messages

#The Max Token Limit In ChatGPT: gpt-4 (8,192 tokens), gpt-4-0613 (8,192 tokens), gpt-4-32k (32,768 tokens), gpt-4-32k-0613 (32,768 tokens). maximum limit (4097 tokens for gpt-3.5-turbo )
total_tokens = 0
maxtoken = 1000
prmtoken = 8192 - (maxtoken * 1.3)
keep_persona = True
language = '1'

'''
gpt-3.5-turbo: InvalidRequestError: This model's maximum context length is 4097 tokens. However, you requested 4201 tokens (3201 in the messages, 1000 in the completion). Please reduce the length of the messages or completion.
last token report 2568 (not cosidering response tokens that could be up to 500-700)
'''
persona = ''
system = ''
def send_message_gpt(message, language='eng'):
    global conversation_gpt
    global total_tokens

    if persona != '':
        persona_dict = {'character': "You are now impersonating "+persona+". Please reflect "+persona+"'s traits in all interactions. Make sure to use an appropriate language style and uphold an attitude or mindset that aligns with "+persona+"'s character.",
                        'personaggio': "Stai impersonando "+persona+", . Ricorda di riflettere i tratti di "+persona+" in tutte le interazioni. Assicurati di utilizzare uno stile linguistico appropriato e di mantenere un atteggiamento o una mentalità in linea con il personaggio di "+persona}
        if language == 'eng':
            conversation_gpt.append({"role": "system",
                                     "content": persona_dict['character']})
        if language == 'ita':
            conversation_gpt.append({"role": "system",
                                     "content": persona_dict['personaggio']})
    if system != '':
        conversation_gpt.append({"role": "system",
                                 "content": system})

    if total_tokens > prmtoken:
        print('\n\nWarning: reaching token limit. \nThis model maximum context length is '+ str(prmtoken)+ ' in the messages, ' + str(prmtoken + maxtoken) + ' in total.')
        #print('\n Inizializing new conversation.')
        #conversation_gpt.clear()
        if language == 'eng':
            print('\n the first third of the conversation was forgotten')
        elif language == 'ita':
            print("il primo terzo della conversazione è stato dimenticato")

        if model == 'gpt-3.5-turbo-16k':
            cut_length = len(conversation_gpt) // 10
        if model == 'gpt-4':
            cut_length = len(conversation_gpt) // 6
        if model == 'gpt-3.5-turbo':
            cut_length = len(conversation_gpt) // 3
        conversation_gpt = conversation_gpt[cut_length:]
        if keep_persona:
            if language == 'eng':
                conversation_gpt.append(
                    {"role": "system", "content": persona_dict['character']})
            elif language == 'ita':
                conversation_gpt.append({"role": "system",
                                         "content": persona_dict['personaggio']})

    #send message
    expand_conversation_gpt(message)
    messages = build_messages(conversation_gpt)
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=maxtoken  # set max token
    )

    # Add the assistant's reply to the conversation
    conversation_gpt.append({"role": "assistant", "content": response.choices[0].message.content})

    # write reply in log
    with open('conversation_log.txt', 'a', encoding='utf-8') as file:
        file.write('---------------------------')
        file.write('\nUser: ' + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '\n' + message)
        if persona != '' and persona.find(',') != -1:
            comma_ind = persona.find(',')
            persona_p = persona[:comma_ind]
        elif persona != '' and persona.find(',') == -1:
            persona_p = persona
        elif persona == '':
            persona_p = 'GPT'
        file.write('\n\n' + persona_p + ':\n' + response.choices[0].message.content + '\n\n')

    # Print the assistant's response
    #print_mess = message.replace('\r', '\n').replace('\n\n', '\n')
    #print('user:',print_mess,'\n...')
    answer = response.choices[0].message.content

    if persona != '':
        print('\n'+persona+':', answer)
    else:
        print('\ngpt-4:', answer)
    total_tokens = response.usage.total_tokens
    print('(token count: '+str(total_tokens)+')')


#-----------------------------------------------------
assistant = ''
# Run script:
while True:  # external cycle
    safe_word = ''
    print(
        '''---------------------\nWelcome to Bash-GPT!\n\nGPT-4 will answer every question.\n\nReply with:\n- 'restartnow' to start over the application.\n- 'exitnow' to shut down the application.\n- 'maxtoken' to set up max token in response (chat mode).\n- 'system' to set new system instructions' to change system instructions (instruct mode)'\n\nwritten by JohnDef64\n---------------------\n''')
    language = '1'  # setting Italian default temporarly
    while language not in ['1', '2']:
        language = input('\nPlease choose language: \n1. English\n2. Italian\n\nLanguage number:')
        if language not in ['1', '2']:
            print("Invalid choice.")
    #----------
    if language == '1':
        choose = ""
        while choose not in ['1', '2', '3', '4']:
            # Prompt the user to make a choice between three predefined strings
            choose = input(
                '''\nChoose settings:\n1. Ask to gpt-4\n2. Ask to gpt-4 (instruct)\n3. Ask to someone\n4. Chat with gpt-4\n\nSetting number:''')

            # Verify user's choice
            if choose == '1':
                with open('conversation_log.txt', 'a') as file:
                    file.write('\n' + str(datetime.now()) + ": <You are asking the virtual assistant>\n")
                print("You chose option 1.")

            elif choose == '2':
                add = input('''
                            Enter system instructions.
                            examples:
                             - Answer the question and give only the correct answer:
                             - Correct this code:

                              Instruction:''')
                with open('conversation_log.txt', 'a') as file:
                    file.write(
                        '\n' + str(datetime.now()) + ": <You are asking the virtual assistant, system: " + add + "> \n")
                print("You chose option 2.")

            elif choose == '3':
                persona = input('Tell me who you want to talk to:')
                add = "**instruction: you are " + persona + ", answer like you are " + persona + "** \n"
                with open('conversation_log.txt', 'a') as file:
                    file.write('\n' + str(datetime.now()) + ': <You asked to ' + persona + '>\n')
                print("I'm connecting you with " + persona)

            elif choose == '4':
                print("You chose to have some conversation.")
                assistant = input(
                    "\nWho do you want to converse with?: \n1. the assistant\n2. the poet \n3. the scientist\n4. someone else\n\nNumber: ")

                if assistant == '1':
                    conversation_gpt.append({"role": "system",
                                             "content": "Today, you are my assistant expert in everything. Answer in accordance with this."})
                    with open('conversation_log.txt', 'a') as file:
                        file.write('\n' + str(datetime.now()) + ': <You are chatting with the general assistant>\n')

                elif assistant == '2':
                    conversation_gpt.append({"role": "system",
                                             "content": "Today, you are the greatest poet ever, inspired, profound, sensitive, visionary and creative. Answer in accordance with this."})
                    with open('conversation_log.txt', 'a') as file:
                        file.write('\n' + str(datetime.now()) + ': <You are chatting with the greatest poet ever>\n')

                elif assistant == '3':
                    conversation_gpt.append({"role": "system",
                                             "content": "Today, you are the greatest and most experienced scientist ever, analytical, precise and rational. Answer in accordance with this."})
                    with open('conversation_log.txt', 'a') as file:
                        file.write(
                            '\n' + str(datetime.now()) + ': <You are chatting with the greatest  scientist ever>\n')
                else:
                    persona = input('Tell me who you want to talk to:')
                    persona_dict = {'character': "You are now impersonating "+persona+". Please reflect "+persona+"'s traits in all interactions. Make sure to use an appropriate language style and uphold an attitude or mindset that aligns with "+persona+"'s character.",
                                    'personaggio': "Stai impersonando "+persona+", . Ricorda di riflettere i tratti di "+persona+" in tutte le interazioni. Assicurati di utilizzare uno stile linguistico appropriato e di mantenere un atteggiamento o una mentalità in linea con il personaggio di "+persona}
                    if language == 'eng':
                        conversation_gpt.append({"role": "system",
                                                 "content": persona_dict['character']})
                    if language == 'ita':
                        conversation_gpt.append({"role": "system",
                                                 "content": persona_dict['personaggio']})
                    with open('conversation_log.txt', 'a') as file:
                        file.write('\n' + str(datetime.now()) + ': <You are chatting with ' + persona + '>\n')
            else:
                print("Invalid choice.")


        while safe_word != 'restartnow' or 'exitnow' or 'maxtoken':
            timea = datetime.now()
            print('\n--------------------------------\n')
            message = input('user:')
            safe_word = message

            if safe_word == 'restartnow':
                break
            if safe_word == 'exitnow':
                exit()
            if safe_word == 'maxtoken':
                maxtoken = int(input('set max response tokens (1000 default):'))
                break

            if choose == '4':
                if safe_word == 'system':
                    conversation_gpt = []
                    system = input('define custum system instructions:')
                    print('*system instruction changed*')
                    pass
                else:
                    send_message_gpt(message)
            elif choose != '4':
                ask_gpt(message)
            else:
                pass
            timed = datetime.now() - timea

            #Rate-Limit gpt-4 = 200
            #https://medium.com/@pankaj_pandey/understanding-the-chatgpt-api-key-information-and-frequently-asked-questions-4a0e963fb138#:~:text=The%20ChatGPT%20API%20has%20different,90000%20TPM%20after%2048%20hours.
            #https://platform.openai.com/docs/guides/rate-limits/overview
            #https://platform.openai.com/account/rate-limits
            time.sleep(1)  # Wait 1 second before checking again

