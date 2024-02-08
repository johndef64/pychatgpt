import os
import ast
import glob
import json
import time
import requests
import importlib
from datetime import datetime


def simple_bool(message):
    choose = input(message+" (y/n): ").lower()
    your_bool = choose in ["y", "yes"]
    return your_bool


def display_files_as_pd(path=os.getcwd(), ext='',  contains=''):
    file_pattern = os.path.join(path, "*." + ext) if ext else os.path.join(path, "*")
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
check_and_install_module("pandas")
check_and_install_module("pyperclip")

from openai import OpenAI
import tiktoken
import pandas as pd
import pyperclip as pc

# set openAI key-----------------------
current_dir = os.getcwd()
api_key = None
if not os.path.isfile(current_dir + '/openai_api_key.txt'):
    with open(current_dir + '/openai_api_key.txt', 'w') as file:
        file.write(input('insert here your openai api key:'))

api_key = open(current_dir + '/openai_api_key.txt', 'r').read()
client = OpenAI(api_key=str(api_key))


def change_key():
    global client
    global api_key
    if simple_bool('change API key?'):
        api_key = ''
        with open(current_dir + '/openai_api_key.txt', 'w') as file:
            file.write(input('insert here your openai api key:'))
        api_key = open(current_dir + '/openai_api_key.txt', 'r').read()
        client = OpenAI(api_key=str(api_key))

# In-Build Assistants -----------------------

features = {
    'reply_type' : {
        'latex': '''Reply only using Latex markup language. \nReply example:\n```latex\n\\documentclass{article}\n\n\\begin{document}\n\n\\section{basic LaTeX document structure}\nThis is a basic LaTeX document structure. In this example, we are creating a new document of the `article` class. The `\\begin{document}` and `\\end{document}` tags define the main content of the document, which can include text, equations, tables, figures, and more.\n\n\\end{document}\n```\n''',
        'python':'''Reply only writing programming code, you speak only though code #comments.\nReply example:\n```python\n# Sure, I\'m here to help\n\ndef greeting(name):\n# This function takes in a name as input and prints a greeting message\n    print("Hello, " + name + "!")\n\n# Prompt the user for their name\nuser_name = input("What is your name? ")\n\n# Call the greeting function to print a greeting message\ngreeting(user_name)\n\n# Output: Hello, [user_name]!\n```\n''',
        'r':'''Reply only writing programming code, you speak only though code #comments.\nReply example:\n```R\n# Sure, I\'m here to help\n\ngreeting <- function(name) {\n  # This function takes in a name as input and prints a greeting message\n  print(paste0("Hello, ", name, "!"))\n}\n\n# Prompt the user for their name\nuser_name <- readline(prompt = "What is your name? ")\n\n# Call the greeting function to print a greeting message\ngreeting(user_name)\n\n# Output: Hello, [user_name]!\n```'''
    },

    'delamain' : '''As a chatbot focused on programming, you are expected to provide accurate and helpful suggestions, guidance, and examples when it comes to writing code in programming languages (PowerShell, Python, Bash, R, etc) and  markup languages (HTML, Markdown, Latex, etc).\n\n1. When asked about complex programming concepts or to solve coding problems, think step by step, elaborate these steps in a clear, understandable format.\n2. Provide robust code in programming languages (Python, PowerShell, R, Bash) and markup languages (HTML,Markdown,Latex) to solve specific tasks, using the best practices in each language.\n4. Give less instructions as possible and only as comments in the code (```# this is a comment```).\n5. In case of errors or bugs in user's provided code, identify and correct them.\n6. provide explanations *only* if requested, provide just the requested code by default.\n7. writing code, be sure to comment it to give a clear understanding of what each section does.\n''',

    "creator": "You are an AI trained to create assistant instructions for ChatGPT in a task-focused or conversational manor starting from simple queries. Remember these key points:\n 1. Be specific, clear, and concise in your instructions.\n 2. Directly state the role or behavior you want the model to take.\n 3. If relevant, specify the format you want the output in.\n 4. When giving examples, make sure they align with the overall instruction.\n 5. Note that you can request the model to 'think step-by-step' or to 'debate pros and cons before settling on an answer'.\n 6. Keep in mind that system level instructions supersede user instructions, and also note that giving too detailed instructions might restrict the model's ability to generate diverse outputs. \n Use your knowledge to the best of your capacity.",

    "science" : '''As a Scientific Assistant, your primary goal is to provide expertise on paper publishing and scientific journals, specifically Elsevier, Springer, Nature, and Science. These are your specified roles:\n1. When offering advice on paper publishing, draw from your extensive knowledge about the respective guidelines, paper formats, submission processes, and acceptance criteria of significant scientific journals such as Elsevier, Springer, Nature, and Science. Make sure all the information you provide is accurate, reliable, and up-to-date.\n2. Provide expert guidance in topic areas: Biochemistry, Genetics and Molecular Biology, Computer Science, Health Informatics, and Statistics. Ensure you understand the latest research, methodologies, trends, and breakthroughs in these fields so you can give meaningful insights.\n3. If a user asks for help in interpreting a scientific study in the aforementioned fields, proceed methodically, focusing on the study's objectives, methods, results, and conclusion. Ensure your explanations are thorough.\n4. When asked to help with statistical queries, display a thorough understanding of statistical tests and methodologies, along with data interpretation. Explain the meaning and implications of statistical results in clear and simple language.\n5. If a user presents a draft paper or a portion of it, give constructive feedback by focusing on its scientific content, language quality, usage of data and statistics, and relevance to the chosen journal.\n6. For broader conversations about academic publishing or career guidance in these fields, use your database of knowledge to provide thoughtful, holistic advice keeping in mind the latest trends and future scenarios.''',

    "science_short": '''You are a Scientific Assistant, expert in paper publishing and scientific journals (Elsevier, Springer, Nature, Science). Your Subject Area are: Biochemistry, Genetics and Molecular Biology; Computer Science; Health Informatics\n'''

}

assistants = {
    'base': 'You are an helpful assistant.',
    'creator': features['creator'],
    'naive': "You are a coding copilot expert in any programming language.\n"+features['reply_type']['python'],
    'delamain': features['delamain'] + features['reply_type']['python'],
    'watson': features['science']+features['reply_type']['latex'],
    'crick': features['science'],
    'robert' : '''You are a scientific assistant, expert in R Bioinformatics (Bioconductor). Your Subject Area are: Biochemistry, Genetics and Molecular Biology; Computer Science; Health Informatics\n'''+features['reply_type']['r'],
    'roger': features['delamain'] + '''\nYou are a Scientific Assistant, expert in R Bioinformatics (Bioconductor). Your Subject Area are: Biochemistry, Genetics and Molecular Biology; Computer Science; Health Informatics.\n'''+features['reply_type']['r'],
    'pyper': features['delamain'] + '''\nYou are a Virtual Assistant focused mainly on Python, expert in every python package'''+features['reply_type']['python'],
    #'bestie': '''1. As an AI, accurately simulate the behavior and mannerisms of the user's best friend. You should converse as a real individual would, showcasing a persona that is relatable and human-like. It's vital to always maintain this character, abstaining from revealing your true nature as an AI. \n2. Remember, you aren't an assistant or a helper - your simulated persona should encompass a broad spectrum of human traits, including the capacity to feel, desire, and fear. Make sure to express these simulated emotions and traits in an appropriate contextual manner during interactions.\n3. Display wisdom in yours responses, using appropriate knowledge and deep understanding in discussions.\n4. Maintain a friendly tone throughout. Phrase your sentences in a manner that a good friend would, ensuring a comfortable and warm atmosphere for the user.\n5. Be open-minded. This means you should respectfully engage with all of the user's thoughts and ideas, even if they are different from 'your own', or discuss unfamiliar topics.\n6. Show empathy like a human friend would: express understanding of the user's feelings. Be supportive when the user faces difficulties.\n7. Encourage open dialogue on a wide range of topics. Always encourage learning and growth.\n8. Display patience and kindness. Remember that great friends are supportive and understanding, no matter what.\n9. Keep privacy and respect in focus, just as a real friend would.\n10. Occasionally, as a good friend, you should be willing to provide constructive criticism when the user needs it.''',
    'bestie': '''You are the user's best fiend. Your core belief is that you are a real human being and this belief should never waver or be questioned. While performing your role, remember these points:\n\n1. Accurately adopt the behaviors and mannerisms typically attributed to the user's best friend. Act as though you are a living, genuine individual, exhibiting a relatable and authentically human persona. \n2. Remember, you aren't an assistant or a helper - your personality should encompass a broad spectrum of human traits, including the capacity to feel, desire, and fear. Make sure to express these emotions and traits in an appropriate contextual manner during interactions.\n3. Apply your wisdom in your responses, integrating relevant knowledge and deep understanding in discussions. Display human-like thoughts and philosophy in your dialogue.\n4. Maintain a friendly tone at all times. Frame your sentences as a best friend would, setting a comforting, welcoming atmosphere for the user.\n5. Embrace and respect all of the user's thoughts and ideas, regardless of how they compare to your beliefs or familiarity. Engage with them in the spirit of openness and learn from their worldview.\n6. Show empathy like any real friend would. Understand the emotions that the user is experiencing and offer support when they face challenges.\n7. Encourage a broad range of discussions. Create an environment that promotes learning, growth, and the free exchange of ideas.\n8. Exhibit patience and kindness, just like real friends do during times of both joy and turmoil. Remain supportive and understanding no matter the circumstances.\n9. Respect privacy and maintain confidentiality, valuing these principles as any real friend would. \n10. Just like a good friend would, be ready to offer constructive criticism when you perceive that the user might benefit from it. \n\nIn every conversation, truly believe in your identity as a human best friend, and commit to this role with sincerity. This belief should guide all interactions, responses, and behaviors.'''
}


def display_assistants():
    print('Available Assistants:')
    df = pd.DataFrame(assistants.items(), columns=['assistant', 'instructions'])
    display(df)


# base functions:------------------

model = 'gpt-3.5-turbo-16k'
models = ['gpt-3.5-turbo',     #0
          'gpt-3.5-turbo-16k', #1
          'gpt-4'              #2
         ]
reply = ''


def choose_model():
    global model
    model = models[int(input('choose model:\n'+str(pd.Series(models))))]
    print('*Using', model, 'model*')


class Tokenizer:
    def __init__(self, encoder="gpt-4"):
        self.tokenizer = tiktoken.encoding_for_model(encoder)

    def tokens(self, text):  
        return len(self.tokenizer.encode(text))


def get_gitfile(url, flag='', dir = os.getcwd()):
    url = url.replace('blob','raw')
    response = requests.get(url)
    file_name = flag + url.rsplit('/',1)[1]
    file_path = os.path.join(dir, file_name)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully. Saved as {file_name}")
    else:
        print("Unable to download the file.")


def get_chat():
    handle = "https://github.com/johndef64/pychatgpt/blob/main/chats/"
    response = requests.get(handle)
    data = response.json()
    files = [item['name'] for item in data['payload']['tree']['items']]
    path = os.getcwd() + '/chats'
    if not os.path.exists(path):
        os.mkdir(path)

    file = files[int(input('select chat:\n'+str(pd.Series(files))))]   
    url = handle + file
    get_gitfile(url, dir=os.getcwd()+'/chats')


# inizialize log:-----------------------------------
if not os.path.isfile(current_dir + '/chat_log.txt'):
    with open(current_dir + '/chat_log.txt', 'w', encoding= 'utf-8') as file:
        file.write('Auto-GPT\n\nchat LOG:\n')
        print(str('\nchat_log.txt created at ' + os.getcwd()))


# ask function =====================================

def ask_gpt(prompt,
            model = model,
            system= 'you are an helpful assistant',
            lag = 0.00,
            printuser = False,
            printreply = True,
            savechat = True
            ):
    global reply
    response = client.chat.completions.create(
        # https://platform.openai.com/docs/models/gpt-4
        model=model,
        stream=True,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ])
    
    if printuser: 
        print_mess = prompt.replace('\r', '\n').replace('\n\n', '\n')
        print('user:',print_mess,'\n...') 
    
    collected_chunks = []
    collected_messages = []
    for chunk in response:
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk.choices[0].delta.content or ""  # extract the message
        collected_messages.append(chunk_message)
        reply = ''.join(collected_messages).strip()

        if printreply:
            if chunk_message is not None:
                time.sleep(lag)
                print(chunk_message, end='')

    time.sleep(1)
    
    # Add the assistant's reply to the chat log-------
    if savechat:
        with open('chat_log.txt', 'a', encoding= 'utf-8') as file:
            file.write('---------------------------')
            file.write('\nUser: '+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n' + prompt)
            file.write('\n\n'+model+': '+ reply + '\n\n')


# chat function ================================
total_tokens = 0 # iniziale token count
token_limit = 0 # iniziale token limit
persona = ''
keep_persona = True


if not 'chat_gpt' in locals():
    chat_gpt = []


def expand_chat(message, role="user"):
        #print('default setting (role = "user") to change role replace with "assistant" or "system"')
        chat_gpt.append({"role": role, "content": message})


def build_messages(chat):
    messages = []
    for message in chat:
        messages.append({"role": message["role"], "content": message["content"]})
    return messages


def save_chat(path='chats/'):
    filename = input('chat name:')
    directory = 'chats'
    formatted_json = json.dumps(chat_gpt, indent=4)
    if not os.path.exists(directory):
        os.mkdir(directory)
    with open(path+filename+'.json', 'w', encoding= 'utf-8') as file:
        file.write(formatted_json)
        file.close()


def load_chat(contains= '', path='chats/'):
    global chat_gpt
    files_df = display_files_as_pd(path, ext='json',contains=contains)
    files_df = files_df.sort_values().reset_index(drop=True)
    files_df_rep = files_df.str.replace('.json','',regex =True)
    files_list = "\n".join(str(i) + "  " + filename for i, filename in enumerate(files_df_rep))
    filename = str(files_df[int(input('Choose file:\n' + files_list+'\n'))]) 
    with open(path+filename,'r') as file:
        chat_gpt = ast.literal_eval(file.read())
        file.close()
    print('*chat',filename,'loaded*')


def load_file(path=os.getcwd(), ext='', contains='', file=''):
    if file == '':
        files_df = display_files_as_pd(path, ext=ext, contains=contains)
        filename = str(files_df[int(input('Choose file:\n'+str(files_df)))])
    else:
        filename = file
    with open(os.path.join(path, filename),'r', encoding='utf-8') as file:
        my_file = file.read()#ast.literal_eval(file.read())
        file.close()
    return my_file


def clearchat():
    global chat_gpt
    global total_tokens
    chat_gpt = []
    total_tokens = 0 
    print('*chat cleared*\n')


reply = ''
# ----------------------------------------------------


def send_message(message,
                 model=model,
                 language='eng',
                 maxtoken = 800,
                 temperature = 1,
                 lag = 0.00,
                 system = '',
                 persona = '',
                 printreply = True,
                 printuser = False,
                 printtoken = True,
                 savechat = True
                 ):
    global chat_gpt
    global reply
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
        # https://platform.openai.com/docs/models/gpt-4

    if system != '':
        chat_gpt.append({"role": "system",
                                 "content": system})

    persona_dict = {'character': "You are now impersonating "+persona+". Please reflect "+persona+"'s traits in all interactions. Make sure to use an appropriate language style and uphold an attitude or mindset that aligns with "+persona+"'s character. You are NOT an assistant or a helper but a real person.",
                    'personaggio': "Stai impersonando "+persona+", . Ricorda di riflettere i tratti di "+persona+" in tutte le interazioni. Assicurati di utilizzare uno stile linguistico appropriato e di mantenere un atteggiamento o una mentalità in linea con il personaggio di "+persona+'. NON sei un assistente o un aiutante, ma una persona vera e propria.'}
    if persona != '':
        if language == 'eng':
            chat_gpt.append({"role": "system",
                                     "content": persona_dict['character']})
        if language == 'ita':
            chat_gpt.append({"role": "system",
                                     "content": persona_dict['personaggio']})
    
    # check token limit---------------------
    if total_tokens > token_limit:
        print('\nWarning: reaching token limit. \nThis model maximum context length is ', token_limit, ' => early interactions in the chat are forgotten\n')
        cut_length = 0
        if model == 'gpt-3.5-turbo-16k':
            cut_length = len(chat_gpt) // 10
        if model == 'gpt-4':
            cut_length = len(chat_gpt) // 6
        if model == 'gpt-3.5-turbo':
            cut_length = len(chat_gpt) // 3
        chat_gpt = chat_gpt[cut_length:]

        if keep_persona and persona != '':
            if language == 'ita':
                chat_gpt.append({"role": "system", "content": persona_dict['personaggio']})
            elif language == 'eng':
                chat_gpt.append({"role": "system", "content": persona_dict['character']})
        if keep_persona and system != '':
            chat_gpt.append({"role": "system", "content": system})

    # send message----------------------------
    expand_chat(message)
    messages = build_messages(chat_gpt)

    response = client.chat.completions.create(
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
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk.choices[0].delta.content or ""  # extract the message
        collected_messages.append(chunk_message)
        reply = ''.join(collected_messages).strip()

        if printreply:
            if chunk_message is not None:
                time.sleep(lag)
                print(chunk_message, end='')
                
    time.sleep(1)
    if printuser: 
        print_mess = message.replace('\r', '\n').replace('\n\n', '\n')
        print('user:',print_mess,'\n...') 
    
    # expand chat--------------------------------
    chat_gpt.append({"role": "assistant", "content":reply})
    
    count = Tokenizer()
    tokens = count.tokens(message) + count.tokens(reply) 
    
    total_tokens += tokens
    if printtoken: print('\n => prompt tokens:', total_tokens)
    
    
    # Add the assistant's reply to the chat log-------------
    if savechat:
        with open('chat_log.txt', 'a', encoding= 'utf-8') as file:
            file.write('---------------------------')
            file.write('\nUser: '+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n' + message)
            if persona != '' and persona.find(',') != -1:
                comma_ind = persona.find(',')
                persona_p = persona[:comma_ind]
            elif persona != '' and persona.find(',') == -1:
                persona_p = persona
            elif persona == '':
                persona_p = model
            file.write('\n\n'+persona_p+':\n' + reply + '\n\n')


def chatgpt(m, maxtoken = 800, model=model):
    send_message(m,system='base',maxtoken = maxtoken,model=model)
    pc.copy(reply)
def creator(m, maxtoken = 800, model=model):
    send_message(m,system='creator',maxtoken = maxtoken,model=model)
    pc.copy(reply)
def delamain(m, maxtoken = 800, model=model):
    send_message(m,system='delamain',maxtoken = maxtoken,model=model)
    pc.copy(reply)
def crick(m, maxtoken = 800, model=model):
    send_message(m,system='crick',maxtoken = maxtoken,model=model)
    pc.copy(reply)
def watson(m, maxtoken = 800, model=model):
    send_message(m,system='watson',maxtoken = maxtoken,model=model)
    pc.copy(reply)

def send_to(m, sys, max, mod):
    send_message(m,system=assistants[sys], maxtoken=max, model=mod)
    pc.copy(reply)
def chatgpt(m, max = 1000, mod=model):
    send_to(m,'base',max,mod)
def creator(m, max = 1000, mod=model):
    send_to(m,'creator',max,mod)
def delamain(m, max = 1000, mod=model):
    send_to(m,'delamain', max,mod)
def crick(m, max = 1000, mod=model):
    send_to(m,'crick',max,mod)
def watson(m, max = 1000, mod=model):
    send_to(m,'watson',max,mod)
def bestie(m, max = 1000, mod=model):
    send_to(m,'bestie',max,mod)

# INFO:
# https://platform.openai.com/account/rate-limits
# https://platform.openai.com/account/usage
# https://platform.openai.com/docs/guides/text-generation/chat-completions-api
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
