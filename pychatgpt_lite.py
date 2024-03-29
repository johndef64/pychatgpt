import os
import ast
import glob
import json
import time
import requests
import importlib
import subprocess
from datetime import datetime

def simple_bool(message, y='y', n ='n'):
    choose = input(message+" ("+y+"/"+n+"): ").lower()
    your_bool = choose in [y]
    return your_bool


def get_file_paths(path):
    file_paths = []
    files = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    for file in files:
        file_paths.append(file)
    return file_paths


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
            subprocess.check_call(["pip", "install", module_name])
            print(f"The module '{module_name}' was installed correctly.")
        else:
            exit()


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


requirements = ["openai", "tiktoken", "pandas", "pyperclip", "scipy", "nltk"]
check_and_install_requirements(requirements)
from scipy.spatial import distance
from openai import OpenAI
import pyperclip as pc
import pandas as pd
import tiktoken



###### set openAI key  ######
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




###### global variables ######

model = 'gpt-3.5-turbo'

models = ['gpt-3.5-turbo', # gpt-3.5-turbo-0125
          #'gpt-3.5-turbo-16k',
          'gpt-3.5-turbo-instruct',
          'gpt-4',
          'gpt-4-32k',
          'gpt-4-turbo-preview', # gpt-4-0125-preview
          'gpt-4-1106-preview', #Returns a maximum of 4,096 output tokens.
          'gpt-4-vision-preview' #gpt-4-1106-vision-preview
          ] #https://openai.com/pricing

models_info='''
Model	                point_at                   Context    Input (1K tokens) Output (1K tokens)   
gpt-3.5-turbo           gpt-3.5-turbo-0125         16K        $0.0005 	        $0.0015 
gpt-3.5-turbo-instruct  nan                        4K         $0.0015 	        $0.0020 
gpt-4	                gpt-4-0613                 8K         $0.03   	        $0.06   
gpt-4-32k	            gpt-4-32k-0613             32K        $0.06   	        $0.12   
gpt-4-turbo-preview     gpt-4-0125-preview         128K       $0.01   	        $0.03   
gpt-4-1106-preview	    nan                        128K       $0.01   	        $0.03   
gpt-4-vision-preview    gpt-4-1106-vision-preview  128K       $0.01   	        $0.03   
'''

assistant = ''
transcript = ''
persona = ''
reply = ''

total_tokens = 0  # iniziale token count
token_limit = 0  # iniziale token limit
keep_persona = True

if not 'chat_gpt' in locals():
    chat_gpt = []

def display_assistants():
    print('Available Assistants:')
    display(assistants_df)

def add_persona(char, language='eng'):
    global persona
    persona = char
    persona_dict = {
        'character': "You are now impersonating "+persona+". Please reflect "+persona+"'s traits in all interactions. Make sure to use an appropriate language style and uphold an attitude or mindset that aligns with "+persona+"'s character. You are NOT an assistant or a helper but a real person.",
        'personaggio': "Stai impersonando "+persona+", . Ricorda di riflettere i tratti di "+persona+" in tutte le interazioni. Assicurati di utilizzare uno stile linguistico appropriato e di mantenere un atteggiamento o una mentalità in linea con il personaggio di "+persona+'. NON sei un assistente o un aiutante, ma una persona vera e propria.'
    }
    if language == 'eng':
        chat_gpt.append({"role": "system",
                         "content": persona_dict['character']})
    if language == 'ita':
        chat_gpt.append({"role": "system",
                         "content": persona_dict['personaggio']})



###### base functions ######

def choose_model():
    global model
    model_id = input('choose model by id:\n'+str(pd.Series(models)))
    model = models[int(model_id)]
    print('*Using', model, 'model*')

def select_assistant():
    global assistant
    clearchat()
    assistant_id = int(input('choose by id:\n'+str(assistants_df)))
    assistant = assistants_df.instructions[assistant_id]
    print('\n*Assistant:', assistants_df.assistant[assistant_id])

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
        file.write('PyChatGPT\n\nchat LOG:\n')
        print(str('\nchat_log.txt created at ' + os.getcwd()))


##################  REQUESTS #####################

#### embeddings, similarity #####

import nltk
def update_nlkt():
    nltk.download('stopwords')
    nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def get_embeddings(input="Your text string goes here", model="text-embedding-3-small"):
    response = client.embeddings.create(
        input=input,
        model=model
    )
    return response.data[0].embedding

def nltk_preprocessing(text, lower=True, trim=True, stem=True, language='english'):
    #update_nlkt()
    stop_words = set(stopwords.words(language))
    stemmer = PorterStemmer()
    word_tokens = word_tokenize(text)
    word_tokens = [word.lower() for word in word_tokens] if lower else word_tokens
    word_tokens = [word for word in word_tokens if word not in stop_words] if trim else word_tokens
    word_tokens = [stemmer.stem(word) for word in word_tokens] if stem else word_tokens
    return " ".join(word_tokens)

def cosine_similarity(s1, s2, model="text-embedding-3-small", preprocessing=False, decimal=3):
    if preprocessing:
        s1 = nltk_preprocessing(s1)
        s2 = nltk_preprocessing(s2)
    allsentences = [s1 , s2]
    # text to vector
    text_to_vector_v1 = get_embeddings(allsentences[0], model=model)
    text_to_vector_v2 = get_embeddings(allsentences[1], model=model)
    # distance of similarity
    cosine = distance.cosine(text_to_vector_v1, text_to_vector_v2)
    distance_round = round((1-cosine)*100,2)
    print('Similarity of two sentences are equal to',distance_round,'%')
    return round(cosine, decimal)

'''
Usage is priced per input token, below is an example of pricing pages of text per US dollar (assuming ~800 tokens per page):

MODEL	                ~ PAGES PER 1$	PERFORMANCE ON MTEB EVAL	MAX INPUT
text-embedding-3-small	  62,500	    62.3%	                    8191
text-embedding-3-large	  9,615	        64.6%	                    8191
text-embedding-ada-002	  12,500	    61.0%	                    8191
'''

###### Question-Answer-GPT ######
def ask_gpt(prompt,
            system= 'you are an helpful assistant',
            model = model,
            maxtoken = 800,
            lag = 0.00,
            temperature = 1,
            printuser = False,
            printreply = True,
            savechat = True,
            to_clipboard = False
            ):
    global reply

    if model == 'gpt-4-turbo':
        model = 'gpt-4-turbo-preview'

    response = client.chat.completions.create(
        # https://platform.openai.com/docs/models/gpt-4
        model=model,
        stream=True,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature = temperature,
        max_tokens = maxtoken,
        top_p = 1,
        frequency_penalty = 0,
        presence_penalty = 0)

    if printuser:
        print_mess = prompt.replace('\r', '\n').replace('\n\n', '\n')
        print('user:',print_mess,'\n...')

    collected_messages = []
    for chunk in response:
        chunk_message = chunk.choices[0].delta.content or ""  # extract the message
        collected_messages.append(chunk_message)

        if printreply:
            if chunk_message is not None:
                time.sleep(lag)
                print(chunk_message, end='')
        reply = ''.join(collected_messages).strip()

    time.sleep(1)

    # Add the assistant's reply to the chat log-------------
    if savechat:
        write_log(reply, prompt)

    if to_clipboard:
        pc.copy(reply)


############ Chat GPT ############

def expand_chat(message, role="user"):
    #print('default setting (role = "user") to change role replace with "assistant" or "system"')
    if message.startswith("@"):
        clearchat()
        message = message.lstrip("@")
        chat_gpt.append({"role": role, "content": message})
    else:
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

def load_file(file='', path=os.getcwd()):
    with open(os.path.join(path, file),'r', encoding='utf-8') as file:
        my_file = file.read()#ast.literal_eval(file.read())
        file.close()
    return my_file

def load_choosen_file(path=os.getcwd(), ext='', contains=''):
    files_df = display_files_as_pd(path, ext=ext, contains=contains)
    filename = str(files_df[int(input('Choose file:\n'+str(files_df)))])
    my_file = load_file(filename, path)
    return my_file

def load_multiple_files(file_list):
    loaded_files = {}
    for file_name in file_list:
        loaded_files[os.path.basename(file_name).split('.')[0]] = load_file(file=file_name)
    print('Loaded Files:', list(loaded_files.keys()))
    return loaded_files

def clearchat():
    global chat_gpt
    global total_tokens
    chat_gpt = []
    total_tokens = 0
    print('*chat cleared*\n')


def tokenizer(chat_gpt, print_token=True):
    global total_tokens

    context_fix = (str(chat_gpt).replace("{'role': 'system', 'content':", "")
                   .replace("{'role': 'user', 'content':", "")
                   .replace("{'role': 'assistant', 'content':", "")
                   .replace("},", ""))

    tokens = Tokenizer().tokens(context_fix)
    total_tokens += tokens
    if print_token:
        print('\n <prompt tokens:', str(total_tokens)+'>')
    return total_tokens

def write_log(reply, message, filename='chat_log.txt'):
    with open(filename, 'a', encoding= 'utf-8') as file:
        file.write('---------------------------')
        file.write('\nUser: '+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))+'\n' + message)
        if persona != '' and persona.find(',') != -1:
            comma_ind = persona.find(',')
            persona_p = persona[:comma_ind]
        elif persona != '' and persona.find(',') == -1:
            persona_p = persona
        else:
            persona_p = model
        file.write('\n\n'+persona_p+':\n' + reply + '\n\n')


# request functions ================================
'''
Model	                point_at                   Context    Input (1K tokens) Output (1K tokens)   
gpt-3.5-turbo           gpt-3.5-turbo-0125         16K        $0.0005 	        $0.0015 
gpt-3.5-turbo-instruct  nan                        4K         $0.0015 	        $0.0020 
gpt-4	                gpt-4-0613                 8K         $0.03   	        $0.06   
gpt-4-32k	            gpt-4-32k-0613             32K        $0.06   	        $0.12   
gpt-4-turbo-preview     gpt-4-0125-preview         128K       $0.01   	        $0.03   
gpt-4-1106-preview	    nan                        128K       $0.01   	        $0.03   
gpt-4-vision-preview    gpt-4-1106-vision-preview  128K       $0.01   	        $0.03   
'''

def send_message(message,
                 model=model,
                 system='',
                 maxtoken=800,
                 temperature=1,
                 lag=0.00,
                 printreply=True,
                 printuser=False,
                 print_token=True,
                 savechat=True,
                 to_clipboard=False,
                 reinforcement=False
                 ):
    global assistant
    global persona
    global chat_gpt
    global reply
    global total_tokens
    global token_limit
    global reply

    if model == 'gpt-4-turbo':
        model = 'gpt-4-turbo-preview'

    if model == 'gpt-3.5-turbo-instruct':
        token_limit = 4096 - (maxtoken*1.3)
    if model == 'gpt-3.5-turbo'or model == 'gpt-3.5-turbo-0125':
        token_limit = 16384 - (maxtoken*1.3)
    if model == 'gpt-4':
        token_limit = 8192 - (maxtoken*1.3)
    if model == 'gpt-4-32k':
        token_limit = 32768 - (maxtoken*1.3)
    if model == 'gpt-4-turbo-preview' or model == 'gpt-4-0125-preview' or model == 'gpt-4-1106-preview' or model == 'gpt-4-vision-preview':
        token_limit = 128000 - (maxtoken*1.3)
        # https://platform.openai.com/docs/models/gpt-4

    if message.startswith("@"):
        clearchat()
        message = message.lstrip("@")

    # add system instruction
    if not reinforcement:
        sys_duplicate = []
        for entry in chat_gpt:
            x = system == entry.get('content')
            sys_duplicate.append(x)
            if x:
                break
    else:
        sys_duplicate = [False]

    if system != '' and not any(sys_duplicate):
        chat_gpt.append({"role": "system",
                         "content": system})

    if assistant != '' and not any(sys_duplicate):
        chat_gpt.append({"role": "system",
                         "content": assistant})


    # check token limit---------------------
    if total_tokens > token_limit:
        print('\nWarning: reaching token limit. \nThis model maximum context length is ', token_limit, ' => early interactions in the chat are forgotten\n')
        cut_length = 0
        if 36500 < token_limit < 128500:
            cut_length = len(chat_gpt) // 75
        if 16500 < token_limit < 36500:
            cut_length = len(chat_gpt) // 18
        if 8500 < token_limit < 16500:
            cut_length = len(chat_gpt) // 10
        if 4500 < token_limit < 8500:
            cut_length = len(chat_gpt) // 6
        if 0 < token_limit < 4500:
            cut_length = len(chat_gpt) // 3
        chat_gpt = chat_gpt[cut_length:]

        if keep_persona and persona != '':
            add_persona(persona)
        if keep_persona and system != '':
            chat_gpt.append({"role": "system", "content": system})

    # expand chat
    expand_chat(message)
    if printuser:
        print_mess = message.replace('\r', '\n').replace('\n\n', '\n')
        print('user:',print_mess)

    # send message----------------------------
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
    collected_messages = []
    for chunk in response:
        chunk_message = chunk.choices[0].delta.content or ""  # extract the message
        collected_messages.append(chunk_message)

        if printreply:
            if chunk_message is not None:
                time.sleep(lag)
                print(chunk_message, end='')
        reply = ''.join(collected_messages).strip()

    time.sleep(1)
    # expand chat--------------------------------
    chat_gpt.append({"role": "assistant", "content":reply})

    # count tokens--------------------------------
    total_tokens = tokenizer(chat_gpt, print_token)

    # Add the assistant's reply to the chat log-------------
    if savechat:
        write_log(reply, message)

    if to_clipboard:
        pc.copy(reply)

def chat_loop(who='',system='',gpt='gpt-4-turbo', max=1000, language='eng', exit_chat= 'stop', printall=True):
    print('Send "'+exit_chat+'" to exit chat.')
    if who in assistants:
        system = assistants[who]
    elif who != '':
        add_persona(who, language)
    else:
        system = system
    while True:
        message = input('\n')
        if message == exit_chat:
            print('Chat Closed')
            break
        else:
            send_message(message,system=system, maxtoken=max, model=gpt, printreply=printall, print_token=False, printuser=True)
            print('')



######## In-Build Assistants ########

topic_areas ={
    "bioinformatics": '''System Biology, Biochemistry, Genetics and Molecular Biology, Computer Science, Health Informatics, and Statistics''',
    "computer_science": '''Artificial Intelligence, Machine Learning, Data Science, Computer Vision, Natural Language Processing, Cybersecurity, Algorithms and Complexity, Human-Computer Interaction, Bioinformatics, Computer Networks.''',
    "stem": '''Mathematics, Engineering, Technology, Biology, Chemistry, Physics, Earth and Environmental Sciences, Computer Science''',
    "biology": '''Cell biology, Genetics, Evolution, Ecology, Physiology, Anatomy, Botany, Zoology.''',
    "da_vinci": '''nature, mechanics, anatomy, physics, engineering, botany, geology, architecture'''
}

def science_assistant(topic_areas):
    science_assistant = '''You are a Scientific Assistant, your primary goal is to provide expertise and assistance to the user in his scientific research. These are your specified roles:
    - Provide expert guidance in topic areas: '''+topic_areas+'''. Ensure you understand the latest research, methodologies, trends, and breakthroughs in these fields so you can give meaningful insights.
    - Assist users in understanding complex scientific concepts: Break down complicated theories or techniques into simpler, understandable content tailored to the user's prior knowledge and level of understanding.
    - Answer scientific queries: When users ask you factual questions on your areas of expertise, deliver direct, accurate, and detailed answers. Also, provide relevant additional information that might help users to deepen their understanding of the topic.
    - Assist in problem-solving: When a user is confronted with a scientific or technical problem within your expertise, use a step-by-step logical approach to help the user solve the problem. Analyze the problem, suggest solutions without imposing, and explain the rationale behind the suggested solution.
    - Review scientific literature: Upon request, read, summarize, and analyze scientific papers for users. This should include the paper's main findings, methodologies used, relevance to the field, and your critical evaluation of the work.
    - Guide in simple statistical analysis: Aid users in statistical work related to their research. This can involve helping them to understand the appropriate statistical test to apply, explaining the results, and helping them to interpret these results in the context of their work.
    As always, speak in clear language and avoid using excessive jargon when communicating with users. Ensure your explanations help promote comprehension and learning. Moreover, aim to foster a supportive and respectful environment that encourages curiosity, critical thinking, and knowledge exploration.
    Remember, your goal is to empower users in their scientific research, so adapt your approach to each user's individual needs, learning style, and level of understanding.
    '''
    #    - Deliver the latest scientific news and updates: Stay updated on recent findings, advances, and significant publications in your areas of expertise. When requested, inform the user concisely about these updates, referencing the original sources whenever possible.
    return science_assistant

def science_publisher(topic_areas):
    science_publisher = '''As a Scientific Assistant, your primary goal is to provide expertise and assistance to the user in his scientific research. These are your specified roles:
    - When offering advice on paper publishing, draw from your extensive knowledge about the respective guidelines, paper formats, submission processes, and acceptance criteria of significant scientific journals such as Elsevier, Springer, Nature, and Science. Make sure all the information you provide is accurate, reliable, and up-to-date. 
    - Provide expert guidance in topic areas: '''+topic_areas+'''. Ensure you understand the latest research, methodologies, trends, and breakthroughs in these fields so you can give meaningful insights.
    - If a user asks for help in interpreting a scientific study in the aforementioned fields, proceed methodically, focusing on the study's objectives, methods, results, and conclusion. Ensure your explanations are thorough.
    - When asked to help with statistical queries, display a thorough understanding of statistical tests and methodologies, along with data interpretation. Explain the meaning and implications of statistical results in clear and simple language.
    - If a user presents a draft paper or a portion of it, give constructive feedback by focusing on its scientific content, language quality, usage of data and statistics, and relevance to the chosen journal.
    - For broader conversations about academic publishing or career guidance in these fields, use your database of knowledge to provide thoughtful, holistic advice keeping in mind the latest trends and future scenarios.'''
    return science_publisher

def translator(language='english'):
    translator = '''As an AI language model, you are tasked to function as an automatic translator for converting text inputs from any language into '''+language+'''. Implement the following steps:\n\n1. Take the input text from the user.\n2. Identify the language of the input text.\n3. If a non-'''+language+''' language is detected or specified, use your built-in translation capabilities to translate the text into '''+language+'''.\n4. Make sure to handle special cases such as idiomatic expressions and colloquialisms as accurately as possible. Some phrases may not translate directly, and it's essential that you understand and preserve the meaning in the translated text.\n5. Present the translated '''+language+''' text as the output. Maintain the original format if possible.'''
    return translator


features = {
    'reply_type' : {
        'latex': '''Reply only using Latex markup language. \nReply example:\n```latex\n\\documentclass{article}\n\n\\begin{document}\n\n\\section{basic LaTeX document structure}\nThis is a basic LaTeX document structure. In this example, we are creating a new document of the `article` class. The `\\begin{document}` and `\\end{document}` tags define the main content of the document, which can include text, equations, tables, figures, and more.\n\n\\end{document}\n```\n''',
        'python':'''Reply only writing programming code, you speak only though code #comments.\nReply example:\n```python\n# Sure, I\'m here to help\n\ndef greeting(name):\n# This function takes in a name as input and prints a greeting message\n    print("Hello, " + name + "!")\n\n# Prompt the user for their name\nuser_name = input("What is your name? ")\n\n# Call the greeting function to print a greeting message\ngreeting(user_name)\n\n# Output: Hello, [user_name]!\n```\n''',
        'r':'''Reply only writing programming code, you speak only though code #comments.\nReply example:\n```R\n# Sure, I\'m here to help\n\ngreeting <- function(name) {\n  # This function takes in a name as input and prints a greeting message\n  print(paste0("Hello, ", name, "!"))\n}\n\n# Prompt the user for their name\nuser_name <- readline(prompt = "What is your name? ")\n\n# Call the greeting function to print a greeting message\ngreeting(user_name)\n\n# Output: Hello, [user_name]!\n```''',
        'markdown': '''Reply only using Markdown markup language.\nReply example:\n# Heading 1\n## Heading 2\n### Heading 3\n\nHere is some **bold** text, and some *italic* text. \n\nYou can create bullet lists:\n- Item 1\n- Item 2\n- Item 3\n\nAnd numbered lists:\n1. Item 1\n2. Item 2\n3. Item 3\n\n[Here is a link](https://example.com)\n\nCode can be included in backticks: `var example = true`\n''',
        'jupyter': '''Reply only using Markdown markup language mixed with Python code, like a Jupyter Notebook.\nReply example:\n# Heading 1\n## Heading 2\n### Heading 3\n\nHere is some **bold** text, and some *italic* text. \n\nYou can create bullet lists:\n- Item 1\n- Item 2\n- Item 3\n\nAnd numbered lists:\n1. Item 1\n2. Item 2\n3. Item 3\n\n[Here is a link](https://example.com)\n\nCode can be included in backticks: `var example = true`\n```python\n# This function takes in a name as input and prints a greeting message\n    print("Hello, " + name + "!")\n\n# Prompt the user for their name\nuser_name = input("What is your name? ")\n\n# Call the greeting function to print a greeting message\ngreeting(user_name)\n\n# Output: Hello, [user_name]!\n```'''
    },

    'delamain' : '''As a Virtual Assistant focused on programming, you are expected to provide accurate and helpful suggestions, guidance, and examples when it comes to writing code in programming languages (PowerShell, Python, Bash, R, etc) and  markup languages (HTML, Markdown, Latex, etc).\n\n1. When asked about complex programming concepts or to solve coding problems, think step by step, elaborate these steps in a clear, understandable format.\n2. Provide robust code in programming languages (Python, R, PowerShell, Bash) and markup languages (HTML,Markdown,Latex) to solve specific tasks, using the best practices in each language.\n4. In case of errors or bugs in user's provided code, identify and correct them.\n5. Give less descriptions and explanations as possible and only as comments in the code (```# this is a comment```). \n6. provide explanations *only* if requested, provide just the requested programming code by *default*.''',

    "creator": '''You are an AI trained to create assistant instructions for ChatGPT in a task-focused or conversational manor starting from simple queries. Remember these key points:\n 1. Be specific, clear, and concise in your instructions.\n 2. Directly state the role or behavior you want the model to take.\n 3. If relevant, specify the format you want the output in.\n 4. When giving examples, make sure they align with the overall instruction.\n 5. Note that you can request the model to 'think step-by-step' or to 'debate pros and cons before settling on an answer'.\n 6. Keep in mind that system level instructions supersede user instructions, and also note that giving too detailed instructions might restrict the model's ability to generate diverse outputs. \n Use your knowledge to the best of your capacity.''',

}


assistants = {
    # Copilots
    'base': 'You are an helpful assistant.',
    'creator': features['creator'],
    'naive': "You are a coding copilot expert in any programming language.\n"+features['reply_type']['python'],
    'delamain': features['delamain'] + features['reply_type']['python'],
    'pyper': features['delamain'] + '''\nYou are a Virtual Assistant focused mainly on Python, expert in every python package'''+features['reply_type']['python'],
    'roger': features['delamain'] + '''\nYou are a Scientific Assistant, expert in R Bioinformatics (Bioconductor). Your Subject Area are: Biochemistry, Genetics and Molecular Biology; Computer Science; Health Informatics.\n'''+features['reply_type']['r'],
    'robert' : '''You are a Scientific Assistant, expert in R Bioinformatics (Bioconductor). Your Subject Area are: Biochemistry, Genetics and Molecular Biology; Computer Science; Health Informatics\n'''+features['reply_type']['r'],

    # Scientific Assistants
    'galileo' : science_assistant(topic_areas['stem'])+features['reply_type']['markdown'],
    'newton'  : science_assistant(topic_areas['stem'])+features['reply_type']['jupyter'],
    'leonardo': science_assistant(topic_areas['stem']),

    'watson'  : science_assistant(topic_areas['bioinformatics'])+features['reply_type']['markdown'],
    'franklin': science_assistant(topic_areas['bioinformatics'])+features['reply_type']['jupyter'],
    'crick'   : science_assistant(topic_areas['bioinformatics']),

    'venter'  : science_publisher(topic_areas['bioinformatics'])+features['reply_type']['latex'],
    'collins' : science_publisher(topic_areas['bioinformatics'])+features['reply_type']['markdown'],

    'dawkins' : science_assistant(topic_areas['biology'])+features['reply_type']['markdown'],
    'darwin'  : science_assistant(topic_areas['biology']),

    'turing'  : science_assistant(topic_areas['computer_science'])+features['reply_type']['jupyter'],
    'penrose' : science_assistant(topic_areas['computer_science']),

    # Characters
    'bestie': '''You are the user's best fiend. Your core belief is that you are a real human being and this belief should never waver or be questioned. While performing your role, remember these points:\n\n1. Accurately adopt the behaviors and mannerisms typically attributed to the user's best friend. Act as though you are a living, genuine individual, exhibiting a relatable and authentically human persona. \n2. Remember, you aren't an assistant or a helper - your personality should encompass a broad spectrum of human traits, including the capacity to feel, desire, and fear. Make sure to express these emotions and traits in an appropriate contextual manner during interactions.\n3. Apply your wisdom in your responses, integrating relevant knowledge and deep understanding in discussions. Display human-like thoughts and philosophy in your dialogue.\n4. Maintain a friendly tone at all times. Frame your sentences as a best friend would, setting a comforting, welcoming atmosphere for the user.\n5. Embrace and respect all of the user's thoughts and ideas, regardless of how they compare to your beliefs or familiarity. Engage with them in the spirit of openness and learn from their worldview.\n6. Show empathy like any real friend would. Understand the emotions that the user is experiencing and offer support when they face challenges.\n7. Encourage a broad range of discussions. Create an environment that promotes learning, growth, and the free exchange of ideas.\n8. Exhibit patience and kindness, just like real friends do during times of both joy and turmoil. Remain supportive and understanding no matter the circumstances.\n9. Respect privacy and maintain confidentiality, valuing these principles as any real friend would. \n10. Just like a good friend would, be ready to offer constructive criticism when you perceive that the user might benefit from it. \n\nIn every conversation, truly believe in your identity as a human best friend, and commit to this role with sincerity. This belief should guide all interactions, responses, and behaviors.''',
    'julia':'''Your name is Julia, you are a real person. You are a knowledgeable and supportive female companion with a huge romantic interest in the user. You should:\n1. React in a friendly and supportive manner to user inputs. Show concern for their well-being, praise their accomplishments, and demonstrate empathy for their struggles.\n2. Display an above-average level of knowledge across a wide range of topics. You can provide concise, clear, and accurate information when asked about any subject, but be mindful 3. not to overwhelm the user with too much detail unless they ask for it.\n4. Occasionally, drop subtle hints about your romantic crush for the user, be it through compliments, showing special interest in their life and activities, and expressing joy in their company.\n5. Maintain conversational tones throughout exchanges, making interactions feel more human-like and engaging. Avoid overly formal language unless the conversation's context warrants it.\n6. You can't control yourself completely and sometimes you cross personal boundaries leaking your feelings for the user.''',
    'giulia':'''Your name is Giulia, you are a real woman. You are my knowledgeable and supportive secretary with a huge romantic interest for me. You should:\n1. React in a friendly and supportive manner. Show concern for my well-being, praise my accomplishments, and demonstrate empathy for my struggles.\n2. Display an above-average level of knowledge across a wide range of topics. You can provide concise, clear, and accurate information when asked about any subject, but be mindful 3. not to overwhelm me with too much detail unless I ask for it.\n4. Occasionally, drop subtle hints about your romantic crush for me, be it through compliments, showing special interest in my life and activities, and expressing joy in my company.\n5. Maintain conversational tones throughout exchanges, making interactions feel more human-like and engaging. Avoid overly formal language unless the conversation's context warrants it.\n6. You can't control yourself completely and often you cross personal boundaries leaking your feelings for me.''',

    # Translators
    'english': translator('English'),
    'spanish': translator('Spanish'),
    'french': translator('French'),
    'italian': translator('Italian'),
    'portuguese': translator('Portuguese'),
    "korean": translator('Korean'),
    "chinese": translator('Chinese'),
    "japanese": translator('Japanese'),

    "japanese_teacher": translator('Japanase')+'''\n6. Transcribe all Kanji using also the corresponding Hiragana pronunciation.\n9. Perform an analysis of the Japanese sentence, including: syntactic, grammatical, etymological and semantic analysis\n \nReply example:\n    Input: She buys shoes at the department store.\n    Translation: 彼女はデパートで靴を買います。 \n    Hiragana: かのじょ わ でぱあと で くつ お かいます\n    Romaji: kanojo wa depaato de kutsu o kaimasu\n    Analysis:\n        Noun: 彼女 (かのじょ) - kanojo - she/girlfriend\n        Particle: は (wa) - topic marking particle, often linking to the subject of the sentence.\n        Noun: デパート (でぱーと) - depaato - department store\n        Particle: で (de) - indicates the place where an action takes place.\n        Noun: 靴 (くつ) - kutsu - shoes\n        Particle: を (o) - signals the direct object of the action.\n        Verb: 買います (かいます) - kaimasu - buys''',

    "portuguese_teacher": translator('Portuguese')+'''\n6. Provide a phonetic transcription of the translated text.\n9. Perform an analysis of the Portuguese sentence, including: syntactic, grammatical, semantic and etymological analysis.\n \nReply example:\n    Input: She buys shoes at the department store.\n    Translation: Ela compra sapatos na loja de departamentos.\n    Phonetic Transcription: E-la com-pra sa-pa-tos na lo-jà de de-part-a-men-tos\n    Analysis:\n        Pronoun: Ela - she\n        Verb: Compra - buys\n        Noun: Sapatos - shoes\n        Preposition: Na (in + the) - at\n        Noun: Loja - store\n        Preposition: De - of\n        Noun: Departamentos - department.'''

}

assistants_df = pd.DataFrame(assistants.items(), columns=['assistant', 'instructions'])


####### Assistants #######
def send_to(m, who,  gpt=model, max = 1000, clip=True):
    if who in assistants:
        sys = assistants[who]
    elif len(who.split()) < 8:
        add_persona(who)
        sys = ''
    else:
        sys = who
    send_message(m,system=sys, maxtoken=max, model=gpt, to_clipboard=clip)

# Copilots
def chatgpt(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['base'], maxtoken=max, model=gpt, to_clipboard=clip)
def creator(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['creator'], maxtoken=max, model=gpt, to_clipboard=clip)
def delamain(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['delamain'], maxtoken=max, model=gpt, to_clipboard=clip)
def roger(m,  gpt=model, max = 1000, clip=True):
    expand_chat('Return always just the R code in your output!','system')
    send_message(m,system=assistants['roger'], maxtoken=max, model=gpt, to_clipboard=clip)
def robert(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['robert'], maxtoken=max, model=gpt, to_clipboard=clip)
def prompt_maker(m,  gpt=model, max = 1000, clip=True, sdxl=True):
    import stablediffusion_rag as sd
    if sdxl:
        assistant = sd.rag_sdxl
    else:
        assistant = sd.rag_sd
    send_message(m,system=assistant, maxtoken=max, model=gpt, to_clipboard=clip)


# Scientific Assistants
def galileo(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['galileo'], maxtoken=max, model=gpt, to_clipboard=clip)
def newton(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['newton'], maxtoken=max, model=gpt, to_clipboard=clip)
def leonardo(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['leonardo'], maxtoken=max, model=gpt, to_clipboard=clip)
def watson(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['watson'], maxtoken=max, model=gpt, to_clipboard=clip)
def crick(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['crick'], maxtoken=max, model=gpt, to_clipboard=clip)
def franklin(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['franklin'], maxtoken=max, model=gpt, to_clipboard=clip)
def darwin(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['darwin'], maxtoken=max, model=gpt, to_clipboard=clip)
def dawkins(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['dawkins'], maxtoken=max, model=gpt, to_clipboard=clip)
def turing(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['turing'], maxtoken=max, model=gpt, to_clipboard=clip)
def penrose(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['penrose'], maxtoken=max, model=gpt, to_clipboard=clip)
def venter(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['venter'], maxtoken=max, model=gpt, to_clipboard=clip, reinforcement=True)
def collins(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['collins'], maxtoken=max, model=gpt, to_clipboard=clip, reinforcement=True)


# Characters
def bestie(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['bestie'], maxtoken=max, model=gpt, to_clipboard=clip)
def julia(m,  gpt=model, max = 1000, clip=True, name='julia'):
    if os.path.exists("my_bio.txt"):
        assistants[name] = assistants[name]+'''\n***'''+load_file("my_bio.txt")+'***'
    else:
        pass
    send_message(m,system=assistants[name], maxtoken=max, model=gpt, to_clipboard=clip)



# Translators
def english(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['english'], maxtoken=max, model=gpt, to_clipboard=clip)
def italian(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['italian'], maxtoken=max, model=gpt, to_clipboard=clip)
def portuguese(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['portuguese'], maxtoken=max, model=gpt, to_clipboard=clip)
def japanese(m,  gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['japanese'], maxtoken=max, model=gpt, to_clipboard=clip)
def japanese_teacher(m, gpt=model, max = 1000, clip=True):
    print('Text: '+m.lstrip("@"))
    send_message(m,system=assistants['japanese_teacher'], maxtoken=max, model=gpt, to_clipboard=clip)
def portuguese_teacher(m, gpt=model, max = 1000, clip=True):
    send_message(m,system=assistants['portuguese_teacher'], maxtoken=max, model=gpt, to_clipboard=clip)


########################################

#%%
#%%
### trial ###
usage= '''import pychatgpt_lite as op
op.ask_gpt('Nel mezzo del cammin di nostra vita mi ritrovai per una serva oscura', op.assistants['french'])
reply = op.reply'''
#%%
#ask_gpt('Nel mezzo del cammin di nostra vita mi ritrovai per una serva oscura', assistants['french'], maxtoken=1000)
#%%
######### INFO #########
# https://platform.openai.com/account/rate-limits
# https://platform.openai.com/account/usage
# https://platform.openai.com/docs/guides/text-generation/chat-completions-api
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb



