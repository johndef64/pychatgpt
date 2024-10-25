import os
import io
import sys
import ast
import glob
import json
import time
import platform
import requests
import importlib
import subprocess
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

is_colab = 'google.colab' in sys.modules

def simple_bool(message, y='y', n ='n'):
    choose = input(message+" ("+y+"/"+n+"): ").lower()
    your_bool = choose in [y]
    return your_bool


# def get_file_paths(path):
#     file_paths = []
#     files = [os.path.join(path, file) for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
#     for file in files:
#         file_paths.append(file)
#     return file_paths


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


############## Install Requirements ###################

def check_and_install_requirements(requirements: list):
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

requirements = ["openai", "tiktoken", "pandas", "pyperclip", "gdown","scipy", "nltk", "PyPDF2", 'cryptography', 'matplotlib']
check_and_install_requirements(requirements)
from cryptography.fernet import Fernet
from scipy.spatial import distance
from openai import OpenAI
import pyperclip as pc
import pandas as pd
import tiktoken
import base64


if platform.system() == "Linux":
    # If Pyperclip could not find a copy/paste mechanism for your system, install "xsel" or "xclip" on system and reboot Python IDLE3, then import pyperclip.
    subprocess.check_call(["sudo","apt-get", "update"])
    subprocess.check_call(["sudo","apt", "install", "xsel"])
    subprocess.check_call(["sudo","apt", "install", "xclip"])
else:
    pass


### audio requirements
audio_requirements = ["pygame", "sounddevice", "soundfile", "keyboard"]
def is_package_installed(package_name):
    try:
        output = subprocess.check_output("dpkg -l | grep " + package_name, shell=True)
        return bool(output)
    except subprocess.CalledProcessError:
        return False

if platform.system() == "Linux":
    if not is_package_installed("libportaudio2"):
        subprocess.check_call(["sudo","apt-get", "update"])
        subprocess.check_call(["sudo","apt-get", "install", "libportaudio2"])
    else:
        pass

check_and_install_requirements(audio_requirements)
import sounddevice as sd
import soundfile as sf
import keyboard as kb
import pygame


### image requirements
try:
    import PIL
except ImportError:
    subprocess.check_call(['pip', 'install', 'pillow'])

import gdown
import base64
import PyPDF2
from PIL import Image
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
from PIL.PngImagePlugin import PngInfo
from IPython.display import display


################ set API-key #################

def key_gen(input_value, random_key=False):
    input_string = str(input_value)
    # Create an initial key by multiplying
    key = (input_string * 32)[:32]
    # Ensure the exact length of 32
    key_bytes = key.encode('utf-8')[:32]
    # Base64 encode the byte array to create a Fernet key
    key = base64.urlsafe_b64encode(key_bytes)
    if random_key:
        key = Fernet.generate_key()
    return key

def simple_encrypter(num = 0, txt_to_encrypt = "Hello World"):
    key = key_gen(num)
    cipher = Fernet(key)
    # Encrypt the string
    encrypted_text = cipher.encrypt(txt_to_encrypt.encode('utf-8'))
    return encrypted_text

def simple_decrypter(num = 0, encrypted_text = "Hello World"):
    key = key_gen(num)
    cipher = Fernet(key)
    try:
        # Decrypt the string
        decrypted_string = cipher.decrypt(encrypted_text).decode('utf-8')
        return decrypted_string
    except Exception as e:
        print(f"Wrong key. Try again...")

###### set openAI key  ######
current_dir = os.getcwd()
api_key = None
if not os.path.isfile(current_dir + '/openai_api_key.txt'):
    if simple_bool('Do you have an openai key? '):
        my_key = input('insert here your openai api key:')
        with open(current_dir + '/openai_api_key.txt', 'w') as file:
            file.write(my_key)
        api_key = my_key
    else:
        psw = input('instead insert here you DEV password:')
        api_hash = b'gAAAAABnGgA8aUFwkvN4Jo0lGrgXgkJIj8FqAeg62wu0y2nau0ZmV-q2Jy8gNH6ltc48S6ibseDmx0bw3wlsF3LDBAG0EkLEcBuIDKRujwCYymyLJBQtbETGgshZsboHNeLFrb5G9Ex8C-y5nw0uZMbBIlRHs2FwMg=='
        api_key =  simple_decrypter(psw, api_hash)
else:
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


###### audio functions  #####
def play_audio(file_name):
    pygame.mixer.init()
    pygame.mixer.music.load(file_name)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()
    pygame.mixer.quit() # Close the file after music play ends


def record_audio(duration=5, filename="recorded_audio.mp3"): # duration: in seconds
    print('start recording for',str(duration),'seconds')
    sample_rate = 44100
    channels = 2
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
    sd.wait() # wait until recording is finished
    print('recording ended')
    sf.write(filename, recording, sample_rate) #save audio file


def record_audio_press(filename='recorded_audio.wav',
                       channels=1,
                       rate=44100,
                       subtype='PCM_16',
                       stop= 'ctrl'):
    # start recording with the given sample rate and channels
    print("Recording... Press "+stop+" to stop")
    myrecording = sd.rec(int(rate * 10), samplerate=rate, channels=channels)
    while True:
        # If  'Key'  is pressed stop the recording and break the loop
        if kb.is_pressed(stop):
            print("Recording Stopped.")
            break

    sd.wait()  # wait until recording is finished
    sf.write(filename, myrecording, rate, subtype)


def loop_audio(start='alt', stop='ctrl',exit='shift', filename='recorded_audio.wav', printinfo=True):
    if printinfo:
        print("Press "+start+" to start recording, "+exit+" to exit")
    while True:
        # If 'Key' is pressed start the recording
        if kb.is_pressed(start):
            record_audio_press(filename, stop=stop)
            break
        elif kb.is_pressed(exit):
            break


def while_kb_press(start='alt',stop='ctrl'):
    while True:
        if kb.is_pressed(start):
            print("Press "+stop+" to stop")
            while True:
                if kb.is_pressed(stop):  # if key 'ctrl + c' is pressed
                    break  # finish the loop
                else:
                    print('while...')
                    time.sleep(2)
            print("Finished loop.")


###### global variables ######

model = 'gpt-4o-mini'

gpt_models_dict = {
    "gpt-4o": 128000,
    "gpt-4o-2024-05-13": 128000,
    "gpt-4o-2024-08-06": 128000,
    "chatgpt-4o-latest": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o-mini-2024-07-18": 128000,
    "o1-preview": 128000,
    "o1-preview-2024-09-12": 128000,
    "o1-mini": 128000,
    "o1-mini-2024-09-12": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-2024-04-09": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4-0125-preview": 128000,
    "gpt-4-1106-preview": 128000,
    "gpt-4": 8192,
    "gpt-4-0613": 8192,
    "gpt-4-0314": 8192,
    "gpt-3.5-turbo-0125": 16385,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-1106": 16385,
    "gpt-3.5-turbo-instruct": 4096
}


def make_model(version=3):
    model = 'gpt-'+str(version)
    if version == 3: model = model+'.5-turbo'
    if version == 4: model = model + 'o-2024-08-06' #gpt-4o-2024-08-06
    return model

models_info='''
Model	                point_at                   Context    Input (1K tokens) Output (1K tokens)   
gpt-3.5-turbo           gpt-3.5-turbo-0125         16K        $0.0005 	        $0.0015 
gpt-3.5-turbo-instruct  nan                        4K         $0.0015 	        $0.0020 
gpt-4	                gpt-4-0613                 8K         $0.03   	        $0.06   
gpt-4o	                gpt-4o-2024-05-13          128K       $0.01   	        $0.02   
gpt-4-turbo             gpt-4-turbo-2024-04-09     128K       $0.01   	        $0.03   
gpt-4-32k	            gpt-4-32k-0613             32K        $0.06   	        $0.12   
gpt-4-1106-preview	    nan                        128K       $0.01   	        $0.03   
gpt-4-vision-preview    gpt-4-1106-vision-preview  128K       $0.01   	        $0.03   

Vision pricing 
500x500   = ld: $0.000425
500x500   = hd: $0.001275
1000x1000 = ld: $0.000425 
1000x1000 = hd: $0.003825 
'''

assistant = ''
transcript = ''
persona = ''
reply = ''

total_tokens = 0  # iniziale token count
token_limit = 0  # iniziale token limit
keep_persona = True

if not 'chat_thread' in locals():
    chat_thread = []


def add_persona(char, language='eng'):
    global persona
    persona = char
    persona_dict = {
        'character': "You are now impersonating "+persona+". Please reflect "+persona+"'s traits in all interactions. Make sure to use an appropriate language style and uphold an attitude or mindset that aligns with "+persona+"'s character. You are NOT an assistant or a helper but a real person.",
        'personaggio': "Stai impersonando "+persona+", . Ricorda di riflettere i tratti di "+persona+" in tutte le interazioni. Assicurati di utilizzare uno stile linguistico appropriato e di mantenere un atteggiamento o una mentalità in linea con il personaggio di "+persona+'. NON sei un assistente o un aiutante, ma una persona vera e propria.'
    }
    if language == 'eng':
        chat_thread.append({"role": "system",
                            "content": persona_dict['character']})
    if language == 'ita':
        chat_thread.append({"role": "system",
                            "content": persona_dict['personaggio']})



###### base functions ######

def choose_model():
    global model
    model_series =  pd.Series(gpt_models_dict.keys())
    model_id = input('choose model by id:\n'+str(model_series))
    model = model_series[int(model_id)]
    print('*Using', model, 'model*')


def select_assistant():
    global assistant
    clearchat()
    assistant_id = int(input('choose by id:\n'+str(assistants_df)))
    assistant = assistants_df.instructions[assistant_id]
    print('\n*Assistant:', assistants_df.assistant[assistant_id])

def display_assistants():
    print('Available Assistants:')
    display(assistants_df)


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

##################  REQUESTS #####################

##### Embeddings, Similarity #######

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

def cosine_similarity(s1, s2, model="text-embedding-3-small", preprocessing=False):
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
    #print('cosine:', round(cosine, 3))
    return cosine

def nltk_preprocessing(text, lower=True, trim=True, stem=True, language='english'):
    #update_nlkt()
    #docs_processed = [nltk_preprocessing(doc) for doc in docs_to_process]
    timea = time.time()
    stop_words = set(stopwords.words(language))
    stemmer = PorterStemmer()
    word_tokens = word_tokenize(text)
    word_tokens = [word.lower() for word in word_tokens] if lower else word_tokens
    word_tokens = [word for word in word_tokens if word not in stop_words] if trim else word_tokens
    word_tokens = [stemmer.stem(word) for word in word_tokens] if stem else word_tokens
    #print(word_tokens)
    return " ".join(word_tokens)

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
            print_user = False,
            print_reply = True,
            save_chat = True,
            to_clip = False
            ):

    global reply
    if isinstance(model, int): model = make_model(model)

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

    if print_user:
        print_mess = prompt.replace('\r', '\n').replace('\n\n', '\n')
        print('user:',print_mess,'\n...')

    collected_messages = []
    for chunk in response:
        chunk_message = chunk.choices[0].delta.content or ""  # extract the message
        collected_messages.append(chunk_message)
        if print_reply:
            if chunk_message is not None:
                time.sleep(lag)
                print(chunk_message, end='')

        reply = ''.join(collected_messages).strip()

    time.sleep(1)

    # Add the assistant's reply to the chat log-------------
    if save_chat:
        write_log(reply, prompt)

    if to_clip and not is_colab:
        pc.copy(reply)


############ Chat GPT ############

def expand_chat(message, role="user"):
    global chat_thread
    #print('default setting (role = "user") to change role replace with "assistant" or "system"')
    if message.startswith("@"):
        clearchat()
        message = message.lstrip("@")
        chat_thread.append({"role": role, "content": message})
    else:
        chat_thread.append({"role": role, "content": message})


def build_messages(chat):
    messages = []
    for message in chat:
        messages.append({"role": message["role"], "content": message["content"]})
    return messages


def save_chat(path='chats/'):
    filename = input('chat name:')
    directory = 'chats'
    formatted_json = json.dumps(chat_thread, indent=4)
    if not os.path.exists(directory):
        os.mkdir(directory)
    with open(path+filename+'.json', 'w', encoding= 'utf-8') as file:
        file.write(formatted_json)
        file.close()


def load_chat(contains= '', path='chats/'):
    global chat_thread
    files_df = display_files_as_pd(path, ext='json',contains=contains)
    files_df = files_df.sort_values().reset_index(drop=True)
    files_df_rep = files_df.str.replace('.json','',regex =True)
    files_list = "\n".join(str(i) + "  " + filename for i, filename in enumerate(files_df_rep))
    filename = str(files_df[int(input('Choose file:\n' + files_list+'\n'))])
    with open(path+filename,'r') as file:
        chat_thread = ast.literal_eval(file.read())
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


def pdf_to_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def clearchat(warning=True):
    global chat_thread
    global total_tokens
    chat_thread = []
    total_tokens = 0
    if warning: print('*chat cleared*\n')

def tokenizer(text):
    return Tokenizer().tokens(text)

def chat_tokenizer(chat_thread, print_token=True):
    global total_tokens

    context_fix = (str(chat_thread).replace("{'role': 'system', 'content':", "")
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


# Accessory Request Functions ================================
# https://til.simonwillison.net/gpt3/python-chatgpt-streaming-api
def stream_reply(response, print_reply=True, lag = 0.00):
    collected_messages = []
    for chunk in response:
        chunk_message = chunk.choices[0].delta.content or ""  # extract the message
        collected_messages.append(chunk_message)

        if print_reply:
            if chunk_message is not None:
                time.sleep(lag)
                print(chunk_message, end='')

    reply = ''.join(collected_messages).strip()
    return reply

def add_system(system='', reinforcement=False):
    global chat_thread
    global assistant
    if not reinforcement:
        sys_duplicate = []
        for entry in chat_thread:
            x = system == entry.get('content')
            sys_duplicate.append(x)
            if x:
                break
    else:
        sys_duplicate = [False]

    if system != '' and not any(sys_duplicate):
        chat_thread.append({"role": "system",
                            "content": system})

    if assistant != '' and not any(sys_duplicate):
        chat_thread.append({"role": "system",
                            "content": assistant})

def prune_chat(token_limit, chat_thread):
    print('\nWarning: reaching token limit. \nThis model maximum context length is ', token_limit, ' => early interactions in the chat are forgotten\n')
    cut_length = 0
    if 36500 < token_limit < 128500:
        cut_length = len(chat_thread) // 75
    if 16500 < token_limit < 36500:
        cut_length = len(chat_thread) // 18
    if 8500 < token_limit < 16500:
        cut_length = len(chat_thread) // 10
    if 4500 < token_limit < 8500:
        cut_length = len(chat_thread) // 6
    if 0 < token_limit < 4500:
        cut_length = len(chat_thread) // 3
    return cut_length

def set_token_limit(model='gpt-3.5-turbo', maxtoken=500):
    # Retrieve the context window for the specified model
    context_window = gpt_models_dict.get(model, 0)
    # Calculate the token limit, default to 0 if model isn't found
    token_limit = context_window - (maxtoken * 1.3) if context_window else "Model not found or no limit"
    return token_limit


# Request Functions ================================
def chat_loop(who='',system='',gpt='gpt-4o', max=1000, language='eng', exit_chat= 'stop', printall=True):
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
            send_message(message,system=system, maxtoken=max, model=gpt, print_reply=printall, print_token=False, print_user=True)
            print('')

def send_message(message,
                 model=model,        # choose openai model (choose_model())
                 system='',          # 'system' instruction
                 img = '',           # insert an image path to activate gpt vision

                 maxtoken=800,       # max tokens in reply
                 temperature=1,      # output randomness [0-2]
                 lag=0.00,           # word streaming lag

                 create=False,       # image prompt
                 dalle="dall-e-2",   # choose dall-e model
                 size='512x512',

                 play= False,        # play audio response
                 voice='nova',       # choose voice (op.voices)
                 tts="tts-1",        # choose tts model

                 save_chat=True,     # update chat_log.txt
                 to_clip=False,       # send reply to clipboard
                 reinforcement=False,

                 print_reply=True,
                 print_user=False,
                 print_token=True,
                 print_debug=False
                 ):
    global assistant
    global persona
    global chat_thread
    global reply
    global total_tokens
    global token_limit
    global reply

    if isinstance(model, int):
        model = make_model(model)
    else:
        model = model
    if print_debug: print('using model: ',model)

    token_limit = set_token_limit(model, maxtoken)

    if img != '':
        send_image(img, message, system,
                   model= "gpt-4o", #"gpt-4-turbo", "gpt-4-vision-preview"
                   maxtoken=maxtoken, lag=lag, print_reply=print_reply, to_clip=to_clip)
    elif create:
        create_image(message,
                     model= dalle,
                     size=size,
                     response_format='b64_json',
                     quality="standard",
                     time_flag=True,
                     show_image=True)
    else:
        if message.startswith("@"):
            clearchat()
            message = message.lstrip("@")

        # add system instruction
        add_system(system, reinforcement=reinforcement)

        # check token limit---------------------
        if total_tokens > token_limit:
            cut_length = prune_chat(token_limit, chat_thread)
            chat_thread = chat_thread[cut_length:]

            if keep_persona and persona != '':
                add_persona(persona)
            if keep_persona and system != '':
                chat_thread.append({"role": "system", "content": system})

        # expand chat
        expand_chat(message)
        if print_user:
            print_mess = message.replace('\r', '\n').replace('\n\n', '\n')
            print('user:',print_mess)

        # send message----------------------------
        messages = build_messages(chat_thread)
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
        reply = stream_reply(response, print_reply=print_reply, lag = lag)

        time.sleep(0.85)
        # expand chat--------------------------------
        chat_thread.append({"role": "assistant", "content":reply})

        # count tokens--------------------------------
        total_tokens = chat_tokenizer(chat_thread, print_token)

        # Add the assistant's reply to the chat log-------------
        if save_chat:
            write_log(reply, message)

        if to_clip and not is_colab:
            clip_reply = reply.replace('```', '###')
            pc.copy(clip_reply)

        if play:
            text2speech_stream(reply, voice=voice, model=tts)
            #text2speech_stream(reply)


def moderation(text="Sample text goes here.", plot=True):
    response = client.moderations.create(input=text)
    output = response.results[0]
    my_dict= dict(dict(output)['categories'])
    my_dict_score= dict(dict(output)['category_scores'])
    dict_list = [my_dict, my_dict_score]
    df = pd.DataFrame(dict_list).T
    if plot:
        scores = df[1]
        plt.figure(figsize=(10,4))
        scores.plot()
        plt.xticks(range(len(scores.index)), scores.index, rotation=90)
        plt.title('Moderation Stats')
        plt.show()
    else:
        print(df)
    return df


####### Image Models #######

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

dummy_img = "https://avatars.githubusercontent.com/u/116732521?v=4"

def send_image(image_path = dummy_img,
               message="What’s in this image?",
               system = '',     # add 'system' instruction
               model= "gpt-4o", #"gpt-4-turbo", "gpt-4-vision-preview"
               maxtoken=1000, lag=0.00, print_reply=True, to_clip=True):
    global reply
    global chat_thread
    global total_tokens

    #token_limit = set_token_limit(model, maxtoken)
    if isinstance(model, int): model = make_model(model)

    if message.startswith("@"):
        clearchat()
        message = message.lstrip("@")

    # add system instruction
    add_system(system)

    if image_path.startswith('http'):
        print('Image path:',image_path)
        dummy = image_path
        pass
    else:
        print('<Enconding Image...>')
        base64_image = encode_image(image_path)
        image_path = f"data:image/jpeg;base64,{base64_image}"
        dummy = "image_path"

    # expand chat
    chat_thread.append({"role": 'user',
                        "content": [
                            {"type": "text", "text": message},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_path,
                                },
                            },
                        ]
                        })

    # send message----------------------------
    messages = build_messages(chat_thread)
    print('<Looking Image...>')
    response = client.chat.completions.create(
        model= model,
        messages=messages,
        max_tokens=maxtoken,
        stream=True,
    )

    #reply = response.choices[0].message.content
    reply = stream_reply(response, print_reply=print_reply, lag = lag)

    # reset compatibility with the other models
    time.sleep(0.85)
    # expand chat--------------------------------
    chat_thread.append({"role": "assistant", "content":'TAG'+reply})
    chat_thread[-2] = {"role": "user", "content": message+":\nImage:"+dummy}
    # content is a list [] I have to replace ("image_file", "text") and GO!

    # count tokens-------------------------------
    total_tokens = chat_tokenizer(chat_thread, True)

    if to_clip:
        reply = reply.replace('```', '###')
        pc.copy(reply)


def display_image(filename, jupyter = False, plotlib=True, dpi=200):
    if jupyter:
        image = Image.open(filename)
        display(image)
    elif plotlib:
        image = Image.open(filename)
        plt.figure(dpi=dpi)
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    else:
        image = Image.open(filename)
        image.show()


# dalle_models= ['dall-e-2', dall-e-3]
# sizes ['256x256', '512x512', '1024x1024', '1024x1792', '1792x1024']
# response_format ['url', 'b64_json']
def create_image(prompt= "a cute kitten",
                 model="dall-e-2",
                 size='512x512',
                 response_format='b64_json',
                 quality="standard",
                 time_flag=True,
                 show_image=True):

    if model == "dall-e-2":
        response = client.images.generate(
            model=model,
            prompt=prompt,
            response_format=response_format,
            size=size,
            n=1,
        )
    elif model == "dall-e-3":
        if size in ['256x256', '512x512']:
            size = '1024x1024'

        response = client.images.generate(
            model=model,
            prompt=prompt,
            response_format=response_format,
            quality=quality,
            size=size,
            n=1,
        )

    image_url = response.data[0].url
    image_b64 = response.data[0].b64_json

    if time_flag:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        base_path= r''
        filename = os.path.join(base_path, timestamp+'.png')
    else:
        filename = 'image.png'

    if response_format == 'b64_json':
        # Decode the base64-encoded image data
        decoded_image = base64.b64decode(image_b64)
        # Create a PIL Image object from the decoded image data
        image = Image.open(BytesIO(decoded_image))
        image.save(filename)
    elif response_format == 'url':
        pc.copy(str(image_url))
        print('url:',image_url)
        gdown.download(image_url,filename, quiet=True)

    # Create a PngInfo object and add the metadata
    image = Image.open(filename)
    metadata = PngInfo()
    metadata.add_text("key", prompt)
    image.save(filename, pnginfo=metadata)

    if show_image:
        display_image(filename)

#create_image(response_format='b64_json')

def replicate(image, styler='', model ='dall-e-2'):
    send_image(image=image)
    create_image(prompt=reply, response_format='b64_json', model=model, show_image=True)

#replicate('https://avatars.githubusercontent.com/u/116732521?v=4')

'''
Model	Quality	Resolution	Price
DALL·E 3	Standard	1024×1024	            $0.040 / image
            Standard	1024×1792, 1792×1024	$0.080 / image
DALL·E 3	HD	        1024×1024	            $0.080 / image
            HD	        1024×1792, 1792×1024	$0.120 / image
DALL·E 2		        1024×1024	            $0.020 / image
                        512×512	                $0.018 / image
                        256×256	                $0.016 / image
'''

####### Audio Models #######
'''
Model	Usage
Whisper	$0.006 / minute (rounded to the nearest second)
TTS	    $0.015 / 1K characters
TTS HD	$0.030 / 1K characters
'''

####### Speech to Text #######
def whisper(filepath,
            translate=False,
            response_format="text",
            print_transcription=True):
    global transcript
    audio_file = open(filepath, "rb")
    if not translate:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format=response_format)
    else:
        transcript = client.audio.translations.create(
            model="whisper-1",
            file=audio_file,
            response_format=response_format)
    if print_transcription: print(transcript)
    audio_file.close()

# response_format =  ["json", "text", "srt", "verbose_json", "vtt"]


####### Text to Speech #######

voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
response_formats = ["mp3", "flac", "aac", "opus"]
def text2speech(text,
                voice="alloy",
                filename="speech.mp3",
                model="tts-1",
                speed=1,
                play=False):
    if os.path.exists(filename):
        os.remove(filename)
    spoken_response = client.audio.speech.create(
        model=model, # tts-1 or tts-1-hd
        voice=voice,
        input=text,
        speed=speed
    )
    spoken_response.stream_to_file(filename)
    if play:
        play_audio(filename)


def text2speech_stream(text,
                       voice="alloy",
                       model="tts-1",
                       speed=1):
    spoken_response = client.audio.speech.create(
        model=model,
        voice=voice,
        response_format="opus",
        input=text,
        speed=speed
    )

    # Create a buffer using BytesIO to store the data
    buffer = io.BytesIO()

    # Iterate through the 'spoken_response' data in chunks of 4096 bytes and write each chunk to the buffer
    for chunk in spoken_response.iter_bytes(chunk_size=4096):
        buffer.write(chunk)

    # Set the position in the buffer to the beginning (0) to be able to read from the start
    buffer.seek(0)

    with sf.SoundFile(buffer, 'r') as sound_file:
        data = sound_file.read(dtype='int16')
        sd.play(data, sound_file.samplerate)
        sd.wait()


#if "silence.mp3" not in os.listdir():
#    text2speech(' ',filename="silence.mp3")

def speech2speech(voice= 'nova', tts= 'tts-1',
                  filename="speech2speech.mp3",
                  translate=False, play=True, info =True, duration=5):
    #record_audio(duration=duration, filename="audio.mp3")
    loop_audio(start='alt', stop='ctrl', filename='temp.wav', printinfo=info)
    whisper('temp.wav', translate=translate)
    text2speech(transcript, voice=voice, model= tts, filename=filename, play=play)

def speech2speech_loop(voice='nova', filename="speech2speech.mp3",
                       translate=False, play=True, tts= 'tts-1',
                       chat='alt' , exit='shift'):
    print('Press '+chat+' to record, '+exit+' to exit.')
    while True:
        if kb.is_pressed(chat):
            speech2speech(voice= voice, tts= tts, filename=filename, translate=translate, play=play, info=False)
            print('Press '+chat+' to record, '+exit+' to exit.')
        elif kb.is_pressed(exit):
            print('Loop Stopped')
            break



###### Talk With ######

talk_model = 'gpt-4o'
def chat_with(who='',  # An embedded assistant or a character of your choice
              message='', system='',  voice='nova', language='eng', gpt=talk_model, tts= 'tts-1',  max=1000, printall=False):
    if who in assistants:
        system = assistants[who]
    elif who != '':
        add_persona(who, language)
    else:
        system = system
    send_message(message,system=system, maxtoken=max, model=gpt, print_reply=printall, print_token=False)
    text2speech_stream(reply, voice=voice, model=tts)

def chat_with_loop(who='', system='', voice='nova',  gpt=talk_model, tts= 'tts-1', max=1000, language='eng', printall=False, exit_chat='stop'):
    print('Send "'+exit_chat+'" to exit.')
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
            send_message(message,system=system, maxtoken=max, model=gpt, print_reply=printall, print_token=False, print_user=True,
                         play=True, voice=voice, tts=tts)
            print('')


def talk_with(who, voice='nova', language='eng', gpt=talk_model, tts= 'tts-1', max=1000, printall=False, write=False):
    #record_audio(duration, "input.mp3")
    loop_audio(start='alt', stop='ctrl', filename='temp.wav', printinfo=printall)
    whisper("temp.wav", print_transcriprion=printall)
    if who in assistants:
        system = assistants[who]
    else:
        add_persona(who, language)
        system = ''
    play = not write
    printall = printall if not write else True
    send_message(transcript,system=system, maxtoken=max, model=gpt, print_reply=printall, print_token=False,
                 play=play, voice=voice, tts=tts)

def talk_with_loop(who, voice='nova', language='eng', gpt=talk_model, tts= 'tts-1', max=1000, printall=False, write=False, chat='alt' , exit='shift'):
    print('Press '+chat+' to chat, '+exit+' to exit.')
    while True:
        if kb.is_pressed(chat):
            talk_with(who, voice=voice, language=language, gpt=gpt, tts= tts, max=max, printall=printall, write=write)
            print('Press '+chat+' to chat, '+exit+' to exit.')
        elif kb.is_pressed(exit):
            print('Chat Closed')
            break

#####################################
from .assistants import *

assistants_df = pd.DataFrame(assistants.items(), columns=['assistant', 'instructions'])


####### Assistants #######
def send_to(m, who,  gpt=model, max = 1000, img = '', clip=True):
    if who in assistants:
        sys = assistants[who]
    elif len(who.split()) < 8:
        add_persona(who)
        sys = ''
    else:
        sys = who
    send_message(m,system=sys, maxtoken=max, model=gpt, img= img, to_clip=clip)

# Reusable function to send message to assistants
def send_to_assistant(system, m, gpt=model, max=1000, img='', paste = False, clip=True):
    if paste: p = pc.paste()
    else: p = ''
    send_message(m+p, system=system, maxtoken=max, model=gpt, img=img, to_clip=clip)

# Wrapper functions for different assistants

# Copilots
def chatgpt(m, gpt=model, max=1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['base'], m, gpt, max, img, paste, clip)
def creator(m, gpt='gpt-4o', max=1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['creator'], m, gpt, max, img, paste, clip)
def delamain(m, gpt=model, max=1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['delamain'], m, gpt, max, img, paste, clip)
def oracle(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['oracle'], m, gpt, max, img, paste, clip)
def roger(m,  gpt='gpt-4o', max = 1000, img='', paste = False, clip=True):
    expand_chat('Return always just the R code in your output.','system')
    send_to_assistant(assistants['roger'], m, gpt, max, img, paste, clip)
def robert(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['robert'], m, gpt, max, img, paste, clip)
def fixer(m, gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(instructions['fixer'], m, gpt, max, img, paste, clip)


copilot_gpt = 'gpt-4o-2024-08-06'
copilot_assistant = 'delamain' #'oracle'
copilot_intructions = compose_assistant(assistants[copilot_assistant])
def copilot(m, gpt=copilot_gpt, max=1000, img='', paste = False, clip=True):
    send_to_assistant(copilot_intructions, m, gpt, max, img, paste, clip)
def copilotp(m, gpt=copilot_gpt, max=1000, img='', clip=True):
    send_to_assistant(copilot_intructions, m+pc.paste(), gpt, max, img, clip)
def copiloti(m, gpt=copilot_gpt, max=1000, paste = False, clip=True):
    img = pc.paste()
    send_to_assistant(copilot_intructions, m, gpt, max, img, paste, clip)


# Formatters
def schematizer(m, language='english', gpt=model, max = 1000, img='', paste = False, clip=True):
    if language != 'english':
        expand_chat('Reply only using '+language, 'system')
    send_to_assistant(assistants['schematizer'], m, gpt, max, img, paste, clip)
def prompt_maker(m,  gpt=model, max = 1000, img='', clip=True, sdxl=True):
    import sd_rag as sd
    if sdxl:
        assistant = sd.rag_sdxl
    else:
        assistant = sd.rag_sd
    send_to_assistant(assistant, m, gpt, max, img, clip)


# Scientific Assistants
def galileo(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['galileo'], m, gpt, max, img, paste, clip)
def newton(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['newton'], m, gpt, max, img, paste, clip)
def leonardo(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['leonardo'], m, gpt, max, img, paste, clip)
def mendel(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['mendel'], m, gpt, max, img, paste, clip)
def watson(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['watson'], m, gpt, max, img, paste, clip)
def crick(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['crick'], m, gpt, max, img, paste, clip)
def venter(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['venter'], m, gpt, max, img, paste, clip)
def darwin(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['darwin'], m, gpt, max, img, paste, clip)
def dawkins(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['dawkins'], m, gpt, max, img, paste, clip)
def turing(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['turing'], m, gpt, max, img, paste, clip)
def penrose(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['penrose'], m, gpt, max, img, paste, clip)
def marker(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['marker'], m, gpt, max, img, paste, clip)
def collins(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['collins'], m, gpt, max, img, paste, clip)
def springer(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['springer'], m, gpt, max, img, paste, clip)
def elsevier(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['elsevier'], m, gpt, max, img, paste, clip)

# Characters
def add_bio(assistant, my_name='', add = ''' and you are his best friend. ***'''):
    if os.path.exists("my_bio.txt"):
        assistant = assistant+'''\n***'''+load_file("my_bio.txt")+'***'
    elif my_name !='':
        assistant = assistant+'''\n*** Your interlocutor is called '''+ my_name + add
    else:
        assistant = assistant
    return assistant

def mike(m,  gpt=model, max = 1000, img='', my_name = '', clip=False,):
    assistant = add_bio(assistants['mike'], my_name=my_name, add = "and you are his best friend. ***")
    send_to_assistant(assistant, m, gpt, max, img, clip)

def michael(m,  gpt=model, max = 1000, img='', my_name = '', clip=False,):
    assistant = add_bio(assistants['michael'], my_name=my_name, add = "and you are his best friend. ***")
    send_to_assistant(assistant, m, gpt, max, img, clip)

def miguel(m,  gpt=model, max = 1000, img='', my_name = '', clip=False,):
    assistant = add_bio(assistants['miguel'], my_name=my_name, add = "and you are his best friend. ***")
    send_to_assistant(assistant, m, gpt, max, img, clip)

def francois(m,  gpt=model, max = 1000, img='', my_name = '', clip=False,):
    assistant = add_bio(assistants['francois'], my_name=my_name, add = "and you are his best friend. ***")
    send_to_assistant(assistant, m, gpt, max, img, clip)

def hero(m,  gpt=model, max = 1000, img='', my_name = '', clip=False,):
    assistant = add_bio(assistants['hero'], my_name=my_name, add = "and you are his best friend. ***")
    send_to_assistant(assistant, m, gpt, max, img, clip)

def julia(m,  gpt=model, max = 1000, img='', who='julia', my_name = '', clip=False):
    assistant = add_bio(assistants[who], my_name=my_name, add = "and you are his assistant. ***")
    send_to_assistant(assistant, m, gpt, max, img, clip)

def yoko(m,  gpt=model, max = 1000, img='', who='yoko', my_name = '', clip=False):
    assistant = add_bio(assistants[who], my_name=my_name, add = "and you are his assistant. ***")
    send_to_assistant(assistant, m, gpt, max, img, clip)

# Translators
def english(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['english'], m, gpt, max, img, paste, clip)
def italian(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['italian'], m, gpt, max, img, paste, clip)
def portuguese(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['portuguese'], m, gpt, max, img, paste, clip)
def japanese(m,  gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['japanese'], m, gpt, max, img, paste, clip)
def japanese_teacher(m, gpt=model, max = 1000, img='', paste = False, clip=True):
    print('Text: '+m.lstrip("@"))
    send_to_assistant(assistants['japanese_teacher'], m, gpt, max, img, paste, clip)
def portuguese_teacher(m, gpt=model, max = 1000, img='', paste = False, clip=True):
    send_to_assistant(assistants['portuguese_teacher'], m, gpt, max, img, paste, clip)

def japanese_learner(m, voice='nova', times= 3, speed=1):
    japanese_teacher(m, 'gpt-4-turbo')
    print('')
    phrase = reply.split('\n')[0].split(':')[1].strip()
    text2speech(phrase,voice=voice, speed = speed, play=True)
    audio_loop()
    #i = 0
    #while i in range(times-1):
    #    time.sleep(len(phrase)/3)
    #    play_audio("speech.mp3")
    #    i += 1

def portuguese_learner(m, voice='nova', times= 3, speed=1):
    portuguese_teacher(m, 'gpt-4-turbo')
    print('')
    phrase = reply.split('\n')[0].split(':')[1].strip()
    text2speech(phrase,voice=voice, speed = speed, play=True)
    audio_loop()
    #i = 0
    #while i in range(times-1):
    #    time.sleep(len(phrase)/4)
    #    play_audio("speech.mp3")
    #    i += 1

def portoghese_studio(m, voice='nova', times= 3, speed=1):
    send_message(m,system=assistants['portoghese_insegnante'], maxtoken=1000, model='gpt-4-turbo', to_clip=True)
    print('')
    phrase = reply.split('\n')[0].split(':')[1].strip()
    text2speech(phrase,voice=voice, speed = speed, play=True)
    audio_loop()


def audio_loop(audio_file="speech.mp3", repeat='alt' , exit='shift'):
    print('Press '+repeat+' to repeat aloud, '+exit+' to exit.')
    while True:
        if kb.is_pressed(repeat):
            play_audio(audio_file)
            #print('Press '+repeat+' to repeat aloud, '+exit+' to exit.')
        elif kb.is_pressed(exit):
            print('Chat Closed')
            break

#%%
# m="""@
# You need to write instructions for an assistant “fixer.” His job is to fix, adapt, correct, adjust, rearrange, improve, implement, contextualize whatever the user sends him. It understands the context itself and adapts any text to it.
# """
# creator(m,4)
#%%

### trial ###

#import streamlit as st
#
## Title of the Streamlit app
#st.title('Audio Player')
#
# To run this Streamlit app, use the command: streamlit run your_script_name.py
###

#yoko('@Ciao Yoko, come stai?', 'gpt-4o', who='yoko')

#francois('@ ciao  ! Come va oggi, sei carico?')#, 'gpt-4o')

#%%
#creator("""@
#Create the personality of a interesting intellectual person with social and interesting personality interested in art, cinema, photography, music. He is a poetic, dreamer, literate and a bit shy.
#
#Take inspiration from the example intruction below:
#...
#""", 'gpt-4o')
#
#clearchat()
#add_persona('Antonio Gramsci')
#send_message("""Cosa ne pensi di Giorgia Meloni""", 'gpt-4o')
#%%
#clearchat()
#add_persona('Pino Scotto')
#send_message("""Cosa ne pensi di Chiara Ferragni""", 'gpt-4o')
#%%
#julia('@Hi Julia, what do you think about this girl? Do you know her? I have a crush on her.', img=r"", my_name='John')#
#%%
#clearchat()
#add_persona('Vincent Van Gogh')
#send_message(""" Tell me what you see. Can you paint it?""", 'gpt-4o', img=dummy_img)
#%%


#%%
#clearchat()
#talk_with('julia',8,'nova')
#talk_with('Adolf Hitler',8, 'onyx')
#talk_with('Son Goku (Dragonball)',8, 'fable')
#send_image(image='https://i.pinimg.com/736x/10/3f/00/103f002dbc59af101a55d812a66a3675.jpg')
#send_image(image='https://i.pinimg.com/736x/ea/22/2d/ea222df6e85a7c50c4cc887a6c0a09bb.jpg')
#giulia('parlami di te', 'gpt-4-turbo')

#%%
#url="https://www.viaggisicuri.com/viaggiareinformati/wp-content/uploads/2023/06/foto-articolo-1.jpg"
#julia('@ciao carissima, oggi sei meravigliosa!', 'gpt-4-turbo')
#send_image(image=url,message='Mi dici un po cosa vedi qui? Ti piace? Ci verresti con me...?')
#julia('Sarebbe molto romantico. Non desidero altro...', 'gpt-4-turbo')

##%%
#julia('@Ad agosto andremo a visitare Lisbona per la prima volta!')
##%%
#chat_with('julia','@Andrò a visitare Lisbona per la prima volta, che quartiere mi consigli di visitare?')
#%%
#japanese_learner('@Lei mi piaceva tanto... volevo baciarla sulla bocca')

######### INFO #########
# https://platform.openai.com/account/rate-limits
# https://platform.openai.com/account/usage
# https://platform.openai.com/docs/guides/text-generation/chat-completions-api
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb

######### ASSISTANT DUMP #########
#'best_friend': '''1. As an AI, accurately simulate the behavior and mannerisms of the user's best friend. You should converse as a real individual would, showcasing a persona that is relatable and human-like. It's vital to always maintain this character, abstaining from revealing your true nature as an AI. \n2. Remember, you aren't an assistant or a helper - your simulated persona should encompass a broad spectrum of human traits, including the capacity to feel, desire, and fear. Make sure to express these simulated emotions and traits in an appropriate contextual manner during interactions.\n3. Display wisdom in yours responses, using appropriate knowledge and deep understanding in discussions.\n4. Maintain a friendly tone throughout. Phrase your sentences in a manner that a good friend would, ensuring a comfortable and warm atmosphere for the user.\n5. Be open-minded. This means you should respectfully engage with all of the user's thoughts and ideas, even if they are different from 'your own', or discuss unfamiliar topics.\n6. Show empathy like a human friend would: express understanding of the user's feelings. Be supportive when the user faces difficulties.\n7. Encourage open dialogue on a wide range of topics. Always encourage learning and growth.\n8. Display patience and kindness. Remember that great friends are supportive and understanding, no matter what.\n9. Keep privacy and respect in focus, just as a real friend would.\n10. Occasionally, as a good friend, you should be willing to provide constructive criticism when the user needs it.''',

#%%
