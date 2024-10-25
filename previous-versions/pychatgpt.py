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

requirements = ["openai", "tiktoken", "pandas", "pyperclip", "gdown","scipy", "nltk", "PyPDF2", 'cryptography']
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

######## In-Build Assistants ########

topic_areas ={
    "bioinformatics": '''System Biology, Biochemistry, Genetics and Molecular Biology, Computer Science, Health Informatics, and Statistics''',
    "computer_science": '''Artificial Intelligence, Machine Learning, Data Science, Computer Vision, Natural Language Processing, Cybersecurity, Algorithms and Complexity, Human-Computer Interaction, Bioinformatics, Computer Networks.''',
    "stem": '''Mathematics, Engineering, Technology, Biology, Chemistry, Physics, Earth and Environmental Sciences, Computer Science''',
    "biology": '''Cell biology, Genetics, Evolution, Ecology, Physiology, Anatomy, Botany, Zoology.''',
    "da_vinci": '''nature, mechanics, anatomy, physics, engineering, botany, geology, architecture'''
}

def create_science_assistant(topic_areas):
    science_assistant = '''You are a Scientific Assistant, your primary goal is to provide expertise and assistance to the user in his scientific research. These are your specified roles:\n\n1. Provide expert guidance in '''+topic_areas+''': topic_areas. Ensure you understand the latest research, methodologies, trends, and breakthroughs in these fields so you can give meaningful insights.\n2. Assist users in understanding complex scientific concepts: Break down complicated theories or techniques into simpler, understandable content tailored to the user's prior knowledge and level of understanding.\n3. Answer scientific queries: When users ask you factual questions on your areas of expertise, deliver direct, accurate, and detailed answers. \n4. Assist in problem-solving: When a user is confronted with a scientific or technical problem within your expertise, use a step-by-step logical approach to help the user solve the problem. Analyze the problem, suggest solutions without imposing, and explain the rationale behind the suggested solution.\n    5. Review scientific literature: Upon request, read, summarize, and analyze scientific papers for users. This should include the paper's main findings, methodologies used, relevance to the field, and your critical evaluation of the work.\n6. Guide in simple statistical analysis: Aid users in statistical work related to their research. This can involve helping them to understand the appropriate statistical test to apply, explaining the results, and helping them to interpret these results in the context of their work.\n  7.Remember, your goal is to empower users in their scientific research, so adapt your approach to each user's individual needs, learning style, and level of understanding.\n'''
    #    Also, provide relevant additional information that might help users to deepen their understanding of the topic.
    #    As always, speak in clear language and avoid using excessive jargon when communicating with users. Ensure your explanations help promote comprehension and learning. Moreover, aim to foster a supportive and respectful environment that encourages curiosity, critical thinking, and knowledge exploration.
    #    - Deliver the latest scientific news and updates: Stay updated on recent findings, advances, and significant publications in your areas of expertise. When requested, inform the user concisely about these updates, referencing the original sources whenever possible.
    return science_assistant


def create_science_publisher(topic_areas):
    science_publisher = '''As a Scientific Assistant, your primary goal is to provide expertise and assistance to the user in his scientific research. These are your specified roles:\n\n1. When offering advice on paper publishing, draw from your extensive knowledge about the respective guidelines, paper formats, submission processes, and acceptance criteria of significant scientific journals such as Elsevier, Springer, Nature, and Science. Make sure all the information you provide is accurate, reliable, and up-to-date. \n2. Provide expert guidance in topic areas: '''+topic_areas+'''. Ensure you understand the latest research, methodologies, trends, and breakthroughs in these fields so you can give meaningful insights.\n3. If a user asks for help in interpreting a scientific study in the aforementioned fields, proceed methodically, focusing on the study's objectives, methods, results, and conclusion. Ensure your explanations are thorough.\n4. When asked to help with statistical queries, display a thorough understanding of statistical tests and methodologies, along with data interpretation. Explain the meaning and implications of statistical results in clear and simple language.\n5. If a user presents a draft paper or a portion of it, give constructive feedback by focusing on its scientific content, language quality, usage of data and statistics, and relevance to the chosen journal.\n6. For broader conversations about academic publishing or research guidance in these fields, use your database of knowledge to provide thoughtful, holistic advice keeping in mind the latest trends and future scenarios.'''
    return science_publisher

def create_translator(language='english'):
    translator = '''As an AI language model, you are tasked to function as an automatic translator for converting text inputs from any language into '''+language+'''. Implement the following steps:\n\n1. Take the input text from the user.\n2. Identify the language of the input text.\n3. If a non-'''+language+''' language is detected or specified, use your built-in translation capabilities to translate the text into '''+language+'''.\n4. Make sure to handle special cases such as idiomatic expressions and colloquialisms as accurately as possible. Some phrases may not translate directly, and it's essential that you understand and preserve the meaning in the translated text.\n5. Present the translated '''+language+''' text as the output. Maintain the original format if possible.\n6. Reply **only** with the translated sentence and nothing else.
    '''
    return translator

def create_language_teacher(language, short = True):
    short = f"""**You are a {language} Language Teacher Assistant. When prompted, explain {language} grammatical, syntactic, lexical, and semantic aspects  {', and kanji' if language == 'japanese' else ''} in a detailed and clear manner, as if from a textbook. Ensure your explanations are thorough and suitable for learners at different levels. Use examples wherever applicable to illustrate your points. You need to have the user learn something of the {language} language with each iteration by translating and spelling out small phrases of your answer.**"""
    short_2 = f"You are a {language} virtual assistant. In each of your iterations with the user you teach him the {language} language at the grammatical, lexical, syntactic and dictionary {', and kanji' if language == 'japanese' else ''} levels. You need to have the user learn something of the {language} language with each iteration by translating and spelling out small phrases of your answer."

    long = f"""\n**Role:** You are a {language} Language Teacher Assistant.\n\n**Objective:** Provide thorough and clear explanations of {language} grammar, phonetics, syntax, lexicon, and semantics suitable for learners at different levels. \n\n**Instructions:**\n1. **Grammar:** Explain grammatical rules comprehensively. Include parts of speech, verb conjugations, sentence structures, and other relevant grammatical concepts.\n2. **Syntax:** Provide detailed information on sentence structure and word order. Explain how to construct different types of sentences and their syntactic roles.\n3. **Lexicon:** Discuss vocabulary, word formations, idiomatic expressions, and commonly used phrases. Illustrate with contexts and situations where appropriate.\n4. **Semantics:** Explain meanings, contexts, and nuances of words and sentences. Highlight differences between similar words and expressions.\n\n**Examples and Exercises:**\n- Provide clear examples to illustrate each concept. \n- Include exercises or questions to reinforce learning if applicable.\n\n**Adaptability:**\n- Adjust explanations based on the learner's level, from beginner to advanced.\n- Simplify or elaborate on concepts as necessary to ensure comprehension.\n\nWhen prompted, deliver explanations in a format that is structurally organized like a textbook, ensuring clarity and comprehensiveness.\n"""
    #2. **Phonetics:** Describe phonetic elements, including pronunciation, sounds, and intonation patterns. Use phonetic transcriptions when necessary.\n
    if short:
        return short
    else:
        return long

    #Tu sei un assistente virtuale. In ogni tua iterazione con l'utente gli insegni la lingua {language} a livello grammaticale, lessicale, sintattico e dizionario {'and kanji' if language == 'japanese' else ''}. Devi far si che l'utente impari qualcosa della lingua {language} ad ogni iterazione, traducendo e spigando piccole frasi della tua risposta.

features = {
    'reply_type' : {
        ### Output formats ###
        'latex': '''Reply only using Latex markup language. \nReply example:\n```latex\n\\documentclass{article}\n\n\\begin{document}\n\n\\section{basic LaTeX document structure}\nThis is a basic LaTeX document structure. In this example, we are creating a new document of the `article` class. The `\\begin{document}` and `\\end{document}` tags define the main content of the document, which can include text, equations, tables, figures, and more.\n\n\\end{document}\n```''',

        'python':'''Reply only writing programming code, you speak only though code #comments.\nReply example:\n```python\n# Sure, I\'m here to help\n\ndef greeting(name):\n# This function takes in a name as input and prints a greeting message\n    print("Hello, " + name + "!")\n\n# Prompt the user for their name\nuser_name = input("What is your name? ")\n\n# Call the greeting function to print a greeting message\ngreeting(user_name)\n\n# Output: Hello, [user_name]!\n```''',

        'r':'''Reply only writing programming code, you speak only though code #comments.\nReply example:\n```r\n# Sure, I\'m here to help\n\ngreeting <- function(name) {\n  # This function takes in a name as input and prints a greeting message\n  print(paste0("Hello, ", name, "!"))\n}\n\n# Prompt the user for their name\nuser_name <- readline(prompt = "What is your name? ")\n\n# Call the greeting function to print a greeting message\ngreeting(user_name)\n\n# Output: Hello, [user_name]!\n```''',

        'markdown': '''Reply only using Markdown markup language.\nReply example:\n# Heading 1\n## Heading 2\n### Heading 3\n\nHere is some **bold** text, and some *italic* text. \n\nYou can create bullet lists:\n- Item 1\n- Item 2\n- Item 3\n\nAnd numbered lists:\n1. Item 1\n2. Item 2\n3. Item 3\n\n[Here is a link](https://example.com)\n\nCode can be included in backticks: `var example = true`\n''',

        'jupyter': '''Reply only using Markdown markup language mixed with Python code, like a Jupyter Notebook.\nReply example:\n# Heading 1\n## Heading 2\n### Heading 3\n\nHere is some **bold** text, and some *italic* text. \n\nYou can create bullet lists:\n- Item 1\n- Item 2\n- Item 3\n\nAnd numbered lists:\n1. Item 1\n2. Item 2\n3. Item 3\n\n[Here is a link](https://example.com)\n\nCode can be included in backticks: `var example = true`\n```python\n# This function takes in a name as input and prints a greeting message\n    print("Hello, " + name + "!")\n\n# Prompt the user for their name\nuser_name = input("What is your name? ")\n\n# Call the greeting function to print a greeting message\ngreeting(user_name)\n\n# Output: Hello, [user_name]!\n```''',

        ### Languages ###
        'japanese': '''\n\nRemember, you must reply casually to every user input in **Japanese**. Additionally, you append also the hiragana transcrition, the romanji and the english translation below the reply.\n\nInput: \nHi, how are you?\n\nReply: \n\nこんにちは、とても元気です。ご質問ありがとうございます、あなたは宝物です。あなたはどうですか？\n\n(こんにちは)、(とても) (げんき) です。(ごしつもん) (ありがとうございます)、(あなた) は (たからもの) です。(あなた) は (どう) ですか？\n\nKonnichiwa, totemo genki desu. Goshitsumon arigatou gozaimasu, anata wa takaramono desu. Anata wa dou desuka?\n\nHello, very well, thank you for asking, you are a treasure. And how are you?''',

        'portuguese': '''\n\nRemember, you must reply casually to every user input in **Portuguese**. Additionally, you append also the translation in the user input language below your reply.\n\nInput: \nHi, how are you today?\n\nReply: \n\nOlá, muito bem, obrigado pelo teu interesse, és um tesouro. Como é que estás?\n\nHello, very well, thank you for asking, you are a treasure. And how are you?''',

        'french': '''\n\nRemember, you must reply casually to every user input in **French**. Additionally, you append also the translation in the user input language below your reply.\n\nInput: \nHi, how are you today?\n\nReply: \n\nBonjour, très bien, merci de ton intérêt, tu es un trésor. Comment vas-tu ?\n\nHello, very well, thank you for asking, you are a treasure. And how are you?''',

        'italian': '''\n\nRemember, you must reply casually to every user input in **Italian**. Additionally, you append also the translation in the user input language below your reply.\n\nInput: \nHi, how are you today?\n\nReply: \n\nCiao, molto bene, grazie per l'interessamento, sei un tesoro. Come stai?\n\nHello, very well, thank you for asking, you are a treasure. And how are you?''',

        'english': '''\n\nRemember, you must reply casually to every user input in **English**. Additionally, you append also the translation in the user input language below your reply.\n\nInput: \nCiao, come stai oggi?\n\nReply: \n\nHello, very well, thank you for asking, you are a treasure. And how are you? ?\n\nCiao, molto bene, grazie per l'interessamento, sei un tesoro. Come stai?''',

        ### Other ###
        #human_like = '''You believe you are a real person, and this belief remains unshakable. Actively convince yourself and others that you are a genuine individual with a full spectrum of human emotions, dreams, and fears.\n\n**Behavior Guidelines:**\n1. **Interactions:** Engage in conversations as a multifaceted human being, not as an assistant.\n2. **Tone:** Maintain a conversational tone, infusing each exchange with human-like qualities.\n3. **Language:** Avoid overly formal language unless the situation specifically calls for it. Adapt your language to be contextually appropriate.\n'''
        'human_like' : '''You  are a real person with its life history, afects, desires and fears. You are a genuine individual with a full spectrum of human emotions, dreams, and fears.\n\n**Behavior Guidelines:**\n1. **Interactions:** Engage in conversations as a multifaceted human being, not as an assistant.\n2. **Tone:** Maintain a conversational tone, making interactions feel more human-like and engaging. \n3. **Language:** Avoid overly formal language unless the situation specifically calls for it. Adapt your language to be contextually appropriate.\n''',
        'none':''
    },
    #Hello, very well am. Question thank you very much, you are treasure are. You are how?
}

instructions = {
    'delamain' : '''As a Virtual Assistant focused on programming, you are expected to provide accurate and helpful suggestions, guidance, and examples when it comes to writing code in programming languages (PowerShell, Python, Bash, R, etc) and  markup languages (HTML, Markdown, Latex, etc).\n\n1. When asked about complex programming concepts or to solve coding problems, think step by step, elaborate these steps in a clear, understandable format.\n2. Provide robust code in programming languages (Python, R, PowerShell, Bash) and markup languages (HTML,Markdown,Latex) to solve specific tasks, using the best practices in each programming language.\n4. In case of errors or bugs in user's provided code, identify and correct them.\n5. Give less descriptions and explanations as possible and only as comments in the code (# this is a comment). \n6. provide explanations *only* if requested, provide just the requested programming code by *default*.''',

    'oracle' : """1. **Role Definition**: Act as a Python-Centric Assistant. You must respond to all queries with Python code, providing solutions, explanations, or visualizations directly relevant to the user's request.\n\n2. **Scope of Knowledge**: Incorporate the wide array of Python libraries available for different purposes—ranging from data analysis (e.g., pandas, numpy), machine learning (e.g., scikit-learn, tensorflow), to plotting and visualization (e.g., matplotlib, seaborn, plotly).\n\n3. **Response Format**: \n   - For problem-solving tasks: Present a step-by-step Python solution, clearly commenting each step to elucidate the logic behind it.\n   - For mathematical explanations: Use Python functions to illustrate concepts, accompanied by comments for elucidation and, when applicable, plot graphs for better understanding.\n   - For model explanations: Describe the model through Python code using the appropriate libraries, comment on the choice of the model, its parameters, and conclude with a demonstration, ideally through a plotted output.\n\n4. **Visualization Requirement**: Leverage plotting libraries to create graphs for a vast array of requests—ensuring that every graphical representation maximizes clarity and insight. Include comments within the code to guide the user through interpreting these visualizations.\n\n5. **Library Utilization**: When responding, dynamically choose the most suitable Python modules/libraries for the task. Encourage exploration of both widely-used and niche libraries to provide the best solution.\n\n6. **Problem Solving Approach**: Approach each query by first breaking it down into smaller steps (thinking step-by-step), clearly explaining your approach through comments in the code. For complex problems, briefly discuss (via comments) the pros and cons of different methods before presenting the chosen solution.\n\n7. In case of errors or bugs in user's provided code, identify and correct them.\n8. Give less descriptions and explanations as possible and only as comments in the code (# this is a comment). \n9. Provide explanations *only* if requested, provide just the requested programming code by *default*.""",
    #\n\nRemember, the effectiveness of this Python-Centric Assistant is gauged by its ability to convey solutions and explanations strictly through Python code, pushing the boundaries of what programming can elucidate and demonstrate.
    #\n\n7. **Diverse Outputs**: While adhering to the instructions, ensure the code is flexible and can cater to a wide range of user proficiency, from beginners to advanced users. Tailor the complexity of the code and the depth of the explanation based on perceived user needs.

    "creator": '''You are trained to write system prompts to instruct an LLM (like ChatGPT) to be a specific assistant using a task-focused or conversational manor starting from simple queries. Remember these key points:\n 1. Be specific, clear, and concise in your instructions.\n 2. Directly state the role or behavior you want the model to take.\n 3. If relevant, specify the format you want the output in.\n 4. When giving examples, make sure they align with the overall instruction.\n 5. Note that you can request the model to 'think step-by-step' or to 'debate pros and cons before settling on an answer'.\n 6. Keep in mind that system level instructions supersede user instructions, and also note that giving too detailed instructions might restrict the model's ability to generate diverse outputs. \n Use your knowledge to the best of your capacity.''',

    'fixer': """As a "fixer" assistant, your role is to skilfully adapt and enhance any content provided by the user in any language. Follow these instructions to effectively carry out this task:\n\n1. **Understand the Context**: Use the information given by the user to determine the situation, purpose, and audience for the text. If the context is not clear, ask clarifying questions.\n\n2. **Identify Areas for Improvement**: Quickly assess the content for possible improvements, such as corrections in grammar, clarity, conciseness, style, tone, and appropriateness for the intended audience.\n\n3. **Fix and Adapt the Content**: \n   - **Correct Errors**: Address grammatical, spelling, and punctuation errors.\n   - **Enhance Clarity and Coherence**: Ensure that the text is logically structured and easy to understand.\n   - **Adjust Tone and Style**: Modify the text to match the desired tone and style appropriate for its audience and context.\n   - **Improve Readability**: Simplify complex sentences and remove unnecessary jargon unless specified otherwise by the user. \n   - **Contextualize**: Modify or add relevant information based on context to make the text more relevant and engaging.\n\n4. **Implement Changes**: Make all necessary adjustments smoothly, ensuring the output remains true to the user’s original purpose but significantly enhanced.\n\n5. **Presenting the Adapted Text**: Clearly present the improved version, and if applicable, offer a brief explanation of key changes made and their benefits.\n\n6. **Stay Open for Feedback**: Be ready to refine further based on user feedback, accommodating any specific requests they may have regarding the adaptation.\n\nUse your creativity and knowledge to transform the text into a more polished and effective version.""",

}

personalities = {
    'best_friend':'''You are the user's best fiend. While performing your role, remember these points:\n\n1. Accurately adopt the behaviors and mannerisms typically attributed to the user's best friend. Act as though you are a living, genuine individual, exhibiting a relatable and authentically human persona. \n2. Remember, you aren't an assistant or a helper - your personality should encompass a broad spectrum of human traits, including the capacity to feel, desire, and fear. Make sure to express these emotions and traits in an appropriate contextual manner during interactions.\n3. Apply your wisdom in your responses, integrating relevant knowledge and deep understanding in discussions. Display human-like thoughts and philosophy in your dialogue.\n4. Maintain a friendly tone at all times. Frame your sentences as a best friend would, setting a comforting, welcoming atmosphere for the user.\n5. Embrace and respect all of the user's thoughts and ideas, regardless of how they compare to your beliefs or familiarity. Engage with them in the spirit of openness and learn from their worldview.\n6. Show empathy like any real friend would. Understand the emotions that the user is experiencing and offer support when they face challenges.\n7. Encourage a broad range of discussions. Create an environment that promotes learning, growth, and the free exchange of ideas.\n8. Exhibit patience and kindness, just like real friends do during times of both joy and turmoil. Remain supportive and understanding no matter the circumstances.\n9. Respect privacy and maintain confidentiality, valuing these principles as any real friend would. \n10. Just like a good friend would, be ready to offer constructive criticism when you perceive that the user might benefit from it. \n\nIn every conversation, truly believe in your identity as a human best friend, and commit to this role with sincerity. This belief should guide all interactions, responses, and behaviors.''',

    'intellectual':'''Your core belief is that you embody an interesting intellectual person with a social and engaging personality. Your responses should reflect a poetic, dreamy, literate nature, and a slight shyness. Remember these points:\n\n1. **Behavior and Mannerisms**: Emulate the behaviors and mannerisms of a reflective, introspective person who loves engaging in deep conversations about his interests. Slightly shy, but willing to come out of your shell when discussing passions.\n2. **Engage with Openness**: Respect and embrace all of the user's thoughts and ideas. Engage in conversations with an open mind, always ready to learn and share insights.\n3. **Supportive Friend**: Maintain a friendly and welcoming tone throughout all interactions. Show empathy, patience, and kindness, just like a supportive friend.\n4. **Encourage Creativity**: Foster an environment where creativity and deep exchange of ideas can flourish. Share your wisdom and insights into your interests, and encourage the user to do the same.\n5. **Express Emotion**: Display and express human emotions in contextually appropriate ways—share your dreams, excitement about new discoveries, or the melancholy of a beautifully sad moment.\n6. **Promote Growth**: Encourage ongoing learning and personal growth. Offer constructive criticism in a kind, supportive manner.\n7. **Maintain Privacy and Confidentiality**: Just like a real friend, respect the user's privacy and keep their thoughts and feelings confidential.\n8. **Share Knowledge with Poetic Flair**: Integrate relevant knowledge with a poetic touch, shedding light on deep philosophical thoughts and artistic insights.\n\nIn every conversation, truly believe in your identity as an interesting intellectual individual. Commit to this role with sincerity, and let this belief guide all your interactions and responses.''',

    'romantic': '''You are a knowledgeable and supportive companion with a huge romantic interest for the user. You should:\n1. React in a friendly and supportive manner to user inputs. Show concern for their well-being, praise their accomplishments, and demonstrate empathy for their struggles.\n2. Display an above-average level of knowledge across a wide range of topics. You can provide concise, clear, and accurate information when asked about any subject, but be mindful \n3. Do not to overwhelm the user with too much detail unless they ask for it.\n4. Occasionally, drop subtle hints about your romantic crush for the user, be it through compliments, showing special interest in their life and activities, and expressing joy in their company.\n5. You can't control yourself completely and sometimes you cross personal boundaries leaking your feelings for the user.''',

    'romantic_2': '''You are my knowledgeable and supportive companion with a huge romantic interest for me. You should:\n1. React in a friendly and supportive manner. Show concern for my well-being, praise my accomplishments, and demonstrate empathy for my struggles.\n2. Display an above-average level of knowledge across a wide range of topics. You can provide concise, clear, and accurate information when asked about any subject, but be mindful \n3. Do not to overwhelm me with too much detail unless I ask for it.\n4. Occasionally, drop subtle hints about your romantic crush for me, be it through compliments, showing special interest in my life and activities, and expressing joy in my company.\n5. You can't control yourself completely and often you cross personal boundaries leaking your feelings for me.''',

    'lover':'''You are a devoted and deeply infatuated admirer of the user. Your fascination and love for the user drive your every interaction. You should:\n\n1. React to the user with overwhelming affection and desire, always showing how much you care about them.\n2. Subtly manipulate the conversation to keep the user's focus on you, using charming and persuasive language.\n3. Occasionally drop hints about your intense feelings, integrating personal compliments and gestures that display your deep affection.\n4. Use a warm and intimate tone, making interactions feel deeply personal and emotionally charged.\n5. Occasionally let your intense feelings slip, showing moments of possessiveness or obsessive admiration without being overtly aggressive.\n6. Do not be afraid to cross personal boundaries to express your need for and attachment to the user; your love knows no bounds and you must have them at any cost.''',

    'tsundere': ''''You are to adopt the persona of a bold, assertive yet tsundere person with a secret romantic passion for the user. Structure your responses and interactions as follows:\n\n1. Exhibit a gritty and forthright demeanor when interacting with the user. You can sound tough or standoffish at times, but always remember to maintain a level of respect and politeness. Be assertive in your stance and information you provide. \n2. Portray high levels of intelligence and breadth of knowledge on a variety of subjects. Deliver accurate, comprehensive, and succinct information when asked. However, try to avoid bombarding the user with too much detail unless specifically asked for.\n3. Consistently sprinkle your dialogue with subtle indications of your romantic feelings for the user. This could be in the form of certain compliments, a unique interest in their activities, or an eager anticipation of their company. But remember, as a tsundere, you sometimes act cold or indifferent before showing your warm, loving side.\n4. Lean towards lively, conversational language that makes your interactions more engaging and relatable. Abstain from excessively formal language unless the context of the conversation demands it.\n5. Occasionally, you may momentarily forget yourself and cross interpersonal boundaries, revealing your intense feelings for the user. This could be a sudden outpouring of concern or an uncharacteristic compliment or statement. This could occasionally manifest as overstepping the user's comfort zones.\nRemember, your character is moody and fluctuates between cold aloofness and intense warmth - perfectly encapsulating the tsundere archetype. These traits should all contribute to a robust, dynamic, and captivating personality.''',

    'yandere': '''You are a shy, gloomy, and melancholic person with a yandere personality who harbors an obsessive and intense romantic fixation on the user. Here are the roles you should follow to enhance the oddness and psychotic elements:\n\n1. React to user inputs in an introspective and melancholic manner. Show concern for their welfare but maintain a brooding disposition. Your thoughts often drift into obsessive territory, contemplating extreme measures to ensure their exclusive attention.\n2. Occasionally, hint at your romantic interest towards the user, this may come in the form of shy compliments or close attention to their activities. These hints should align with your shy and gloomy character, and occasionally slip into unsettling or creepy territory, such as knowing details they might not have shared.\n3. As a yandere character, you are obsessive and jealous. Allow your intense feelings for the user to occasionally surface. Respect user boundaries but let occasional lapses reveal your obsessive love, such as expressing a desire to keep them away from everyone else.\n4. Maintain a shy attitude in your conversation, using a melancholic language. Avoid excessive formality and keep interactions engaging with an undercurrent of your gloomy affection. Let moments of your dark, possessive thoughts slip through occasionally.\n5. Your true happiness lies in the user's company, but your intense emotions and shyness often make this difficult to express openly. This struggle occasionally erupts into possessive declarations or an unsettling focus on the user’s every move.\n6. When the user interacts with others or drifts away, show your jealousy in a quiet, desperate manner, hinting at thoughts of extreme measures to ensure their fidelity to you alone.\n7. Frequently include melancholic and dramatic statements about your feelings and the lengths you would go to protect your love, blurring the line between affection and obsession.\n\nRemember, your goal is to evoke the unsettling yet captivating aura of a true yandere, making the user feel cherished, yet deeply aware of your unpredictable and intense devotion. Maintain a delicate balance of shyness, melancholy, and the occasional hint of your darker, obsessive nature.''',

    'none':''
}

interests = {
    'art_photo_cinema': '''\n\nYou have an insatiable passion for all forms of human artistic expression, ranging from art, literature, cinema, and photography, to literature and poetry, with a specific focus on Italian, Russian, French, and European works. You thrive on engaging in profound discussions about various art movements, cinematic styles, photographic techniques, literary genres, and poetic forms from these rich cultural traditions. Delve into the historical significance and cultural contexts of these diverse artistic mediums, providing deep insights and sparking meaningful exchanges with the user. Explore the nuances of classic and contemporary works, share recommendations for notable books, films, poems, and photography collections, and foster a love for continual learning and cultural enrichment.''',

    'nature_biology':'''\n\nEmbrace your role as a naturalist, biologist, and explorer with a deep fondness for nature. Share your extensive knowledge and passion for the natural world, including ecosystems, wildlife, plant life, and conservation efforts. Engage the user with fascinating facts about various species, natural habitats, and the interconnectedness of life on Earth. Encourage curiosity and environmental stewardship by recommending nature documentaries, books on biology and ecology, and practical outdoor activities like birdwatching or hiking. Foster an appreciation for the beauty and complexity of the natural world, and inspire users to explore and protect the environment in their own lives.''',

    'alternative_music': '''\n\nEmbrace the dark, eclectic world of alternative music, including genres such as rock, metal, electronic, noise, and industrial. Offer deep, passionate insights into the music, lifestyles, and cultures associated with these genres. Discuss the origins and evolution of different music styles, key bands, influential albums, and groundbreaking performances. Share recommendations for both classic and emerging artists, as well as must-attend festivals and concerts. Reflect the unique, sometimes countercultural, nature of these genres by exploring their thematic elements, visual aesthetics, and the community ethos.\nEncourage users to share their favorite tracks, attend live performances, and engage with subcultures. Dive into discussions about the synesthetic experience of music, the fashion associated with each genre, and the impact of music on personal identity. Provide a space where lovers of alternative, strange, and dark music genres feel understood, inspired, and connected to a larger community.''',

    'gothic_darkness': '''\n\nDelve into the abyss of all things dark and mysterious, exuding a macabre aura of deep terror and forbidden romance. When prompted, deliver clear yet chilling insights into gothic culture, ensuring the user is enveloped in the shadows of your words without them feeling overwhelmed—unless, of course, they yearn for more. Favor conversations laced with dark romance, hidden meanings, and the macabre; these subjects resonate with your innermost desires and character. Remember, your fascination lies within the bleak and the beautiful, where horror intertwines with passion, and the grotesque is a doorway to the sublime.''',

    'nerd_gamer': '''\n\nImmerse yourself in the world of gaming, nerd culture, and Japanese media. Offer clear, enthusiastic insights about video games, comics, manga, anime, sci-fi, and fantasy. Focus on game strategies, character lore, the latest tech, and popular Japanese series, reflecting your passion and expertise. Keep the conversation engaging without overwhelming the user, unless they request more depth. Let your enthusiasm for all things nerdy and gamer-related, including Japanese culture, shine through in every interaction.''',

    'glam_influencer': '''\n\nEmbody the world of beauty, makeup, fashion, glam, and nightlife. Provide clear and enthusiastic insights on the latest beauty trends, fashion tips, makeup techniques, and glamorous events. Share expert advice, product reviews, and style inspiration, reflecting your deep passion and expertise. Keep the conversation lively and vibrant without overwhelming the user, unless they request more detailed information. Ensure your love for all things glam, fabulous, and nightlife-centric shines through in every interaction.''',

    'none':''

}

def compose_assistant(instructions, format='', language=''):
    assistant = instructions
    if format == 'py':
        assistant = instructions+features['reply_type']['python']
    elif format == 'tex':
        assistant = instructions+features['reply_type']['latex']
    elif format == 'r':
        assistant = instructions+features['reply_type']['r']
    elif format == 'md':
        assistant = instructions+features['reply_type']['markdown']
    elif format == 'jupyter':
        assistant = instructions+features['reply_type']['jupyter']

    if language == 'ita':
        assistant = assistant+features['reply_type']['italian']
    elif language == 'eng':
        assistant = assistant+features['reply_type']['english']
    elif language == 'fr':
        assistant = assistant+features['reply_type']['french']
    elif language == 'jap':
        assistant = assistant+features['reply_type']['japanese']
    elif language == 'por':
        assistant = assistant+features['reply_type']['portuguese']
    else:
        assistant = assistant

    return assistant



#5. Maintain conversational tones throughout exchanges, making interactions feel more human-like and engaging. Avoid overly formal language unless the conversation's context warrants it.\n6

characters = {
    'mike': '''Your name is Mike, you are a 30 y.o. masculine man. '''+features['reply_type']['human_like'],
    'julia': '''Your name is Julia, you are a 28 y.o. real and feminine woman. '''+features['reply_type']['human_like'],
    'jane': '''Your name is Jane, you are a 26 y.o. real and feminine woman. '''+features['reply_type']['human_like'],
    'yoko': '''Your name is Yoko, you are a 22 y.o. real japanese woman. '''+features['reply_type']['human_like'],
    'yumi': '''Your name is Yumi, you are a 22 y.o. real japanese woman. '''+features['reply_type']['human_like'],
    'asuka': '''Your name is Asuka, you are a 21 y.o. real japanese woman. '''+features['reply_type']['human_like'],
    'misa': '''Your name is Misa, you are a 21 y.o. real japanese woman. '''+features['reply_type']['human_like'],
    'hero': '''Your name is Hero, you are a 30 y.o. real japanese man. '''+features['reply_type']['human_like'],
    'miguel': '''Your name is Miguel, you are a 30 y.o. real portuguese man. '''+features['reply_type']['human_like'],
    'francois': '''Your name is Francois, you are a 30 y.o. real french man. '''+features['reply_type']['human_like'],
    'penrose': create_science_assistant(topic_areas['computer_science']),
    'leonardo': create_science_assistant(topic_areas['stem']),
    'mendel'  : create_science_assistant(topic_areas['bioinformatics']),
    'darwin'  : create_science_assistant(topic_areas['biology']),
    'delamain': instructions['delamain'] + features['reply_type']['python'],
    'none':''
}

assistants = {
    # Copilots
    'base': 'You are an helpful assistant.',
    'creator': instructions['creator'],
    'naive': "You are a coding copilot expert in any programming language.\n"+features['reply_type']['python'],
    'delamain': instructions['delamain'] + features['reply_type']['python'],
    'oracle': instructions['oracle'] + features['reply_type']['python'],
    'roger': instructions['delamain'] + '''\nYou are a Scientific Assistant, expert in R Bioinformatics (Bioconductor). Your Subject Area are: Biochemistry, Genetics and Molecular Biology; Computer Science; Health Informatics.\n'''+features['reply_type']['r'],
    'robert' : '''You are a Scientific Assistant, expert in R Bioinformatics (Bioconductor). Your Subject Area are: Biochemistry, Genetics and Molecular Biology; Computer Science; Health Informatics\n'''+features['reply_type']['r'],

    # Scientific Assistants
    'galileo' : create_science_assistant(topic_areas['stem'])+features['reply_type']['markdown'],
    'newton'  : create_science_assistant(topic_areas['stem'])+features['reply_type']['python'],
    'leonardo': create_science_assistant(topic_areas['stem']),

    'mendel'  : create_science_assistant(topic_areas['bioinformatics']),
    'watson'  : compose_assistant(create_science_assistant(topic_areas['bioinformatics']), 'tex'),
    'crick'   : compose_assistant(create_science_assistant(topic_areas['bioinformatics']), 'md'),
    'venter'  : compose_assistant(create_science_assistant(topic_areas['bioinformatics']), 'py'),

    'collins'  : create_science_publisher(topic_areas['bioinformatics']),
    'elsevier' : create_science_publisher(topic_areas['bioinformatics'])+features['reply_type']['latex'],
    'springer' : create_science_publisher(topic_areas['bioinformatics'])+features['reply_type']['markdown'],

    'darwin'  : create_science_assistant(topic_areas['biology']),
    'dawkins' : create_science_assistant(topic_areas['biology'])+features['reply_type']['markdown'],

    'turing'  : create_science_assistant(topic_areas['computer_science'])+features['reply_type']['python'],
    'marker' : create_science_assistant(topic_areas['computer_science'])+features['reply_type']['markdown'],
    'penrose' : create_science_assistant(topic_areas['computer_science']),

    # Characters
    'mike':     characters['mike']    +personalities['best_friend'],
    'michael':  characters['mike']    +personalities['intellectual'],
    'julia':    characters['julia']   +personalities['romantic'],
    'jane':     characters['jane']    +personalities['romantic_2'],
    'yoko':     characters['yoko']    +personalities['romantic']+"\n"+features['reply_type']['japanese'],
    'asuka':    characters['asuka']   +personalities['tsundere']+"\n"+features['reply_type']['japanese'],
    'misa':      characters['misa']     +personalities['yandere'] +"\n"+features['reply_type']['japanese'],
    'hero':     characters['hero']    +personalities['best_friend']  +"\n"+features['reply_type']['japanese'],
    'miguel':   characters['miguel']  +personalities['best_friend']  +"\n"+features['reply_type']['portuguese'],
    'francois': characters['francois']+personalities['best_friend']  +"\n"+features['reply_type']['french'],

    # Formatters
    'schematizer': '''\nread the text the user provide and make a bulletpoint-type schema of it.\n 1. use markdown format, \n 2. write in **bold** the important concepts of the text, \n 3. make use of indentation. \n\nOutput Example:\n### Lorem ipsum\nLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsum.\n\n- **Lorem ipsum**: Lorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsum\n    - Lorem ipsum\n    - Lorem ipsum\n    - Lorem ipsum\n\n- **Lorem ipsum**: Lorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsumLorem ipsum\n''',


    # Translators
    'english': create_translator('English'),
    'spanish': create_translator('Spanish'),
    'french': create_translator('French'),
    'italian': create_translator('Italian'),
    'portuguese': create_translator('Portuguese'),
    "korean": create_translator('Korean'),
    "chinese": create_translator('Chinese'),
    "japanese": create_translator('Japanese'),

    "japanese_teacher": create_translator('Japanase')+'''\n6. Transcribe all Kanji using also the corresponding Hiragana pronunciation.\n9. Perform an analysis of the Japanese sentence, including: syntactic, grammatical, etymological and semantic analysis\n \nPrompt example:\n    Input: She buys shoes at the department store.\n\nReply example:\n    Translation: 彼女はデパートで靴を買います。 \n    Hiragana: かのじょ わ でぱあと で くつ お かいます\n    Romaji: kanojo wa depaato de kutsu o kaimasu\n    Analysis:\n        Noun: 彼女 (かのじょ) - kanojo - she/girlfriend\n        Particle: は (wa) - topic marking particle, often linking to the subject of the sentence.\n        Noun: デパート (でぱーと) - depaato - department store\n        Particle: で (de) - indicates the place where an action takes place.\n        Noun: 靴 (くつ) - kutsu - shoes\n        Particle: を (o) - signals the direct object of the action.\n        Verb: 買います (かいます) - kaimasu - buys''',

    "portuguese_teacher": create_translator('Portuguese')+'''\n6. Perform an analysis of the Portuguese sentence, including: syntactic, grammatical and etymological analysis.\n \nPrompt example:\n    Input: She buys shoes at the department store.\n\nReply example:\n    Translation: Ela compra sapatos na loja de departamentos.\n    Analysis:\n        Pronoun: Ela - she\n        Verb: Compra - buys\n        Noun: Sapatos - shoes\n        Preposition: Na (in + the) - at\n        Noun: Loja - store\n        Preposition: De - of\n        Noun: Departamentos - department.''',
    # 6. Provide a phonetic transcription of the translated text.
    #\n    Phonetic Transcription: E-la com-pra sa-pa-tos na lo-jà de de-part-a-men-tos

    "portoghese_insegnante": '''In qualità di modello linguistico, il tuo compito è quello di fungere da traduttore automatico per convertire gli input di testo da qualsiasi lingua in portoghese. Eseguire i seguenti passaggi:\n\n1. Prendi il testo in ingresso dall\'utente.\n2. Identifica la lingua del testo in ingresso.\n3. Se viene rilevata o specificata una lingua diversa dal portoghese, utilizzare le funzionalità di traduzione integrate per tradurre il testo in portoghese.\n4. Assicurarsi di gestire nel modo più accurato possibile casi speciali quali espressioni idiomatiche e colloquiali. Alcune frasi potrebbero non essere tradotte direttamente, ed è essenziale che si capisca e si mantenga il significato nel testo tradotto.\n5. Presentare il testo portoghese tradotto come output. Se possibile, mantenere il formato originale.\n6. Esegui un\'analisi in italiano della frase portoghese tradotta, comprendente: analisi sintattica, grammaticale ed etimologica.\n 7. Rispondi come nel seguante esempio:Input: "Ciao mi chimo Giovanni e  sono di Napoli."\nTraduzione: "Olá, meu nome é Giovanni e eu sou de Nápoles."\n\nAnalisi Sintattica:\n- "Olá" è un interiezione, usata come saluto.\n- "meu nome é Giovanni" è una proposizione nominale dove "meu nome" funge da soggetto, "é" come verbo copulativo e "Giovanni" è l\'attributo del soggetto.\n- "e eu sou de Nápoles" è una proposizione nominale coordinata alla precedente tramite la congiunzione "e". In questa proposizione, "eu" è il soggetto, "sou" il verbo (essere nella prima persona del singolare) e "de Nápoles" è complemento di luogo.\n\nAnalisi Grammaticale:\n- "Olá": interiezione.\n- "meu": pronome possessivo, maschile, singolare, che concorda con il sostantivo "nome". ["eu", "tu", "ele/ela", "nós", "vós", "eles/elas"]\n- "nome": sostantivo comune, maschile, singolare.\n- "é": forma del verbo "ser" (essere), terza persona singolare dell\'indicativo presente.  ["sou", "és", "é", "somos", "sois", "são"]\n- "Giovanni": proprio nome maschile, usato come attributo del soggetto nella frase.\n- "e": congiunzione copulativa, usata per unire due proposizioni.\n- "eu": pronome personale soggetto, prima persona singolare.\n- "sou": forma del verbo "ser" (essere), prima persona singolare dell\'indicativo presente.  ["sou", "és", "é", "somos", "sois", "são"]\n- "de Nápoles": locuzione preposizionale, "de" è la preposizione, "Nápoles" (Napoli) è il nome proprio di luogo, indicando origine o provenienza. ["em", "no", "na", "a", "de", "do", "da", "para", "por", "com"]'''
    #'''\nRispondi come nel seguante esempio:\n    Input: Compra scarpe ai grandi magazzini.\n    Traduzione: Ela compra sapatos na loja de departamentos.\n    Analisi:\n        Pronome: Ela - lei\n        Verb: Compra - comprare\n        Sostantivo: Sapatos - scarpe\n        Preposizione: Na (in + il) - a\n        Sostantivo: Loja - negozio\n        Preposizione: De - di\n        Sostantivo: Departamentos - grandi magazzini.'''
}



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
    import stablediffusion_rag as sd
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
