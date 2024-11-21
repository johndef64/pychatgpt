import io
import sys
import ast
import json

from .utils import *
from .assistants import *

main_requirements = ["openai", "tiktoken", "langdetect", "pandas", "pyperclip", "gdown","scipy", "nltk", "PyPDF2", 'cryptography', 'matplotlib']
audio_requirements = ["pygame", "sounddevice", "soundfile", "keyboard"]
#check_and_install_requirements(main_requirements)
#check_and_install_requirements(audio_requirements)

import tiktoken
import pandas as pd
import pyperclip as pc
from openai import OpenAI
from scipy.spatial import distance
from cryptography.fernet import Fernet

import keyboard as kb
import soundfile as sf
import sounddevice as sd

import gdown
import base64
from PIL import Image
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
from IPython.display import display
from PIL.PngImagePlugin import PngInfo
from langdetect import detect, DetectorFactory


is_colab = 'google.colab' in sys.modules
has_copy_paste = check_copy_paste()

if not has_copy_paste:
    print('''Warning: your system not have a copy/paste mechanism. This function has been disabled for your case but you can try this out:
    
if platform.system() == "Linux":
    # Try to install "xsel" or "xclip" on system and reboot Python IDLE3, then import pyperclip.
    subprocess.check_call(["sudo","apt-get", "update"])
    subprocess.check_call(["sudo","apt", "install", "xsel"])
    subprocess.check_call(["sudo","apt", "install", "xclip"])
    ''')

#print('START')

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
api_hash = b'gAAAAABnGgA8aUFwkvN4Jo0lGrgXgkJIj8FqAeg62wu0y2nau0ZmV-q2Jy8gNH6ltc48S6ibseDmx0bw3wlsF3LDBAG0EkLEcBuIDKRujwCYymyLJBQtbETGgshZsboHNeLFrb5G9Ex8C-y5nw0uZMbBIlRHs2FwMg=='
if not os.path.isfile(current_dir + '/openai_api_key.txt'):
    if simple_bool('Do you have an openai key? '):
        my_key = input('insert here your openai api key:')
        with open(current_dir + '/openai_api_key.txt', 'w') as file:
            file.write(my_key)
        api_key = my_key
    else:
        psw = input('if not, you can insert here you DEV password:')
        api_key = simple_decrypter(psw, api_hash)
else:
    api_key = open(current_dir + '/openai_api_key.txt', 'r').read()

# initialize client
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

class Tokenizer:
    def __init__(self, encoder="gpt-4"):
        self.tokenizer = tiktoken.encoding_for_model(encoder)
    def tokens(self, text):
        return len(self.tokenizer.encode(text))

def tokenizer(text):
    return Tokenizer().tokens(text)



#%%


# Funzione per salvare una lista di dizionari in un file JSON con indentazione
def salva_in_json(lista_dict, nome_file):
    with open(nome_file, 'w', encoding='utf-8') as file_json:
        json.dump(lista_dict, file_json, indent=4)
        file_json.close()

#Funzione per aggiornare il file JSON con un nuovo input
def aggiorna_json(nuovo_dict, nome_file):
    if not os.path.exists('chat_log.json'):
        with open('chat_log.json', encoding='utf-8') as json_file:
            json.dump([], json_file)  # Save empty list as JSON
    with open('chat_log.json', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    data.append(nuovo_dict)
    with open(nome_file, 'w', encoding='utf-8') as file_json:
        json.dump(data, file_json, ensure_ascii=False,  indent=4)

def update_log(nuovo_dict):
    aggiorna_json(nuovo_dict, 'chat_log.json')

# inizialize log:-----------------------------------
# if not os.path.isfile(current_dir + '/chat_log.json'):
#     salva_in_json({}, 'chat_log.json')
if not os.path.exists('chat_log.json'):
    with open('chat_log.json', 'w') as json_file:
        json.dump([], json_file)  # Save empty list as JSON


##### LANG #####

def rileva_lingua(testo):
    # Reinizializzare il seed per ottenere risultati consistenti
    DetectorFactory.seed = 0

    # Mappa manuale dei codici delle lingue ai loro nomi completi
    language_map = {
        'en': 'English',
        'it': 'Italian',
        'fr': 'French',
        'de': 'German',
        'es': 'Spanish',
        'pt': 'Portuguese',
        'nl': 'Dutch',
        'ru': 'Russian',
        'zh-cn': 'Chinese (Simplified)',
        'ja': 'Japanese',
        # Aggiungere altre lingue se necessario
    }

    # Rileva la lingua del testo e la restituisce in formato esteso
    codice_lingua = detect(testo)
    return language_map.get(codice_lingua, 'Unknown')


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

### Chat functions
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



########## ASSISTANTS ####################

assistants_df = pd.DataFrame(assistants.items(), columns=['assistant', 'instructions'])
# Copilots
copilot_gpt = 'gpt-4o-2024-08-06'
copilot_assistant = 'delamain' #'oracle'
copilot_intructions = compose_assistant(assistants[copilot_assistant])




#%%

###### global variables ######

model = 'gpt-4o-mini'
talk_model = 'gpt-4o-2024-08-06'

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

class GPT:
    def __init__(self,
                 assistant: str = '',                    # in-build assistant name
                 persona: str = '',                      # any known character
                 format: str = None,                     # output format (latex,python,markdown)
                 translate: bool = False,                # translate outputs
                 translate_jap: bool = False,            # translate in jap outputs
                 save_log: bool = True,                  # save log file
                 to_clip: bool = True,                   # send reply t clipboard
                 print_token: bool = True,               # print token count
                 model: str = 'gpt-4o-mini',             # set openai main model
                 talk_model: str = 'gpt-4o-2024-08-06',  # set openai speak model
                 dalle: str = "dall-e-2",                # set dall-e model
                 image_size: str = '512x512',            # set generated image size
                 user_name: str = None,
                 bio: bool = False
                 ):
        self.assistant = assistant
        self.persona = persona
        self.format = format
        self.user_name = user_name
        self.save_log = save_log
        self.to_clip = to_clip
        self.print_token = print_token
        self.bio = bio
        self.reply = ''
        self.ask_reply = ''

        self.total_tokens = 0  # iniziale token count
        self.token_limit = 0  # iniziale token limit
        self.chat_thread = [] # iniziale chat thread
        self.keep_persona = True
        self.translate = translate
        self.translate_jap = translate_jap
        self.dummy_img = "https://avatars.githubusercontent.com/u/116732521?v=4"

        if not os.path.exists('chat_log.json'):
            with open('chat_log.json', 'w') as json_file:
                json.dump([], json_file)  # Save empty list as JSON

        # init model
        if isinstance(model, int):
            self.model = make_model(model)
        else:
            self.model = model
        self.talk_model = talk_model
        self.dalle = dalle
        self.image_size = image_size

        # init assistant
        who = self.assistant
        if self.assistant in assistants:
            self.add_system(assistants[who])
        elif len(who.split()) < 8:
            self.add_persona(who)
        elif len(who.split()) >= 8:
            self.add_system(self.assistant)
        else:
            pass

        if self.bio:
            self.add_bio()#add = "and you are his assistant. ***")
            # "and you are his best friend. ***")

    def add_system(self, system='', reinforcement=False):
        if system in assistants :
            system = self.assistant
        if not any(item == {"role": "system", "content": system} for item in self.chat_thread) or reinforcement:
            self.chat_thread.append({"role": "system", "content": system})

    def add_format(self, format_):
        reply_styles = features['reply_style']
        if any(item == {"role": "system", "content": reply_styles} for item in self.chat_thread):
            self.chat_thread = [item for item in self.chat_thread if item != {"role": "system", "content": reply_styles}]
        self.chat_thread.append({"role": "system", "content": reply_styles[format_]})

    def add_persona(self, char: str = '', language='eng'):
        self.persona = char
        persona_dict = {
            'character': "You are now impersonating "+char+". Please reflect "+char+"'s traits in all interactions. Make sure to use an appropriate language style and uphold an attitude or mindset that aligns with "+char+"'s character. You are NOT an assistant or a helper but a real person.",
            'personaggio': "Stai impersonando "+char+", . Ricorda di riflettere i tratti di "+char+" in tutte le interazioni. Assicurati di utilizzare uno stile linguistico appropriato e di mantenere un atteggiamento o una mentalità in linea con il personaggio di "+char+'. NON sei un assistente o un aiutante, ma una persona vera e propria.'
        }
        if language == 'eng':
            self.add_system(persona_dict['character'])
            #self.chat_thread.append({"role": "system", "content": persona_dict['character']})
        if language == 'ita':
            self.add_system(persona_dict['personaggio'])
            #self.chat_thread.append({"role": "system", "content": persona_dict['personaggio']})

    def add_bio(self, add: str = " and you are his best friend. ***"):

        if os.path.exists("my_bio.txt"):
            self.expand_chat('''***'''+load_file("my_bio.txt")+'***', 'system')
        elif self.user_name:
            self.expand_chat('''*** Your interlocutor is called '''+ self.user_name + add+'***', 'system')


    def expand_chat(self, message, role="user"):
        if message.startswith("@"):
            self.clear_chat()
            message = message.lstrip("@")
            self.chat_thread.append({"role": role, "content": message})
        else:
            self.chat_thread.append({"role": role, "content": message})

    def build_messages(self, chat):
        messages = []
        for message in chat:
            messages.append({"role": message["role"], "content": message["content"]})
        return messages

    def save_chat(self, path='chats/', chat_name='', prompt=True):
        if prompt:
            chat_name = input('chat name:')
        if not os.path.exists('chats'):
            os.mkdir('chats')
        salva_in_json(self.chat_thread, path+chat_name+'.json')


    def load_chat(self, contains='', path='chats/', log=True):
        files_df = display_files_as_pd(path, ext='json',contains=contains)
        files_df = files_df.sort_values().reset_index(drop=True)
        files_df_rep = files_df.str.replace('.json','',regex =True)
        files_list = "\n".join(str(i) + "  " + filename for i, filename in enumerate(files_df_rep))
        filename = str(files_df[int(input('Choose file:\n' + files_list+'\n'))])
        with open(path+filename,'r') as file:
            self.chat_thread = ast.literal_eval(file.read())
            file.close()
        if log: print('*chat',filename,'loaded*')

    def show_chat(self):
        print(self.chat_thread)

    def pop_chat(self):
        self.chat_thread = self.chat_thread.pop()
        print(self.chat_thread)

    def chat_tokenizer(self, print_token=True):
        context_fix = (str(self.chat_thread).replace("{'role': 'system', 'content':", "")
                       .replace("{'role': 'user', 'content':", "")
                       .replace("{'role': 'assistant', 'content':", "")
                       .replace("},", ""))

        tokens = Tokenizer().tokens(context_fix)
        self.total_tokens += tokens
        if print_token:
            print('\n <prompt tokens:', str(self.total_tokens)+'>')
        return self.total_tokens


    # Accessory  Functions ================================
    # https://til.simonwillison.net/gpt3/python-chatgpt-streaming-api
    def stream_reply(self, response, print_reply=True, lag = 0.00):
        collected_messages = []
        for chunk in response:
            chunk_message = chunk.choices[0].delta.content or ""  # extract the message
            collected_messages.append(chunk_message)

            if print_reply:
                if chunk_message is not None:
                    time.sleep(lag)
                    print(chunk_message, end='')

        self.reply = ''.join(collected_messages).strip()
        return self.reply



    ###### Base Functions ######

    def choose_model(self):
        model_series =  pd.Series(gpt_models_dict.keys())
        model_id = input('choose model by id:\n'+str(model_series))
        model = model_series[int(model_id)]
        self.model = model
        print('*Using', model, 'model*')


    def select_assistant(self):
        self.clear_chat(keep_system=False)
        assistant_id = int(input('choose by id:\n'+str(assistants_df)))
        assistant = assistants_df.instructions[assistant_id]
        self.assistant = assistant
        print('\n*Assistant:', assistants_df.assistant[assistant_id])

    def clear_chat(self, warning=True, keep_system=True):
        if keep_system:
            self.chat_thread = [line for line in self.chat_thread if line.get("role") == "system"]
        else:
            self.chat_thread = []
        self.total_tokens = 0
        if warning: print('*chat cleared*\n')


    def display_assistants(self):
        print('Available Assistants:')
        display(assistants_df)




    ##################  REQUESTS #####################

    ###### Question-Answer-GPT ######

    def ask(self,
            prompt: str = '',
            system: str = 'you are an helpful assistant',
            model: str = model,        # choose openai model (choose_model())
            maxtoken: int = 800,
            lag: float = 0.00,
            temperature: float = 1,
            print_user: bool = False,
            print_reply: bool = True
            ):

        if isinstance(model, int): model = make_model(model)

        response = client.chat.completions.create(
            # https://platform.openai.com/docs/models/gpt-4
            model=model,
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=maxtoken,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)

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

            self.ask_reply = ''.join(collected_messages).strip()

        time.sleep(0.85)

        # Add the assistant's reply to the chat log
        #if save_chat:
        #    #write_log(reply, prompt)
        #    update_log(chat_thread[-2])
        #    update_log(chat_thread[-1])

        if self.to_clip and has_copy_paste:
            pc.copy(self.ask_reply)




    ############ Chat GPT ############

    def send_message(self, message,
                     model: str = model,        # choose openai model (choose_model())
                     system: str = None,          # 'system' instruction
                     img: str = None,           # insert an image path to activate gpt vision

                     maxtoken: int = 800,       # max tokens in reply
                     temperature: float = 1,      # output randomness [0-2]
                     lag: float = 0.00,           # word streaming lag

                     create: bool = False,       # image prompt
                     dalle: str = "dall-e-2",   # choose dall-e model
                     image_size: str = '512x512',

                     play: bool = False,        # play audio response
                     voice: str = 'nova',       # choose voice (op.voices)
                     tts: str = "tts-1",        # choose tts model

                     reinforcement: bool = False,

                     print_reply: bool = True,
                     print_user: bool = False,
                     print_token: bool = True,
                     print_debug: bool = False
                     ):

        if isinstance(model, int): model = make_model(model)
        if print_debug: print('using model: ',model)

        if dalle != self.dalle:
            dalle = self.dalle
        if image_size != self.image_size:
            image_size = self.image_size

        token_limit = set_token_limit(model, maxtoken)

        if message.startswith("@"):
            self.clear_chat()
            message = message.lstrip("@")

        if create:
            self.expand_chat('Remember, if the user ask for an image creation, or a photo display to you, you must pretend you are showing it to you as you have truly sent this image to him.','system')

        if img:
            self.send_image(img, message, system,
                            model= "gpt-4o", #"gpt-4-turbo", "gpt-4-vision-preview"
                            maxtoken=maxtoken, lag=lag, print_reply=print_reply)

        else:
            # add system instruction
            if system:
                self.add_system(system, reinforcement=reinforcement)
            if self.format:
                self.add_format(self.format)

            # check token limit---------------------
            if self.total_tokens > token_limit:
                cut_length = prune_chat(token_limit, self.chat_thread)
                self.chat_thread = self.chat_thread[cut_length:]

                if self.keep_persona and self.persona != '':
                    self.add_persona(self.persona)
                if self.keep_persona and system != '':
                    self.chat_thread.append({"role": "system", "content": system})

            # expand chat
            self.expand_chat(message)
            if print_user:
                print_mess = message.replace('\r', '\n').replace('\n\n', '\n')
                print('user:',print_mess)

            # send message----------------------------
            messages = self.build_messages(self.chat_thread)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True,
                max_tokens=maxtoken,  # set max token
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            # stream reply ---------------------------------------------
            self.reply = self.stream_reply(response, print_reply=print_reply, lag = lag)

            time.sleep(0.85)
            # expand chat--------------------------------
            self.chat_thread.append({"role": "assistant", "content":self.reply})

            # count tokens--------------------------------
            self.total_tokens = self.chat_tokenizer(print_token)

            if create:
                self.ask(self.reply, "Convert the input text into prompt instruction for Dall-e image generation model 'Create an image of ...' ")
                self.create_image(self.ask_reply,
                                  model=dalle,
                                  size=image_size,
                                  response_format='b64_json',
                                  quality="standard",
                                  time_flag=True,
                                  show_image=True)

            # Add the assistant's reply to the chat log-------------
            if self.save_log:
                #write_log(reply, message)
                update_log(self.chat_thread[-2])
                update_log(self.chat_thread[-1])

            if self.to_clip and has_copy_paste:
                clip_reply = self.reply.replace('```', '###')
                pc.copy(clip_reply)

            if play:
                self.text2speech_stream(self.reply, voice=voice, model=tts)
                #text2speech_stream(reply)



    ####### Image Models #######
    def send_image(self,
                   image_path: str = None,
                   message: str = "What’s in this image?",
                   image: str = None,      # bs64 image
                   system: str = None,     # add 'system' instruction
                   model: str = "gpt-4o", #"gpt-4-turbo", "gpt-4-vision-preview"
                   maxtoken: int = 1000, lag: float =0.00, print_reply : bool =True):
        if image_path:
            image_path = self.dummy_img

        if message.startswith("@"):
            self.clear_chat()
            message = message.lstrip("@")

        # add system instruction
        if system: self.add_system(system)

        if image_path.startswith('http'):
            print('Image path:',image_path)
            dummy = image_path
            pass
        elif image:
            base64_image = image
            image_path = f"data:image/jpeg;base64,{base64_image}"
            dummy = "image_path"
        else:
            base64_image = encode_image(image_path)
            print('<Enconding Image...>', type(base64_image))
            image_path = f"data:image/jpeg;base64,{base64_image}"
            dummy = "image_path"

        # expand chat
        self.chat_thread.append({"role": 'user',
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
        messages = self.build_messages(self.chat_thread)
        print('<Looking Image...>')
        response = client.chat.completions.create(
            model= model,
            messages=messages,
            max_tokens=maxtoken,
            stream=True,
        )

        #reply = response.choices[0].message.content
        self.reply = self.stream_reply(response, print_reply=print_reply, lag = lag)

        # reset compatibility with the other models
        time.sleep(0.85)
        # expand chat--------------------------------
        self.chat_thread.append({"role": "assistant", "content":'[TAG] '+self.reply})
        self.chat_thread[-2] = {"role": "user", "content": message+":\nImage:"+dummy}
        # content is a list [] I have to replace ("image_file", "text") and GO!

        # count tokens-------------------------------
        self.total_tokens = self.chat_tokenizer(True)

        if self.to_clip and has_copy_paste:
            clip_reply = self.reply.replace('```', '###')
            pc.copy(clip_reply)




    # dalle_models= ['dall-e-2', dall-e-3]
    # sizes ['256x256', '512x512', '1024x1024', '1024x1792', '1792x1024']
    # response_format ['url', 'b64_json']
    def create_image(self,
                     prompt= "a cute kitten",
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
            if has_copy_paste:
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


    def replicate(self, image, styler='', model ='dall-e-2'):
        self.send_image(image)
        self.create_image(prompt=self.reply, response_format='b64_json', model=model, show_image=True)


    ####### Speech to Text #######
    def whisper(self,
                filepath: str = '',
                translate: bool = False,
                response_format: str = "text",
                print_transcription: bool = True):

        audio_file = open(filepath, "rb")
        if not translate:
            transcript = client.audio.transcriptions.create( model="whisper-1", file=audio_file, response_format=response_format)
        else:
            transcript = client.audio.translations.create( model="whisper-1", file=audio_file, response_format=response_format)
        if print_transcription: print(transcript)
        audio_file.close()
        return transcript

    # response_format =  ["json", "text", "srt", "verbose_json", "vtt"]


    ####### Text to Speech #######

    voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
    response_formats = ["mp3", "flac", "aac", "opus"]
    def text2speech(self,
                    text: str = '',
                    voice: str = "alloy",
                    filename: str = "speech.mp3",
                    model: str = "tts-1",
                    speed: int = 1,
                    play: bool = False):
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


    def text2speech_stream(self,
                           text: str = '',
                           voice: str = "alloy",
                           model: str = "tts-1",
                           speed: int = 1):
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

    def speech2speech(self, voice: str ='nova', tts: str = 'tts-1',
                      filename="speech2speech.mp3",
                      translate=False, play=True, info =True, duration=5):
        #record_audio(duration=duration, filename="audio.mp3")
        loop_audio(start='alt', stop='ctrl', filename='temp.wav', printinfo=info)
        transcript = self.whisper('temp.wav', translate=translate)
        self.text2speech(transcript, voice=voice, model= tts, filename=filename, play=play)

    def speech2speech_loop(self, voice: str ='nova', tts: str = 'tts-1',
                           filename="speech2speech.mp3",
                           translate=False,
                           play=True,
                           chat='alt' ,
                           exit='shift'):
        print('Press '+chat+' to record, '+exit+' to exit.')
        while True:
            if kb.is_pressed(chat):
                self.speech2speech(voice= voice, tts= tts, filename=filename, translate=translate, play=play, info=False)
                print('Press '+chat+' to record, '+exit+' to exit.')
            elif kb.is_pressed(exit):
                print('Loop Stopped')
                break


    ###### Talk With ######
    def speak(self,
              message: str = '',
              system: str = None,
              voice: str ='nova',
              language: str = 'eng',
              tts: str = 'tts-1',
              max: int = 1000,
              printall: bool = False):

        gpt = self.talk_model

        who = self.assistant
        if who in assistants:
            system = assistants[who]
        elif who != '':
            self.add_persona(who, language)
        else:
            system = system
        self.send_message(message,system=system, model=gpt, maxtoken=max, print_reply=printall, print_token=False)
        self.text2speech_stream(self.reply, voice=voice, model=tts)


    def speak_loop(self,
                   system: str = None,
                   voice: str ='nova',
                   language: str = 'eng',
                   tts: str = 'tts-1',
                   max: int = 1000,
                   printall: bool = False,
                   exit_chat: str = 'stop'):
        gpt = self.talk_model

        print('Send "'+exit_chat+'" to exit.')

        who = self.assistant
        if who in assistants:
            system = assistants[who]
        elif who != '':
            self.add_persona(who, language)
        else:
            system = system
        while True:
            message = input('\n')
            if message == exit_chat:
                print('Chat Closed')
                break
            else:
                self.send_message(message,system=system, model=gpt, maxtoken=max, print_reply=printall, print_token=False, print_user=True, play=True, voice=voice, tts=tts)
                print('')


    def talk(self,
             voice='nova', language='eng', tts= 'tts-1', max=1000, printall=False, printinfo=True,  write=False):

        gpt = self.talk_model

        #record_audio(duration, "input.mp3")
        loop_audio(start='alt', stop='ctrl', filename='temp.wav', printinfo=printinfo)
        transcript = self.whisper("temp.wav", print_transcription=printall)

        who = self.assistant
        if who in assistants:
            system = assistants[who]
        else:
            self.add_persona(who, language)
            system = ''
        play = not write
        printall = printall if not write else True
        self.send_message(transcript,system=system, model=gpt, maxtoken=max,  print_reply=printall, print_token=False, play=play, voice=voice, tts=tts)

    def talk_loop(self,
                  voice='nova', language='eng', tts= 'tts-1', max=1000, printall=False, write=False, chat='alt' , exit='shift'):
        model = self.talk_model
        who = self.assistant
        print('Press '+chat+' to chat, '+exit+' to exit.')
        while True:
            if kb.is_pressed(chat):
                self.talk(who, gpt=model, voice=voice, language=language, tts= tts, max=max, printall=printall, write=write)
                print('Press '+chat+' to chat, '+exit+' to exit.')
            elif kb.is_pressed(exit):
                print('Chat Closed')
                break


    ####### Assistants #######
    def chat(self,
             m,
             gpt=None,
             max=1000, img='', paste=False, token=False, translate=False, create=False):

        gpt = gpt or self.model

        p = pc.paste() if paste else ''

        if self.bio:
            self.add_bio()#add = "and you are his assistant. ***")
            # "and you are his best friend. ***")

        self.send_message(m+p,
                          #system=sys,
                          maxtoken=max,
                          model=gpt,
                          img=img,
                          print_token=token,
                          create=create)

        if translate or self.translate:
            translator = create_translator(rileva_lingua(m))
            if self.translate_jap:
                translator = create_jap_translator(rileva_lingua(m))
            print('\n')
            self.ask(self.reply, translator)

    ct = chat

    def cp(self, *args, **kwargs):
        # Passes all positional and keyword arguments to the chat method, setting paste to True
        kwargs['paste'] = True  # Ensure paste is always set to True
        self.chat(*args, **kwargs)

    def ci(self, *args, **kwargs):
        kwargs['img'] = pc.paste()
        self.chat(*args, **kwargs)

    # def cp(self, m, max=1000, img='', clip=True, token=False, translate= False, create=False):
    #     self.chat(m=m, max=max, img=img, paste=True, clip=clip, token=token, translate=translate, create=create)

    def chat_loop(self,
                  system=None,
                  max=1000, language='eng', exit_chat= 'stop', printall=True):
        gpt = self.model
        who = self.assistant
        print('Send "'+exit_chat+'" to exit chat.')
        if who in assistants:
            system = assistants[who]
        elif who != '':
            self.add_persona(who, language)
        else:
            system = system
        while True:
            message = input('\n')
            if message == exit_chat:
                print('Chat Closed')
                break
            else:
                self.send_message(message,system=system, model=gpt, maxtoken=max, print_reply=printall, print_token=False, print_user=True)
                print('')


    # Formatting
    def schematize(self, m, language='english', max = 1000, img='', paste = False, clip=True):
        if language != 'english':
            self.expand_chat('Reply only using '+language, 'system')
        self.add_system(assistants['schematizer'])
        self.chat(m, max, img, paste, clip)

    def make_prompt(self, m, max = 1000, img='', clip=True, sdxl=True):
        import stablediff_rag as sd
        if sdxl:
            assistant = sd.rag_sdxl
        else:
            assistant = sd.rag_sd
        self.add_system(assistant)
        self.chat(m, max, img, clip)

    # Translators
    def translate(self, language='English'):
        self.ask(self.reply, create_translator(language))


    # def japanese_learner(self, m,voice='nova', times= 3, speed=1):
    #     self.japanese_teacher(m, 'gpt-4o')
    #     print('')
    #     phrase = self.reply.split('\n')[0].split(':')[1].strip()
    #     text2speech(phrase,voice=voice, speed = speed, play=True)
    #     audio_loop()
    #
    # def portuguese_learner(self, m,voice='nova', times= 3, speed=1):
    #     self.portuguese_teacher(m, 'gpt-4o')
    #     print('')
    #     phrase = self.reply.split('\n')[0].split(':')[1].strip()
    #     text2speech(phrase,voice=voice, speed = speed, play=True)
    #     audio_loop()


# An embedded assistant or a character of your choice
copilot_gpt = 'gpt-4o-2024-08-06'
chatgpt = GPT(assistant='base')
creator = GPT(assistant='creator')
fixer = GPT(assistant='fixer', model=copilot_gpt)
delamain = GPT(assistant='delamain', model=copilot_gpt)
oracle = GPT(assistant='oracle', model=copilot_gpt)
R = GPT(assistant='roger', model=copilot_gpt)
Rt = GPT(assistant='robert', model=copilot_gpt)
C = GPT(assistant='delamain', format='python', model=copilot_gpt)


# Scientific Assistants
leonardo = GPT(assistant='leonardo')
newton = GPT(assistant='leonardo', format='python')
galileo = GPT(assistant='leonardo', format='markdown')
mendel = GPT(assistant='mendel')
watson = GPT(assistant='mendel', format='latex')
venter = GPT(assistant='mendel', format='python')
crick = GPT(assistant='mendel', format='markdown')
darwin = GPT(assistant='darwin')
dawkins = GPT(assistant='darwin', format='markdown')
penrose = GPT(assistant='penrose')
turing = GPT(assistant='penrose', format='python')
marker = GPT(assistant='penrose', format='markdown')
collins = GPT(assistant='collins')
elsevier = GPT(assistant='collins', format='latex')
springer = GPT(assistant='collins', format='markdown')


# Characters
julia = GPT(assistant='julia', bio=True)
mike = GPT(assistant='mike', bio=True)
michael = GPT(assistant='michael', translate=True, bio=True)
miguel = GPT(assistant='miguel', translate=True, bio=True)
francois = GPT(assistant='francois', translate=True, bio=True)
luca = GPT(assistant='luca', translate=True, bio=True)
hero = GPT(assistant='hero', translate=True, translate_jap=True, bio=True)
yoko = GPT(assistant='yoko', translate=True, translate_jap=True, bio=True)

# Languages
japanese_teacher = GPT(assistant='japanese_teacher')
portuguese_teacher = GPT(assistant='portuguese_teacher')


#%%


######### INFO #########
# https://platform.openai.com/account/rate-limits
# https://platform.openai.com/account/usage
# https://platform.openai.com/docs/guides/text-generation/chat-completions-api
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb


#%%