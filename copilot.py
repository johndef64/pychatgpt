#%%
import os
import sys

import requests
def get_gitfile(url, flag='', dir = os.getcwd()):
    url = url.replace('blob','raw')
    response = requests.get(url)
    file_name = flag + url.rsplit('/',1)[1]
    file_path = os.path.join(dir, file_name)
    with open(file_path, 'wb') as file:
        file.write(response.content)

'''if not os.getcwd().endswith('pychatgpt.py'):
    handle="https://raw.githubusercontent.com/johndef64/pychatgpt/main/"
    files = ["pychatgpt.py"]
    for file in files:
       url = handle+file
       get_gitfile(url)'''

if 'google.colab' in sys.modules:
    clip=False
else:
    clip=True

import pychatgpt as op

# Copilot Assistants:
op.display_assistants()
#op.choose_model()

#%% Load file as variable
path = os.getcwd()
file = op.load_file(path)

#%% expand chat
op.clearchat()
#op.expand_chat('''  ''', 'system')
op.expand_chat('''  ''', 'assistant')
#op.expand_chat('''  ''', 'user')

#%%
m=''' '''
op.chatgpt(m, 'gpt-4', 1000, clip)
#%%
op.clearchat()
m='''

'''
#op.crick(m, clip=False)
op.crick(m, 'gpt-4', 1000, clip)
#%%
op.clearchat()
m = ''' 

'''
op.newton(m, 'gpt-4', 1000, clip)
#%%
op.clearchat()
m = ''' 

'''
op.leonardo(m, 'gpt-4', 1000, clip)
#%%
op.clearchat()
m = '''
oggi vado al mare. domani andrò al mare. ieri sono andato al mare '''
op.japanese_teacher(m, 'gpt-4', 1000, clip)
#%%

m=''
op.roger(m, 'gpt-4', 1000, clip)
#%%

m=''' io bene, grazie. sono distratto e non riesco a lavorare'''
op.bestie(m, 'gpt-4', 1000, clip)
op.text2speech(op.reply,'onyx', play=True)
#%%

op.clearchat()
op.talk_with('bestie', 8, 'onyx')
#%% Load chat
op.load_chat()
df = op.pd.DataFrame(op.chat_gpt)
print(df.head())
#%%
m=''' '''
op.chatgpt(m, 'gpt-4', 1000, clip)
#%%
op.chat_gpt
