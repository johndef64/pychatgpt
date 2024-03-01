#%%
import os
import sys
import requests

def get_gitfile(url, dir=os.getcwd()):
    file = requests.get(url.replace('blob', 'raw')).content
    open(os.path.join(dir, url.rsplit('/', 1)[1]), 'wb').write(file)
    print('file downloaded')
if not os.path.exists('pychatgpt.py'):
    get_gitfile("https://raw.githubusercontent.com/johndef64/pychatgpt/main/pychatgpt.py")

clip = False if 'google.colab' in sys.modules else True

import pychatgpt as op

# Copilot Assistants:
op.display_assistants()
#op.choose_model()

#%% Load file as variable
path = os.getcwd()
file = op.load_file(path)

#%% expand chat
op.clearchat()
op.expand_chat(file, 'user') #system #assistant
#%%
m = ''' 

'''
op.delamain(m, 'gpt-4', 1000, clip)
#%%
op.clearchat()
m = ''' 

'''
op.leonardo(m, 'gpt-4', 1000, clip)
#%%
op.clearchat()
m = ''' 

'''
op.newton(m, 'gpt-4', 1000, clip)
#%%
op.clearchat()
m = ''' 

'''
op.crick(m, 'gpt-4', 1000, clip)
#%%
op.clearchat()
m = '''
Today I'm going to the sea. Tomorrow I will go to the sea. Yesterday I went to the sea. '''
op.japanese_teacher(m, 'gpt-4', 1000, clip)
#%%
op.clearchat()
m = ''' 

'''
op.roger(m, 'gpt-4', 1000, clip)
#%%
op.clearchat()
m = ''' 

'''
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
m='''
 
'''
op.chatgpt(m, 'gpt-4', 1000, clip)
#%%
op.chat_gpt
#%%
# Notes
'''
Translation: 今日、私は海に行きます。 明日、海に行くつもりです。昨日、海に行きました。

Hiragana: きょう、わたしは うみ に いきます。 あした、うみに いくつもりです。きのう、うみに いきました。

Romaji: Kyou, watashi wa umi ni ikimasu. Ashita, umi ni iku tsumori desu. Kinou, umi ni ikimashita.

Analysis:
   Sentence1:
       Noun: 今日 (きょう) - kyou - today
       Pronoun: 私 (わたし) - watashi - I
       Particle: は (wa) - topic marking particle, often linking to the subject of the sentence.
       Noun: 海 (うみ) - umi - sea
       Particle: に (ni) - direction particle indicates the direction or goal of the action. 
       Verb: 行きます (いきます) - ikimasu - go

   Sentence2:
       Noun: 明日 (あした) - ashita - tomorrow
       Noun: 海 (うみ) - umi - sea
       Verb: 行くつもりです (いくつもりです) - iku tsumori desu - plan to go 

   Sentence3:
       Noun: 昨日 (きのう) - kinou - yesterday
       Noun: 海 (うみ) - umi - sea
       Verb: 行きました (いきました) - ikimashita - went
'''