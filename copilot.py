#%%
import os
import sys
import requests
#%%

def get_gitfile(url, dir=os.getcwd()):
    file = requests.get(url.replace('blob', 'raw')).content
    open(os.path.join(dir, url.rsplit('/', 1)[1]), 'wb').write(file)
    print('file downloaded')
if not os.path.exists('pychatgpt.py'):
    get_gitfile("https://raw.githubusercontent.com/johndef64/pychatgpt/main/pychatgpt.py")

#%%
import pychatgpt as op
op.display_assistants()

clip = False if 'google.colab' in sys.modules else True
#%% Load file as variable
path = os.getcwd()
file = op.load_file(path)

#%% expand chat
op.clearchat()
op.expand_chat(file, 'user') #system #assistant
#%%
m = ''' 
shutdoan windos cmd terminal pc dopo mezz ora epr favore, il comado

'''
op.delamain(m, op.model, 1000, clip)
#%%
op.clearchat()
m = ''' 

'''
op.leonardo(m, op.model, 1000, clip)
#%%
op.clearchat()
m = ''' 

'''
op.newton(m, op.model, 1000, clip)
#%%
op.clearchat()
m = ''' 

'''
op.crick(m, op.model, 1000, clip)
#%%
op.clearchat()
m = '''
Today I'm going to the sea. Tomorrow I will go to the sea. Yesterday I went to the sea. '''
op.japanese_teacher(m, op.model, 1000, clip)
#%%
op.clearchat()
m = ''' 

'''
op.roger(m, op.model, 1000, clip)
#%%

#%% Load chat

op.load_chat()
df = op.pd.DataFrame(op.chat_gpt)
print(df.head())
#%%
m='''
 
'''
op.chatgpt(m, op.model, 1000, clip)
#%%
op.chat_gpt

#%%
# Whisper

op.whisper('audio.mp3', translate = True, response_format = "text", print_transcriprion = True)
#%%
# Text to Speech

op.voices
# ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
#%%
for i in op.voices:
    op.text2speech('''One does not simply walk into Mordor''',i,play=True)
#%%
m='''

'''
op.text2speech(m,'alloy',play=True)
#%%

op.speech2speech(15,'onyx', play=True, translate=True)
#%%
m = '''@ 
What's up bro?!'''
op.bestie(m, 'gpt-4', 1000, clip)
op.text2speech(op.reply,'onyx', play=True)
#%%

# Chat with...
m = '''@ 
What's up bro?!'''
op.chat_with('bestie',m, voice='onyx')
#%%
m='''@ 
write an introduction to machine learning as if it were the first lecture in my course
'''
op.chat_with('leonardo',m,voice='onyx', printall=True)
#%%
#%%
m='''@ 
ciao tesoro, come sei bella oggi?'''
op.chat_with('julia',m,voice='nova', printall=True)
#%%
op.chat_thread
#%%
ly='''@ 
Please, write the lyrics of a song in your style.
'''
#m='@Hi, who are you?'
#%%
op.chat_with('Nergal (Behemoth Frontman)', ly, voice='onyx', printall=True)
#%%
op.chat_with('Dua Lipa', ly, voice='nova', printall=True)
#%%
import pychatgpt as op
# Talk with ...
op.clearchat()
op.talk_with('bestie','onyx', printall=True)
#%%
op.clearchat()
op.talk_with_loop('julia','nova',printall=True)
#%%
# Japanese Learner
import pychatgpt as op
m= '''@Hello, I am a biologist studying computer science.'''
op.japanese_learner(m, 1)
#%%
# Describe Image
op.clearchat()
url = "https://i.pinimg.com/564x/2c/b7/25/2cb725d68d10b3e1fa87fa6c48769072.jpg"
url = r"F:\horizon_ai_repo\midjourney\mj_creations\alex gris\EvanJonas_by_Conrad_Roset_posing_starry_night_background_celest_4af6c7d1-1669-481a-aad1-3ce0c1f21ed6.png"
op.send_image(url)

#%%
# Stable Diffusion Prompt Maker

m='''@
An alchemist master brewer in a brewery laboratory who taps beer and talks to the observer, 1920s, vintage style, red-haired with a monocle, dressed elegantly.
'''
ml='''
The image features an illustration of a female character with noticeably unique characteristics, particularly her red eyes and the makeup, consisting of blue lipstick. She has dark hair with blue highlights and appears to be styled in a gothic fashion, indicated by her black choker, rings, and dark-colored clothing. The background suggests an urban setting with buildings and a clear sky. This seems to be artwork from a comic book or graphic novel, known for their stylized and vibrant portrayals of characters.
'''
op.clearchat()
m='''
The image features an artistic illustration of a person with their eyes closed, which gives a peaceful or contemplative expression. The character has short, dark hair with a slight wave, fair skin, and is wearing a black top with what appears to be a necklace. The style of the artwork is modern and somewhat ethereal, with a use of cool tones predominately blue, and there are speckled effects across the image that resemble stars or particles, adding to the dreamy quality of the illustration. Overall, the image has an intimate and serene mood. By Conrad Rosett, by Alphone Mucha
'''
m='''
a swedish man in psychedlic mushroom world, fractal patterns, violet lights
'''
op.prompt_maker(m)
#%%
op.english(''' un alchimista mastro birraio in una birreria laboratorio che spila birra e parla all ossevatore, anni '20, vintage stile, rossi di capelli con il monocolo, vestito elegante ''')
#%%
op.chat_thread
#%%
###vb.net
{'Positive prompt': 'artistic illustration, person with eyes closed, peaceful expression, short dark wavy hair, fair skin, black top, necklace, modern, ethereal style, cool tones, blue predominant, speckled star/particle effects, intimate, serene mood, (artstation:1.2), (ethereal:1.1)',
 'Negative prompt': '(photographic, realistic, 3d, illustration style, cartoonish), eyes open'}
###

###vb.net
{'Positive prompt': 'ethereal modern art, person with closed eyes, contemplative, short dark wavy hair, fair skin, black top, necklace, cool blue tones, star/particle effects, dreamy, serene, Alphonse Mucha inspired, (artstation:1.2)',
 'Negative prompt': '(photographic, realistic, 3d, illustration style, cartoonish), eyes open'}
###

###vb.net
{'Positive prompt': 'serene artwork, closed-eyed person, peaceful expression, modern ethereal style, short dark wavy hair, fair skin, black top, necklace, cool blue tones, starry/particle effects, dreamy, intimate mood, Conrad Roset inspired, (ethereal:1.1)',
 'Negative prompt': '(photographic, realistic, 3d, illustration style, cartoonish), eyes open'}
###
#%%

prompts = [
    {'Positive prompt': '1920s vintage style portrait, an alchemist master brewer in brewery lab, taps beer, talks to observer, red-haired with monocle, elegant attire, industrial setting, cinematic shot, film grain texture:1.1, (vintage, classy:1.2)', 'Negative prompt': '(worst quality, low quality, illustration, 3d, 2d, cartoons), modern clothing, outdoors setting'},
    {'Positive prompt': '1920s vintage style, alchemist master brewer pours beer, conversing, red-haired monocled figure in elegant attire, brewery lab, detailed interior, natural lighting, (cinematic, vintage film:1.2)', 'Negative prompt': '(low resolution, blurry, unrealistic lighting, cartoonish style), casual clothing, contemporary setting'},
    {'Positive prompt': '1920s vintage-style setting, alchemist master brewer discussing while pouring beer, red-haired character with monocle, elegant outfit, laboratory ambiance, focused expression, warm-toned lighting, (realistic, detailed, high resolution:1.2)', 'Negative prompt': '(caricatured, minimalistic, abstract, unnatural colors), bright lighting, casual outfit, modern background'},

    {'Positive prompt': 'comic book style, illustration of a female character with red eyes, blue lipstick, gothic fashion, dark hair with blue highlights, black choker, urban background with buildings, clear sky, vibrant colors, (comic art, graphic novel:1.2)', 'Negative prompt': '(realistic, photographic, portrait, 3d render, painting), bright smile'},

    {'Positive prompt': 'vibrant graphic novel art, female character with red eyes and blue lipstick, gothic fashion style, dark hair with blue highlights, urban background, black choker, stylized rings, (comic book art:1.2), (graphic novel:1.1)', 'Negative prompt': '(realistic, photographic, portrait, 3d render, painting), open mouth'},

    {'Positive prompt': 'comic book illustration, female character with red eyes, unique blue lipstick, gothic style, dark hair with blue highlights, black choker, urban setting, stylized rings, (vibrant colors:1.2), (comic art, graphic novel:1.1)', 'Negative prompt': '(realistic, photographic, portrait, 3d render, painting), smiling widely'}
]


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