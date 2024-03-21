# packages
import ast

import numpy as np

import pychatgpt as op
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from datetime import datetime


from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns
import matplotlib.pyplot as plt

#####################################
import nltk
def update_nlkt():
    nltk.download('stopwords')
    nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import time

def nltk_preprocessing(text, lower=True, trim=True, stem=True, language='enghlish'):
    update_nlkt()
    #docs_processed = [nltk_preprocessing(doc) for doc in docs_to_process]
    timea = time.time()
    stop_words = set(stopwords.words(language))
    stemmer = PorterStemmer()
    word_tokens = word_tokenize(text)
    word_tokens = [word.lower() for word in word_tokens] if lower else word_tokens
    word_tokens = [word for word in word_tokens if word not in stop_words] if trim else word_tokens
    word_tokens = [stemmer.stem(word) for word in word_tokens] if stem else word_tokens
    print(time.time()-timea)
    print(word_tokens)
    return " ".join(word_tokens)

#####################################


text_models = ["all-mpnet-base-v2",
               # designed as general purpose model, The all-mpnet-base-v2 model provides the best quality,
               "all-MiniLM-L6-v2",  # while all-MiniLM-L6-v2 is 5 times faster and still offers good quality.
               'albert-base-v2',
               'bert-base-cased',
               "allenai-specter", # SPECTER is a model trained on scientific citations and can be used to estimate the similarity of two publications. We can use it to find similar papers.
               ]
# https://www.sbert.net/docs/pretrained_models.html

# Initialize the SentenceTransformer model
sentence_transformer = SentenceTransformer(text_models[0])

# NEW MODELS GOES IN TORCH CACHE
# C:\Users\Utente/.cache\torch\sentence_transformers\albert-base-v2


def extract_embedding(input_text):
    # Encode the input text to get the embedding
    embedding = sentence_transformer.encode(input_text)
    return embedding


def plot_similarity_heatmap(results_df, save_plot=True, colormap='YlOrRd'):
    heatmap_data = results_df[results_df.columns[1:]]
    # Truncate sentence to max 25 characters for labels
    truncated_sentences = results_df.iloc[:, 0].str.slice(0, 25)
    # Plotting the heatmap with truncated labels for y-axis
    sns.heatmap(heatmap_data, yticklabels=truncated_sentences, cmap=colormap)
    plt.tight_layout()
    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        plt.savefig(timestamp+'_heat_map.png')
    plt.show()


############
# BUTCH PIPELINE
def get_embedding_list(sentences, model="text-embedding-3-large", preprocessing=False, openai=True):
    if preprocessing:
        for sentence in sentences:
            sentences_temp = []
            sentences_temp.append(nltk_preprocessing(sentence))
        sentences = sentences_temp
    embeddings = []
    for sentence in sentences:
        if openai:
            embeddings.append(op.get_embeddings(sentence, model=model))
        else:
            embeddings.append(sentence_transformer.encode(sentence))
    return embeddings


def get_cosine_distance_list(embeddings, ref_embedding, use_scipy=True, print_all=False, decimal=3):
    similarity_list=[]
    for embedding in embeddings:
        if use_scipy:
            cosine = distance.cosine(embedding, ref_embedding)
            similarity = round((1-cosine)*100,decimal)
        else:
            cosine = cosine_similarity(np.array(embedding).reshape(1, -1),
                                       np.array(ref_embedding).reshape(1, -1))
            similarity = round((cosine[0][0])*100,decimal)
        similarity_list.append(similarity)
        if print_all:
            print('Similarity of two sentences are equal to',similarity,'%')
            print(similarity)
    return similarity_list


def sentiment_analysis(sentences, sentiments, openai=True, preprocessing=False):
    embeddings = get_embedding_list(sentences, openai=openai, preprocessing=preprocessing)
    sentiments_processed = []
    for sentiment_key in sentiments.keys():
        sentiment_list = sentiments[sentiment_key]
        sentiment = ' '.join(sentiment_list)
        sentiments_processed.append(sentiment)
    ref_embeddings = get_embedding_list(sentiments_processed, openai=openai, preprocessing=preprocessing)

    df = pd.DataFrame(columns=['sentence'], data=sentences)

    for t in range(len(ref_embeddings)):
        similarity_list = get_cosine_distance_list(embeddings, ref_embeddings[t])
        #print(similarity_list)

        for i in range(len(similarity_list)):
            df.at[i, list(sentiments.keys())[t]] = similarity_list[i]

    plot_similarity_heatmap(df, save_plot=True)
    return df

sentiments = {
    'happiness': ['Joy', 'Happiness', 'Bliss', 'Euphoria', 'Delight', 'Contentment', 'Glee', 'Ecstasy', 'Jubilation', 'Cheerfulness', 'Laughter', 'Smiling', 'Radiance', 'Serenity', 'Exhilaration'],
    'sadness': ['Sadness', 'Grief', 'Sorrow', 'Melancholy', 'Depression', 'Despair', 'Heartache', 'Misery', 'Disappointment', 'Loneliness', 'Regret', 'Desolation', 'Despondency', 'Pain', 'Unhappiness'],
    'fear': ['Fear', 'Anxiety', 'Panic', 'Terror', 'Dread', 'Worry', 'Phobia', 'Fright', 'Nervousness', 'Insecurity', 'Apprehension', 'Scare', 'Shock', 'Horror', 'Unease'],
    'anger': ['Anger', 'Rage', 'Fury', 'Hostility', 'Irritation', 'Frustration', 'Resentment', 'Outrage', 'Hatred', 'Wrath', 'Aggression', 'Vexation', 'Annoyance', 'Displeasure', 'Retribution'],
    'disgust': ['Disgust', 'Revulsion', 'Contempt', 'Abhorrence', 'Loathing', 'Repugnance', 'Distaste', 'Antipathy', 'Aversion'],
    'love': ['Love', 'Affection', 'Fondness', 'Adoration', 'Devotion', 'Attachment', 'Passion', 'Amour', 'Infatuation', 'Lust'],
    'hate': ['Hate', 'Loathing', 'Detestation', 'Abhorrence', 'Animosity', 'Antipathy', 'Hostility', 'Enmity', 'Resentment', 'Revulsion'],
    'sexuality': ['Attraction', 'Desire', 'Lust', 'Passion', 'Intimacy', 'Libido', 'Sensuality', 'Eroticism', 'Arousal', 'Love'],
    'violence': ['Violence', 'Aggression', 'Brutality', 'Force', 'Ferocity', 'Rage', 'Assault', 'Attack', 'Coercion', 'Domination'],
    'science': ['Inquiry', 'Exploration', 'Experimentation', 'Observation', 'Analysis', 'Study', 'Research', 'Technology', 'Innovation', 'Discovery'],
    'bizarre': ['Bizarre', 'Odd', 'Uncanny', 'Weird', 'Strange', 'Eerie', 'Surreal', 'Quirky', 'Unusual', 'Grotesque']
}

seven_deadly_sins = {'pride': ['Pride', 'Satisfaction', 'Dignity', 'Ego', 'Confidence', 'Self-respect', 'Vanity', 'Arrogance', 'Honor', 'Glory', 'Triumph', 'Joy', 'Fulfillment', 'Esteem', 'Self-worth'], 'greed': ['Greed', 'Avarice', 'Covetousness', 'Acquisitiveness', 'Rapacity', 'Materialism', 'Insatiability', 'Selfishness', 'Graspingness', 'Gluttony', 'Desire', 'Envy', 'Lust', 'Obsession', 'Hoarding'], 'lust': ['Lust', 'Desire', 'Longing', 'Craving', 'Passion', 'Ardor', 'Sensuality', 'Libido', 'Yearning', 'Infatuation', 'Attraction', 'Carnality', 'Hunger', 'Thirst', 'Fervor'], 'envy': ['Envy', 'Jealousy', 'Covetousness', 'Resentment', 'Begrudging', 'Lust', 'Longing', 'Grudge', 'Spite', 'Emulation', 'Rivalry', 'Invidiousness', 'Malice', 'Bitterness', 'Competitiveness'], 'gluttony': ['Gluttony', 'Overeating', 'Excess', 'Greed', 'Indulgence', 'Bingeing', 'Overindulgence', 'Insatiability', 'Hunger', 'Appetite', 'Craving', 'Gorging', 'Self-indulgence', 'Compulsion', 'Intemperance'], 'wrath': ['Wrath', 'Anger', 'Rage', 'Fury', 'Ire', 'Temper', 'Indignation', 'Annoyance', 'Resentment', 'Vexation', 'Outrage', 'Spite', 'Hatred', 'Hostility', 'Retaliation'], 'sloth': ['Sloth', 'Laziness', 'Idleness', 'Indolence', 'Inaction', 'Lethargy', 'Inactivity', 'Sluggishness', 'Torpor', 'Apathy', 'Neglect', 'Procrastination', 'Disinterest', 'Inertia', 'Listlessness']}

def make_sentiments_df(sentiments):
    return pd.DataFrame({'sentiment': sentiments.keys(), 'representation': sentiments.values()})
def clear_sentiments():
    global sentiments
    sentiments = {}

sentiments_df = make_sentiments_df(sentiments)

sentences = ['''Tomorrow is my birthday! It's time to party.''',
             '''Yesterday my cat died... I'm so sad...''',
             '''I won the lottery! It's my lucky day!''',
             '''my girlfriend broke up with me. I'm so angry!''',
             '''I broke up with my girlfriend, I'm a worthless man''',
             '''President Obama spoke for the rights of black people in Chicago''',
             '''Autism is a neurodevelopmental condition of variable severity with lifelong effects that can be recognized from early childhood, chiefly characterized by difficulties with social interaction and communication and by restricted or repetitive patterns of thought and behaviour.''',
             '''Atherosclerosis is a disease of the arteries characterized by the deposition of fatty material on their inner walls.''',
             ''' Hitler's army is invading Poland; the World War begins''',
             ''' Today I'll buy roses for my beloved girlfriend''',
             ''' Today I'll buy condoms to have fun with my hot girlfriend''']

###############
def generate_sentiment(new_sentiment= 'friendship', hint='', clearchat=True, max=1000):
    if clearchat:
        op.clearchat()

    op.chatgpt("""
    
    """+str(sentiments)+"""
    
    Please, following this dictionary example above, write down this new entry:
    '"""+new_sentiment+"""': ['"""+new_sentiment.replace('_',' ')+"""','"""+hint+"""','','','','','','','','','','','','',
    
    Reply example:
    'anger': ['Anger', 'Rage', 'Fury', 'Hostility', 'Irritation', 'Frustration', 'Resentment', 'Outrage', 'Hatred', 'Wrath', 'Aggression', 'Vexation', 'Annoyance', 'Displeasure', 'Retribution']
    
    """, 'gpt-4-turbo',max)
    new_entry = ast.literal_eval('{'+op.reply+'}')
    return new_entry


#new_sentiment = {    '': []}
def add_sentiment(new_sentiment_entry, replace=False):
    global sentiments
    if new_sentiment_entry not in list(sentiments.keys()) or replace:
        sentiments.update(new_sentiment_entry)

def generate_update_sentiment(new_sentiment= 'friendship', hint='', replace=False, clearchat=True):
    global sentiments_df
    new_sentiment = generate_sentiment(new_sentiment= new_sentiment, hint=hint, clearchat=clearchat)
    add_sentiment(new_sentiment, replace=replace)
    sentiments_df = make_sentiments_df(sentiments)
    display(sentiments_df)

def generate_update_sentiments(new_sentiments=['pride', 'greed', 'lust', 'envy', 'gluttony', 'wrath', 'sloth'], clearchat=False):
    global sentiments_df
    for entry in new_sentiments:
        generate_update_sentiment(entry, clearchat=clearchat)

def generate_replace_sentiments(new_sentiments=['pride', 'greed', 'lust', 'envy', 'gluttony', 'wrath', 'sloth'], clearchat=False):
    global sentiments_df
    clear_sentiments()
    for entry in new_sentiments:
        generate_update_sentiment(entry, clearchat=clearchat)

###############

def generate_new_sentences(topic= 'friendship', hint='', count= 5, clearchat=True, max = 1000):
    if clearchat:
        op.clearchat()

    op.chatgpt("""
    
    """+str(sentences)+"""
    
    Please, following this sentence list example above, write down """+str(count)+""" new sentences on """+topic+""" topic as a list. """+hint+"""
    
    
    Reply example for 'friendship' topic:
    ["My best friend and I have been inseparable since kindergarten, truly partners in crime.",
    "True friendship is like sound health; the value of it is seldom known until it is lost.",
    "Spending time with good friends after a long week feels like healing to the soul.",
    "Friendships require effort and care, but the joy and support they bring are worth every bit.",
    "Finding a friend who understands your tears is much more valuable than finding a bunch of friends who only know your smile."
    ]
    
    """, 'gpt-4-turbo',max)
    new_sentences = op.reply.replace('```python\n', '')
    new_sentences = new_sentences.replace('\n```', '')
    new_sentences = ast.literal_eval(new_sentences)
    return new_sentences

def append_sentences(new_sentences):
    global sentences
    if not isinstance(new_sentences, list):
        new_sentences = [new_sentences]
    for sentence in new_sentences:
        if sentence not in sentences:
            sentences.append(sentence)

def generate_append_sentences(topic= 'friendship', hint='', count= 5, clearchat=True, max = 1000):
    new_sentences=generate_new_sentences(topic, hint, count, clearchat, max)
    append_sentences(new_sentences)

def replace_sentences(new_sentences):
    global sentences
    sentences = new_sentences

print('Default Sentiments:')
display(sentiments_df)
#%%
# USAGE
#generate_update_sentiments(['genetics', 'diseases', 'politics'])
#seven_deadly_sins=generate_sentiment('seven_deadly_sins','Pride')
#%%
#add_sentence(""" """)
#%%
#sentiment_analysis(sentences, sentiments)
#%%
#sentiment_analysis(sentences, seven_deadly_sins)
#%%
#new_sentences = generate_append_sentences('Obama President')
#%%
#new_sentences = generate_append_sentences()
#%%
#sentiment_analysis(sentences, sentiments)
#%%




#%%

