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
def make_sentiments_df(sentiments):
    return pd.DataFrame({'sentiment': sentiments.keys(), 'representation': sentiments.values()})

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


def write_new_sentiment(new_sentiment= 'friendship', hint='', clearchat=True):
    if clearchat:
        op.clearchat()
    else:
        pass
    op.chatgpt("""
    
    """+str(sentiments)+"""
    
    Please, following this dictionary example, write down this new entry:
    '"""+new_sentiment+"""': ['"""+new_sentiment.replace('_',' ')+"""','"""+hint+"""','','','','','','','','','','','','',
    
    Reply example:
    'anger': ['Anger', 'Rage', 'Fury', 'Hostility', 'Irritation', 'Frustration', 'Resentment', 'Outrage', 'Hatred', 'Wrath', 'Aggression', 'Vexation', 'Annoyance', 'Displeasure', 'Retribution']
    
    """, 'gpt-4-turbo',2000)
    new_entry = ast.literal_eval('{'+op.reply+'}')
    return new_entry


new_sentiment = {
    '': []}
def add_sentiment(new_sentiment_entry, replace=False):
    global sentiments
    if new_sentiment_entry not in list(sentiments.keys()) or replace:
        sentiments.update(new_sentiment_entry)

def write_add_sentiment(new_sentiment= 'friendship', hint='', replace=False, clearchat=True):
    global sentiments_df
    new_sentiment = write_new_sentiment(new_sentiment= new_sentiment, hint=hint, clearchat=clearchat)
    add_sentiment(new_sentiment, replace=replace)
    sentiments_df = make_sentiments_df(sentiments)
    display(sentiments_df)

def add_sentence(new_sentence):
    global sentences
    if new_sentence not in sentences:
        sentences.append(new_sentence)

print('Default Sentiments:')
display(sentiments_df)
#%%

# USAGE
#new_sentiment=write_new_sentiment()
#%%
#write_add_sentiment(hint='Pals',replace=True)
#%%

#add_sentence("")
#%%
#sentiment_analysis(sentences, sentiments)
#%%

#%%

couples=  [['The president greets the press in Chicago', 'Obama speaks to the media in Illinois'],
           ['The president greets the press in New York', 'Obama speaks to the media in Illinois'],
           ['The president greets the press in Chicago', 'Dua Lipa speaks to the media in Illinois'],
           ['The president greets the jews in Chicago', 'Obama speaks against racism'],
           ['The president greets the US army in Chicago', 'Obama speaks against Russia'],
           ['The president greets the US army in Chicago', 'Obama speaks against war'],
           ['My girlfriend eats an ice cream in Chicago', 'Obama speaks against Russia'],
           ['Honey, count sheep, my little boy, and sleep well, my dear. See you tomorrow.', "The Lord Satan unleashed a horde of demons from the gates of hell to slaughter the angels of heaven."],
           ['Honey, count sheep, my little boy, and sleep well, my dear. See you tomorrow.', "The Lord Satan unleashed a horde of demons from the gates of hell to put the world on fire."],
           ['Honey, count sheep, my little boy, and sleep well, my dear. See you tomorrow.', "Conditional generative adversarial network driven radiomic prediction of mutation status based on magnetic resonance imaging of breast cancer"],
           ["My cat Fuffy ate his pet food and spitted a fur ball",'Obama, Clinton, Kennedy all Presidents of US']
           ]
