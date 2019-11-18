import json
import bz2
import os
import ast
import matplotlib.pyplot as plt
import textblob
from textblob import TextBlob
import lzma
import nltk
import numpy as np
import pandas as pd
from textblob import TextBlob
from textblob import Blobber
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re
import operator
import string

subreddit_list = ['politics','uspolitics','americanpolitics','progressive','democrats','Liberal','Republican',
                    'Conservative','Libertarian']

subreddit_members = {'politics':5.5E6,'uspolitics':1.65E4,'americanpolitics':1.12E4,'progressive':6.17E4,'democrats':1.11E5,
                        'Liberal':7.78E4,'Republican':9.21E4,'Conservative':268E5,'Libertarian':345E5}
                        
# get more conservative and liberal words???
liberal_words = ["progressive","Biden","universal basic income","AOC", "Ocasio-Cortez", "liberal", "democrat", "Obama", "Clinton", "Sanders", 
"green new deal", "leftist", "Yang", "Warren", "Kamala", "medicare for all"]

conservative_words = ["Cheney","Shapiro","Koch","Paul Ryan","Rand Paul","Bush","Palin","Mattis","McCain","Romney", "Trump",
"Cruz", "republican", "Kushner", "conservative", "GOP", "Pence"]

# output is of the form {"subreddit_1_name":[('post_1_title',post_1_score),(''post_2_title',post_2_score),...], "subreddit_2_name"...}
output = {}
scores = {}
aggregated_titles = {}
bigram_count = {}

def add_data(line):
    '''
    Parses through all the lines in a file (where each line in the file is a dict),
    creates a tuple containing the title and score of the post, and adds it to the
    list of tuples in its specified spot in the output dict.
    '''
    post = json.loads(line)
    sub = post.get("subreddit")
    if sub in subreddit_list:
        if post.get("score") > 10: # arbitrary choice, should think about this more and change the threshold to be specific to each sub.
            normalized_score = (post.get("score") * 1.0) / subreddit_members.get(sub)
            if sub in output:
                output[sub].append((post.get("title"), normalized_score))
            else:
                output[sub] = [(post.get("title"), normalized_score)]

def open_files():
    '''
    TODO: figure out how to handle zst files
    Goes through the directory containing all the data files.
    '''
    path = os.path.expanduser('/data/files.pushshift.io/reddit/submissions')
    os.chdir('/data/files.pushshift.io/reddit/submissions')
    files = [f for f in os.listdir(path)]
    for i in files:
        print('opening file')
        if i.endswith('.bz2'):
            with bz2.open(i, "r") as content: 
                 for line in content:
                    add_data(line)
        elif i.endswith('.xz'):
            with lzma.open(i, 'rt') as content:
                for line in content:
                    add_data(line)
        # elif i.endswith('.zst'): # need to figure out how to open these.
        #     with as content:
        #         add_data()

def aggregate_titles(subreddit):
    '''
    Aggregate all the post titles for each subreddit
    '''
    aggregated_titles[subreddit] = " ".join(j[0] for j in output[subreddit])
        
def create_metric(subreddit):
    '''
    TODO: count number of conservative and liberal words. if num conservative > num liberal, multiply by -1.
    Creates a bar graph with each subreddit and their aggregated political bias score
    '''
    post_list = output[subreddit]
    scores[subreddit] = 0
    for j in post_list:
        num_cons_words = 0
        num_lib_words = 0
        title_category_factor = 1 # 1 if title is about a liberal topic, -1 if about conservative topic
        for word in j[0]:
            if word in conservative_words:
                cons_words+=1
            elif word in liberal_words:
                lib_words+=1
        if num_cons_words >= num_lib_words:
            title_category_factor = -1
        title = TextBlob(j[0])
        if title.sentiment.subjectivity > 0.0:
            # heavier weighting for subjective article titles
            # since the min subjectivity > 0 is 0.1, multiplying by 50 gives it at least 5x weight
            sntmnt = (title.sentiment.polarity * 50.0 * title.sentiment.subjectivity)
        else:
            sntmnt = title.sentiment.polarity
        scores[subreddit] += ((j[1] * 1.0) + sntmnt) * title_category_factor
    
def create_bigrams(subreddit):
    '''
    Creates a dict with the frequency of the bigrams for each subreddit
    '''
    bigram_count_mini = {} # holds bigram frequencies for each subreddit
    text = aggregated_titles[subreddit]
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = text.lower()
    words = text.split()
    bi_grams = list(zip(words, words[1:]))
    for gram in bi_grams:
        if gram not in bigram_count_mini:
            bigram_count_mini[gram] = 1
        else:
            bigram_count_mini[gram]+=1
    bigram_count[subreddit] = bigram_count_mini

def plot_bigrams():
    for subreddit in bigram_count:
        bigram_dict = bigram_count[subreddit]
        bigram_string = subreddit + '_top_10_bigrams_.png'
        top_10_bigrams_ = dict(sorted(bigram_dict.items(), key=operator.itemgetter(1), reverse=True)[:10])
        plt.bar(range(len(top_10_bigrams_)), list(top_10_bigrams_.values()), align='center')
        plt.xticks(range(len(top_10_bigrams_)), list(top_10_bigrams_.keys()))
        plt.savefig(bigram_string)

def plot_metric():
    plt.bar(range(len(scores)), list(scores.values()), align='center')
    plt.xticks(range(len(scores)), list(scores.keys()))
    plt.savefig('subreddit_scores.png')

def plot_wordclouds(subreddit):
    '''
    Creates a wordcloud for each subreddit
    '''
    agg_text = aggregated_titles[subreddit]
    stopwords= set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords,max_font_size=50, max_words=100, background_color="white").generate(agg_text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    wordcloud_string = subreddit + '_wordcloud.png'
    wordcloud.to_file(wordcloud_string)

def main():
    open_files()
    for subreddit in output:
        aggregate_titles(subreddit)
        create_metric(subreddit)
    for subreddit in aggregated_titles:
        plot_wordclouds(subreddit)
        create_bigrams(subreddit)
    plot_bigrams()
    plot_metric()
    
main()