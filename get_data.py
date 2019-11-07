import json
import bz2
import os
import ast
import matplotlib.pyplot as pyplot
import textblob
import nltk
import numpy as np
import pandas as pd
# list of subreddits to be considered
subreddits = ['politics'] # adding more later after preliminary test
# blank json file that i need to make in the repo?

# for filename in os.listdir(directory), where directory is "/data/files.pushshift.io/reddit/submissions" to go through each file,
# but need to have cases for each file type since not all of them are bz2

f = os.chdir(~/data/files.pushshift.io/reddit/submissions)
for i in f:
    if i.endswith('.bz2'):
        with bz2.open(filename:/data/files.pushshift.io/reddit/submissions, "r") as content:
            # loop through each line (each line is a dict) and only save the posts that are in r/politics
            # add each of those dicts to a new file (bz2 file?) and add it to my home directory
            output = []
            for i, line in enumerate(content):
                post = json.loads(line)
                if post.get("subreddit") in subreddits: 
                    output.append(post)
            # add the whole dict to the new file

# move output to a new file and move that to my home directory?


liberal_words = ["liberal", "democrat, ""obama", "clinton", "sanders", "green new deal", "progressive", "leftist", ]
conservative_words = ["trump", "white house", "cruz", "republican", "kushner", "conservative"]  

upvotes = dict.get("upvotes")
title = blob(dict.get("title")) # or something like that
sntmnt = title.sentiment()
metric = (upvotes * 1.0) * sntmnt

# how to deal with titles that have both conservative and liberal words?