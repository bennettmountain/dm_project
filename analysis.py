import matplotlib.pyplot as pyplot
import textblob
import nltk
import numpy as np
import pandas as pd

liberal_words = ["liberal", "democrat, ""obama", "clinton", "sanders", 
"green new deal", "progressive", "leftist", ]
conservative_words = ["trump", "white house", "cruz", "republican", 
"kushner", "conservative"]  

upvotes = dict.get("upvotes")
title = blob(dict.get("title")) # or something like that
sntmnt = title.sentiment()
metric = (upvotes * 1.0) * sntmnt

# how to deal with titles that have both conservative and liberal words?