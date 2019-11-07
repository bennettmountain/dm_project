import json
import bz2
import os
import ast
# list of subreddits to be considered
subreddits = ['politics'] # adding more later after preliminary test
 # blank json file that i need to make in the repo?

# for filename in os.listdir(directory), where directory is "/data/files.pushshift.io/reddit/submissions" to go through each file,
# but need to have cases for each file type since not all of them are bz2

f = os.listdir(~/data/files.pushshift.io/reddit/submissions)
for i in f:
    if i.endswith('.bz2'):
        # with bz2.open().....

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