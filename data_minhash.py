# handle imports
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import csv
import random as rd
import datasketch

#  import 
from textblob import TextBlob

# function for sentiment using textblob
def get_sentiment(text):
    if isinstance(text, str):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    else:
        return None

# function to gen shingles
def gen_shin(text, k):
    return [text[i:i + k] for i in range(len(text) - k + 1)]

# setup
test      = False
# one       = ["china buys russia"]
# k         = 5
# nhashes   = 10

# ran       = 10 
# top_sim   = 0
# top_idx   = -1

# function min_hash
# input(s)
#  - main   : array with multiple strings
#  - query  : array with 1 string
#  - k      : int
#  - n      : int
#  - res    : int
# return(s)
#  - array (size of res) [index, approx. jaccard score] (from highest to lowest score)

# reuse of HW, W3S1 w/ mod
# compute actual jaccard similarities among all pairs of data samples

def min_hash(main, query, k, n, res):
    ran = res
    sim_list = []
    nhashes = n
    all_entries2 = [entry[0] for entry in main]
    one = query
    one_shin = gen_shin(" ".join(one), k)
    m_one = datasketch.MinHash(nhashes)
    for i in one_shin:
        m_one.update(i.encode('utf8'))
    if test:
        print("one_shin: " + str(one_shin))
        print("m2           : " + str(m_one))

    # setup minhas for all_entries2
    for idx, doc in enumerate(all_entries2):
        all_shin = gen_shin(doc, k)
        m_all = datasketch.MinHash(nhashes)
        for i in all_shin:
            m_all.update(i.encode('utf8'))

        # find approx jaccard sim
        sim = m_one.jaccard(m_all)

        # append to list
        sim_list.append((sim, idx, main[idx][1]))

    # sort list
    sim_list2 = sorted(sim_list, key=lambda x: -x[0])[:ran]

    # get sentiment of query
    sen = [get_sentiment(text) for text in query]

    sim_list2 = sorted(sim_list2, key=lambda x: abs(x[2] - sen[0]))

    return sim_list2