# handle imports
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# reuse of HW, W3S1 w/ mod

## Jaccard similarity between lists <a> and <b>
def jaccard(a,b):
    sa = set(a)
    sb = set(b)

    # adding in this part to avoid the 0 errors
    if len(sa.union(sb)) == 0:
        return 0

    return len(sa.intersection(sb))/len(sa.union(sb))

# Utility function to print the shape of an array of arrays
def printshape(lx):
    print('shape: '+str(len(lx))+' x '+str(len(lx[0])))

#  import 
from textblob import TextBlob

# function for sentiment using textblob
def get_sentiment(text):
    if isinstance(text, str):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    else:
        return None

# function jac sim
# input(s)
#  - main   : array with multiple strings
#  - query  : array with 1 string
#  - k      : int
#  - res    : int
# return(s)
#  - array (size of res) [index, jaccard score] (from highest to lowest score)

# reuse of HW, W3S1 w/ mod
# compute actual jaccard similarities among all pairs of data samples

def jac_sim(main, query, k, res):
    out_file = 'jac_sim.csv'
    sim   = []

    # create shingles
    main2 = [entry[0] for entry in main]
    shingles  = [[item[n:n+k] for n in range(len(item)-k+1)] for item in main2]
    shingles2 = [[item[n:n+k] for n in range(len(item)-k+1)] for item in query]

    # get sentiment of query
    sen = [get_sentiment(text) for text in query]

    #print(sen)

    # create dataframe to output 
    #df = pd.DataFrame(columns=['items', 'jaccard'])
    #df.to_csv(out_file, mode='w', header=True, index=False)

    # loop through shingles and compare
    for k in range(len(shingles)):
        for m in range(len(shingles2)):
            jsim = jaccard(shingles[k], shingles2[m])
            # append index and score
            sim.append([k, jsim, main[k][1]])
            # also save index and score to output file
            #with open(out_file, mode='a', newline='') as file:
            #    df = pd.DataFrame([{'items': k, 'jaccard': jsim}])
            #    df.to_csv(file, header=False, index=False)
    
    # sort by jaccard scores (highest ascending)
    sim = sorted(sim, key=lambda x: x[1], reverse=True)[0:res]

    # sort by closest sentiment score
    sim = sorted(sim, key=lambda x: abs(x[1] - sen[0]))

    # return back sorted list but up to resolution desired size
    #return sim[0:res]
    return sim