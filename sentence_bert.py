from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np


# function sent_bert
# input(s)
#  - main   : array with multiple strings
#  - query  : array with 1 string
#  - res    : int
# return(s)
#  - array (size of res) (from highest to lowest score)

# compute sentence bert score among the data samples
def sent_bert(main, query, res, file = None):
    # this model can be used for semantic searches
    model = SentenceTransformer('all-MiniLM-L6-v2')

    if file == None:
        all_entries2 = [entry[0] for entry in main]
        all_embed = model.encode(all_entries2)
    else:
        all_embed = np.load(file)
        all_entries2 = [entry[0] for entry in main]

    # encode new test sentence
    one_embed = model.encode(query)

    # use cosine
    cos_score = util.cos_sim(one_embed, all_embed)

    # fix from 2D to 1D and remove NaN items
    cos_score = np.nan_to_num(cos_score)
    cos_score = cos_score.flatten()

    # return search range
    search_idx = np.argsort(cos_score)[::-1][:res]

    ret_val = []

    for i, idx in enumerate(search_idx):
        ret_val.append([idx, all_entries2[idx], cos_score[idx] ])

    return ret_val