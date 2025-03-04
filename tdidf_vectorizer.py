# Import all required libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Vector search using cosine similarity
def vector_search(query, vectorizer, mtx, database, k=5):
    temp = cosine_similarity(mtx, vectorizer.transform([query])).flatten()
    new = temp.argsort()[-k:][::-1]

    results = []
    for i in new:
        date = database.iloc[i]['Date']
        headline = database.iloc[i]['News_Headlines']
        sentiment = database.iloc[i]['Sentiment']
        score = temp[i]
        results.append((i, date, headline, score, sentiment))

    #print(f'\n*** Results for: "{query}" ***')
    #for result in results:
    #    print(f'Similarity Score: {result[3]:.4f}| Date: {result[1]} | {result[2]} | Sentiment value: {result[4]:.1f} | Index: {result[0]}')

    return results