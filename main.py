import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import pickle
import nltk
from nltk.stem.porter import PorterStemmer

# Load the datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets on the 'title' column
movies = movies.merge(credits, on='title')

# Helper functions for data preprocessing
def convert(text):
    try:
        L = []
        for i in ast.literal_eval(text):
            L.append(i['name'])
        return L
    except:
        return []

def fetch_director(text):
    try:
        L = []
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                L.append(i['name'])
        return L
    except:
        return []

# Apply helper functions to preprocess the data
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['cast'] = movies['cast'].apply(lambda x: x[0:3])

# Combine features into a single 'tags' column
def collapse(L):
    return [i.replace(" ", "") for i in L]

movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)

# Ensure that the 'overview' field is a string and fill NaNs with empty strings
movies['overview'] = movies['overview'].fillna('')

# Combine the columns into a single 'tags' column
movies['tags'] = movies['overview'] + " " + movies['genres'].apply(lambda x: " ".join(x)) + " " + movies['keywords'].apply(lambda x: " ".join(x)) + " " + movies['cast'].apply(lambda x: " ".join(x)) + " " + movies['crew'].apply(lambda x: " ".join(x))

ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# Apply stemming to the 'tags' column
movies['tags'] = movies['tags'].apply(stem)

# Create a new dataframe with selected columns
new = movies[['movie_id', 'title', 'tags']].copy()

# Vectorize the 'tags' column
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new['tags']).toarray()

# Compute cosine similarity matrix
similarity = cosine_similarity(vectors)

# Save the movie list and similarity matrix
pickle.dump(new['title'].values, open('movie_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

