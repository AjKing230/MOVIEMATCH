from flask import Flask, request, render_template
import pickle
import numpy as np
import urllib.parse

app = Flask(__name__)

# Load the movie list and similarity matrix
with open('model/movie_list.pkl', 'rb') as file:
    movies = pickle.load(file)

with open('model/similarity.pkl', 'rb') as file:
    similarity = pickle.load(file)

def generate_google_search_url(movie_title):
    base_url = 'https://www.google.com/search'
    query_params = {'q': f'{movie_title} watch online'}
    url_encoded_params = urllib.parse.urlencode(query_params)
    return f'{base_url}?{url_encoded_params}'

@app.route('/')
def index():
    return render_template('index.html', movies=movies)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie = request.form['movie']
    if movie:
        idx = np.where(movies == movie)[0][0]
        sim_scores = list(enumerate(similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_5 = [{
            'title': movies[i[0]],
            'google_search_url': generate_google_search_url(movies[i[0]])
        } for i in sim_scores[1:6]]
        return render_template('recommendations.html', recommendations=top_5, movie=movie)
    else:
        return render_template('index.html', movies=movies)

if __name__ == '__main__':
    app.run(debug=True)

