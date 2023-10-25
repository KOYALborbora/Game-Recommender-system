from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from scipy.constants import pt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

highly_rated_games = pickle.load(open('highly_rated_games.pkl','rb'))
games = pickle.load(open('games.pkl','rb'))
cosine_sim= pickle.load(open('cosine_sim.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',game_name = list(highly_rated_games['name'].values),image = list(highly_rated_games['header_image'].values))

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_games',methods=['post'])
def get_game_recommendations(num_recommendations=10):
    game_title = request.form.get("game-title")
    # Get the index of the game in the dataset
    game_index = games.index[games['name'] == game_title].tolist()[0]

    # Get the cosine similarity scores for the game
    similar_scores = list(enumerate(cosine_sim[game_index]))

    # Sort the games by similarity score in descending order
    similar_games = sorted(similar_scores, key=lambda x: x[1], reverse=True)

    # Get the top N similar games
    top_similar_games = similar_games[1:num_recommendations + 1]

    # Extract the game titles
    recommended_games =recommended_games = list([games.iloc[i[0]]['name'],games.iloc[i[0]]['header_image']] for i in top_similar_games)

    return render_template("recommend.html", data=recommended_games)

if __name__ =='__main__':
    app.run(debug=True)