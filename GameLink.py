import streamlit as st
import pickle
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

posters_df = pd.read_csv('steam_media_data.csv')

file_path = 'tfidf_matrix.pkl'
if os.path.exists(file_path):
    print(f"The file '{file_path}' exists.")
else:
    print(f"The file '{file_path}' does not exist.")

games_dict = pickle.load(open('games_dict.pkl', 'rb'))
games = pd.DataFrame(games_dict)

# Combining relevant columns into a single column for TF-IDF vectorization
games['combined_features'] = games['categories'] + ' ' + games['genres'] + ' ' + games['developer']

# Creating a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the TF-IDF vectorizer on the combined features
tfidf_matrix = tfidf_vectorizer.fit_transform(games['combined_features'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_game_recommendations(game_title, num_recommendations=10):
    # Get the index of the game in the dataset
    game_index = games.index[games['name'] == game_title].tolist()[0]

    # Get the cosine similarity scores for the game
    similar_scores = list(enumerate(cosine_sim[game_index]))

    # Sort the games by similarity score in descending order
    similar_games = sorted(similar_scores, key=lambda x: x[1], reverse=True)

    # Get the top N similar games
    top_similar_games = similar_games[1:num_recommendations + 1]

    # Extract the game titles
    recommended_games = [games.iloc[i[0]]['name'] for i in top_similar_games]

    return recommended_games

def recommend(game):
    user_searched_game = game
    recommendations = get_game_recommendations(user_searched_game)
    if recommendations:
        print(f"Recommended games for '{user_searched_game}':")
        return recommendations  # Return the list of recommended games
    else:
        print(f"Game '{user_searched_game}' not found in the dataset.")
        return []

st.title('GameLink')
selected_game_name = st.selectbox(
    'Select a game:',
    (games['name'].values))

if st.button('Recommend'):
    recommended_games = recommend(selected_game_name)
    if recommended_games:
        st.write(f"Recommended games for '{selected_game_name}':")
        for game in recommended_games:
            st.write(game)



