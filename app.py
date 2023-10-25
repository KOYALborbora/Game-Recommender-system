from flask import Flask,render_template
import pandas as pd
import pickle

games = pickle.load(open('games.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',game_name = list(games['name'].values),image = list(games['header_image'].values))


if __name__ =='__main__':
    app.run(debug=True)