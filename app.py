import sys

from jf_tokenize_package.tokenize import tokenize

import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize, pos_tag

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
import re

import nltk


# def tokenize(text):
#     """
#     Create lemmatized tokens from words in a string.
#
#     Input:
#     A string made of oen to several sentences.
#
#     Output:
#     A list of tokenised and lemmatized words.
#     """
#     # Replacing urls in text with placeholder
#     from nltk.corpus import stopwords
#
#     detected_urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'\
#                                '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
#     for url in detected_urls:
#         text = text.replace(url,"urlplaceholder")
#
#
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()
#     sw_nltk = stopwords.words('english')
#
#     clean_tokens = []
#     for tok in tokens:
#         if tok not in sw_nltk:
#             clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#             clean_tokens.append(clean_tok)
#
#     return clean_tokens

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messageslabeled', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    genre_df = pd.DataFrame(list(zip(genre_names,genre_counts)), columns=['Genres','Counts'])
    genre_df.sort_values(by='Counts', ascending=False, inplace=True)

    cat = pd.DataFrame(df.iloc[:,3:].sum(axis=0), columns=['count'])
    cat.sort_values(by='count', ascending=False, inplace=True)



    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_df['Genres'],
                    y=genre_df['Counts']
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=list(cat.index),
                    y=cat['count'].values
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }


    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

# def main():
#     # app.run(host='0.0.0.0', port=3001, debug=True)
#     app.run(debug=True)
#
# if __name__ == '__main__':
#     main()
