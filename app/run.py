
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
import re



app = Flask(__name__)

def tokenize(text):
    """
    Create lemmatized tokens from words in a string.

    Input:
    A string made of oen to several sentences.

    Output:
    A list of tokenised and lemmatized words.
    """
    # Replacing urls in text with placeholder
    detected_urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in detected_urls:
        text = text.replace(url,"urlplaceholder")


    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class VerbCounter(BaseEstimator, TransformerMixin):
    """
    Custom transformer to make a feature of the number of verbs in a message.
    """

    def counting_verbs(self, text):
        """
        Counts the number of verbs in a text using pos_tags.

        Input:
        The text.

        Output:
        The number of verbs in the text.
        """
        # tokenize by sentences
        sentence_list = sent_tokenize(text)

        verbs_count = 0
        verb_tags = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']

        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            pos_tags = pos_tag(word_tokenize(sentence))

            # iterate tags to count verbs
            for i in range(len(pos_tags)):
                if pos_tags[i][1] in verb_tags:
                    verbs_count +=1

        return verbs_count

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply counting_verb function to all values in X
        X_tagged = pd.Series(X).apply(self.counting_verbs)

        return pd.DataFrame(X_tagged)

class PronounsCounter(BaseEstimator, TransformerMixin):
    """
    Custom transformer to make a feature of the number of pronouns in a message.
    """

    def counting_prons(self, text):
        """
        Counts the number of pronouns in a text using pos_tags.

        Input:
        The text.

        Output:
        The number of pronouns in the text.
        """
        # tokenize by sentences
        sentence_list = sent_tokenize(text)

        prons_count = 0
        pron_tags = ['PRP', 'PRP$']

        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            pos_tags = pos_tag(word_tokenize(sentence))

            # iterate tags to count verbs
            for i in range(len(pos_tags)):
                if pos_tags[i][1] in pron_tags:
                    prons_count +=1

        return prons_count


    def fit(self, x, y=None):
            return self

    def transform(self, X):
            # apply counting_prons function to all values in X
            X_tagged = pd.Series(X).apply(self.counting_prons)

            return pd.DataFrame(X_tagged)


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

    cat = pd.DataFrame(df.iloc[:,3:].sum(axis=0), columns=['count'])



    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
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
                    x=cat['count'].values,
                    y=list(cat.index)
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

def main():
    # app.run(host='0.0.0.0', port=3001, debug=True)
    app.run(debug=True)

if __name__ == '__main__':
    main()
