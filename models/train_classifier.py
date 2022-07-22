import sys
sys.path.append('..')

from utilities.utils import tokenize

import numpy as np
import pandas as pd

import pickle

from sqlalchemy import create_engine

from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report,  f1_score
from sklearn.base import BaseEstimator, TransformerMixin

import re
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','omw-1.4',
               'stopwords'])


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
#
# # class VerbsCounter(BaseEstimator, TransformerMixin):
# #
# #     def counting_verbs(self, text):
# #         # tokenize by sentences
# #         sentence_list = sent_tokenize(text)
# #
# #         verbs_count = 0
# #         verb_tags = ['VB', 'VBG', 'VBD', 'VBN', 'VBP', 'VBZ']
# #
# #         for sentence in sentence_list:
# #             # tokenize each sentence into words and tag part of speech
# #             pos_tags = pos_tag(word_tokenize(sentence))
# #
# #             # iterate tags to count verbs
# #             for i in range(len(pos_tags)):
# #                 if pos_tags[i][1] in verb_tags:
# #                     verbs_count +=1
# #
# #         return verbs_count
# #
# #     def fit(self, x, y=None):
# #         return self
# #
# #     def transform(self, X):
# #         # apply counting_verb function to all values in X
# #         X_tagged = pd.Series(X).apply(self.counting_verbs)
# #
# #         return pd.DataFrame(X_tagged)
# #
# # class PronounsCounter(BaseEstimator, TransformerMixin):
# #
# #     def counting_prons(self, text):
# #         # tokenize by sentences
# #         sentence_list = sent_tokenize(text)
# #
# #         prons_count = 0
# #         pron_tags = ['PRP', 'PRP$']
# #
# #         for sentence in sentence_list:
# #             # tokenize each sentence into words and tag part of speech
# #             pos_tags = pos_tag(word_tokenize(sentence))
# #
# #             # iterate tags to count verbs
# #             for i in range(len(pos_tags)):
# #                 if pos_tags[i][1] in pron_tags:
# #                     prons_count +=1
# #
# #         return prons_count
# #
# #
# #     def fit(self, x, y=None):
# #             return self
# #
# #     def transform(self, X):
# #             # apply counting_prons function to all values in X
# #             X_tagged = pd.Series(X).apply(self.counting_prons)
# #
# #             return pd.DataFrame(X_tagged)
# #

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messageslabeled', con=engine)


    X = df.message.values
    y = df.drop(['id', 'message', 'genre'], axis=1)
    category_names = y.columns.tolist()

    return X, y, category_names



def build_model():

    # Creating pipeline
    pipeline = Pipeline([

    ('text_pipeline', Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer(use_idf=True))
    ])),

    ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=42)))
    ])

    # Setting up the gridsearch for parameters tuning

    parameters = {
    'clf__estimator__learning_rate': [0.5, 1.0],
    'clf__estimator__n_estimators': [10, 20, 50]
    }

    cv = GridSearchCV(pipeline, parameters, verbose=3, n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=category_names)

    f1_scores = []

    for category in category_names:
        print('{} classification report:'.format(category))
        print(classification_report(Y_test[category], Y_pred[category]))
        f1_scores.append(f1_score(Y_test[category], Y_pred[category],
                         average='micro'))

    print('Average of labels micro averaged' \
          'f1 scores: {}'.format(sum(f1_scores)/len(f1_scores)))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
