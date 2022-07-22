import re

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','omw-1.4',
              'stopwords'])

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, sent_tokenize, pos_tag

def tokenize(text):
    """
    Create lemmatized tokens from words in a string and remove stopwords.

    Input:
    Text as a string.

    Output:
    A list of tokenised and lemmatized words.
    """
    # Replacing urls in text with placeholder
    from nltk.corpus import stopwords

    detected_urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'\
                               '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for url in detected_urls:
        text = text.replace(url,"urlplaceholder")


    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    sw_nltk = stopwords.words('english')

    clean_tokens = []
    for tok in tokens:
        if tok not in sw_nltk:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens
