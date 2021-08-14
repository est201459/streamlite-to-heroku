import pandas as pd
import numpy as np
import joblib
import streamlit
import spacy
from typing import List
from typing import Generator, List
import pickle
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import spacy
from spacy.lang.es import Spanish
from typing import List
import os, sys

sys.path.insert(0, '../customerclassificationtigo')
sys.path.insert(0, '../customerclassificationtigo/src')
sys.path.insert(0, '../customerclassificationtigo/models')


def sentences_to_words(sentences: List[str]) -> List[List[str]]:
    words = []
    for sentence in sentences:
        words.append(simple_preprocess(str(sentence), deacc=True))
        # deacc=True elimina la puntuación
    return words


def remove_stopwords(documents: List[List[str]]) -> List[List[str]]:
    return [[word for word in simple_preprocess(str(doc)) if word not in stopwords.words('spanish')]
            for doc in documents]


def learn_bigrams(documents: List[List[str]]) -> List[List[str]]:
    # We learn bigrams
    #  https://radimrehurek.com/gensim/models/phrases.html#gensim.models.phrases.Phrases
    bigram = Phrases(documents, min_count=5, threshold=10)

    # we reduce the bigram model to its minimal functionality
    bigram_mod = Phraser(bigram)

    # we apply the bigram model to our documents
    return [bigram_mod[doc] for doc in documents]


def lemmatization(nlp: Spanish, texts: List[List[str]], allowed_postags: List = None) -> List[List[str]]:
    if allowed_postags is None:
        allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']

    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def tokenize(documents: List[str]) -> List[List[str]]:
    document_words = sentences_to_words(documents)
    document_words = remove_stopwords(document_words)
    document_words = learn_bigrams(document_words)

    return document_words


def learn_bigrams(documents: List[List[str]]) -> List[List[str]]:
    # We learn bigrams
    #  https://radimrehurek.com/gensim/models/phrases.html#gensim.models.phrases.Phrases
    bigram = Phrases(documents, min_count=5, threshold=10)

    # we reduce the bigram model to its minimal functionality
    bigram_mod = Phraser(bigram)

    # we apply the bigram model to our documents
    return [bigram_mod[doc] for doc in documents]


def predict(documents: List[str]):
    word_classes = tokenize(documents)

    with open( 'models/model.pkl','rb') as input_file:
        model = pickle.load(input_file)

    document_words = word_classes

    predictions = []
    for document in document_words:
        positive_prob = model['POS_PROB']
        negative_prob = model['NEG_PROB']
        for word in document:
            if word in model['COND_POS_PROBS']:
                positive_prob += model['COND_POS_PROBS'][word]['logprob']
            else:
                positive_prob += model['COND_POS_PROBS'][-1]['logprob']

            if word in model['COND_NEG_PROBS']:
                negative_prob += model['COND_NEG_PROBS'][word]['logprob']
            else:
                negative_prob += model['COND_NEG_PROBS'][-1]['logprob']

        if positive_prob >= negative_prob:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions


def lr_prediction(var_1):
    model_prediction1 = predict([var_1])
    if model_prediction1[0] == 1:
        model_prediction = ' Corporate'
    else:
        model_prediction = 'Mobile'
    model_prediction = model_prediction
    return model_prediction


def run():
    streamlit.title("Customer Name Classification")
    html_temp = """

    """

    streamlit.markdown(html_temp)
    var_1 = streamlit.text_input("Input customer name:")

    prediction = ""

    if streamlit.button("Predict"):
        prediction = lr_prediction(var_1)
    streamlit.success("This customer is {}".format(prediction))


if __name__ == '__main__':
    run()