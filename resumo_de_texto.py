import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from heapq import nlargest
import string

def preprocess_text_pt(text):
    tokens = word_tokenize(
        text.lower(), language='portugueses'
    )
    stop_words = set(stopwords.words('portuguese'))
    words = [word for word in tokens if word.isalnum() and word not in stop_words and word not in string.punctuation]
    return " ".join(words)

def genetate_summary(text, num_sentences):
    sentences = sent_tokenize(text, language='portuguese')
    preprocess_text = preprocess_text_pt(text)
    tfidf_vectorizer = TfidfVectorizer(stopwords=stopwords.words('portuguese'))
    tfidf_matrix = tfidf_vectorizer.fit_trandform([preprocess_text])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    words_scores = {}
    for word, score in zip(feature_names, tfidf_matrix.toarray()[0]):
        words_scores[word] = score

    sentences_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_words = word_tokenize(sentence.lower(), language='portuguese')
        score = sum(words_scores[word] for word in sentence_words if word in words_scores)
        sentences_scores[i] = score / len(sentence_words) if len(sentence_words) > 0 else 1





