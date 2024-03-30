import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from heapq import nlargest
import string
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from colorama import Fore, Style

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text_pt(text):
    tokens = word_tokenize(text.lower(), language='portuguese')
    stop_words = set(stopwords.words('portuguese'))
    words = [word for word in tokens if word.isalnum() and word not in stop_words and word not in string.punctuation]
    return " ".join(words)

def generate_summary(text, num_sentences):
    sentences = sent_tokenize(text, language='portuguese')
    preprocess_text = preprocess_text_pt(text)
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('portuguese'))
    tfidf_matrix = tfidf_vectorizer.fit_transform([preprocess_text])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    words_scores = {}
    for word, score in zip(feature_names, tfidf_matrix.toarray()[0]):
        words_scores[word] = score
    sentences_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_words = word_tokenize(sentence.lower(), language='portuguese')
        score = sum(words_scores[word] for word in sentence_words if word in words_scores)
        sentences_scores[i] = score / len(sentence_words) if len(sentence_words) > 0 else 0
    selected_sentences = nlargest(num_sentences, sentences_scores, key=sentences_scores.get)
    summary = ' '.join(sentences[i] for i in sorted(selected_sentences))
    return summary

cor_resumo = Fore.GREEN
bot = ChatBot('Bot')

trainer = ListTrainer(bot)

trainer.train([
    'Oi',
    'Olá, como posso ajudar?',
    'Quero um resumo do meu texto',
    'Texto resumido, se ficou como esperava me dê um ok. :)',
    'ok, ficou ótimo, obrigado!',
    'não',
    'ok, tchau.'
    
])

while True:
    request = input('Você: ')
    if request.lower() == 'ok':
        print('Bot: Tchau')
        break
    elif 'quero um resumo do meu texto' in request:
        texto = input("Digite o texto que deseja resumir: ")
        resumo = generate_summary(texto, 2)
        print(f'{cor_resumo}Resumo:{resumo}{Style.RESET_ALL}')
    else:
        response = bot.get_response(request)
        print('Bot:', response)
