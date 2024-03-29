import nltk

nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def analisador(text):
    dicionario_sentimento = sia.polarity_scores(text)
    print("Dados da análise: ")
    print(f"Positivo: {dicionario_sentimento['pos'] * 100} % ")
    print(f"Negativo: {dicionario_sentimento['neg'] * 100} % ")
    print(f"Neutro: {dicionario_sentimento['neu'] * 100} % ")
    print(f"Compound: {dicionario_sentimento['compound']} ")
    print("*******************************************************************")
    print("*******************************************************************")



#Noticia alegre
text2 = "Eu gosto de comer pizza e me sinto muito bem, gosto de camarão, " \
        "todos gostam, cansei, não gosto de nada, nem de comida, nem de sair de casa"

analisador(text2)



