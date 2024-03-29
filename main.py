# Tokenização por sentenças
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

text = "Olá, mundo! Estou vendo sobre IA, bem legal né mas estou com sono! Acredito que preciso de uma xícara de café. Acabei de ver que não tem café."

sentencas = sent_tokenize(text)

print('Sentenças Tokenizadas', sentencas)


print(sentencas[0])
print(sentencas[1])
print(sentencas[2])
print(sentencas[3])



