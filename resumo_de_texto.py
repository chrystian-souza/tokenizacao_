import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from heapq import nlargest
import string

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text_pt(text):
    tokens = word_tokenize(
        text.lower(), language='portuguese'
    )
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

text =  (" Havia uma vez uma pequena cidade aninhada entre montanhas verdejantes, onde o sol nascia todas as manhãs pintando o céu de tons dourados e rosados. Esta cidade, chamada Vale Verde, era conhecida por sua tranquilidade e beleza natural.\n"
        "\n"
        "No coração de Vale Verde, erguia-se uma velha árvore centenária, cujos galhos se estendiam em todas as direções, como abraços acolhedores para os moradores da cidade. Esta árvore, chamada de 'Ancião', era o símbolo da comunidade e abrigava segredos e histórias de gerações passadas.\n"
        "\n"
        "Certa tarde, enquanto as crianças brincavam nas ruas de paralelepípedos e os adultos se ocupavam com suas tarefas diárias, um forasteiro apareceu na cidade. Ele vestia roupas gastas e carregava uma mala surrada. Seu nome era Lúcio e ele era um contador de histórias.\n"
        "\n"
        "Lúcio sentou-se sob a sombra do Ancião e começou a contar suas histórias para os moradores de Vale Verde. Ele falou sobre lugares distantes, aventuras emocionantes e personagens extraordinários que cativaram a imaginação de todos que o ouviam.\n"
        "\n"
        "À medida que as semanas passavam, Lúcio se tornou uma figura querida na cidade. Os moradores aguardavam ansiosamente suas histórias todas as tardes, reunindo-se em volta do Ancião para ouvir suas palavras mágicas.\n"
        "\n"
        "Um dia, Lúcio revelou uma história especial, uma história sobre o poder da comunidade, da amizade e da solidariedade. Ele contou sobre como os moradores de Vale Verde se uniram para superar desafios difíceis e como encontraram força uns nos outros.\n"
        "\n"
        "Após terminar sua história, Lúcio sorriu para a multidão, sabendo que deixaria uma marca duradoura em seus corações. Os moradores de Vale Verde aprenderam com ele que, embora as histórias possam ser contadas, são os laços entre as pessoas que verdadeiramente tornam uma comunidade especial.\n"
        "\n"
        "E assim, sob a sombra do Ancião, Lúcio continuou a contar suas histórias, compartilhando sabedoria e inspiração com todos que tinham a sorte de ouvi-lo.")

summary = generate_summary(text, 2)
print(summary)
