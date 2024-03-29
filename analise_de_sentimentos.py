import nltk

nltk.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

text = "Os criminosos estão sempre à procura de formas rápidas de fazer dinheiro. Desta vez, as vítimas têm sido donos de Hyundai Creta da antiga geração, cujas molduras das colunas dianteiras se tornaram alvo de uma onda de furtos."
text += "Sobre a forma como os donos do SUV têm tido suas molduras furtadas, os relatos costumam ser muito parecidos: os criminosos costumam agir quando o dono deixa o carro estacionado na rua e quando não há ninguém por perto. Para desencaixar a peça, bastam poucos segundos e quase nenhum esforço."
text += "Uma solução encontrada pelos donos, que costuma ser bastante comentada em postagens sobre o tema, a fixação da peça com cola de para-brisa (ou cola P.U). Segundo relatos nas redes sociais, isso já é o bastante para dificultar a vida do ladrão.Ainda que rodar com o Creta sem essa peça não seja algo problemático,"
text += " a estética fica comprometida. Além disso, são maiores as chances da ausência da moldura provocar acúmulo de água e detritos nas canaletas das portas."
text += "Outro risco que acaba surgindo é o da compra de uma peça furtada, já que os ladrões costumam levar o item para revendê-lo."
text += "Por isso, como sempre, é preciso ficar atento em relação à procedência e sempre exigir a nota fiscal."
text += "Mas vale salientar que, se o procedimento não for realizado adequadamente (sem a limpeza correta da superfície antes da aplicação ou com quantidade insuficiente de cola, por exemplo), isso pode acabar não impedindo a ação dos ladrões."
text += "Fácil de remover, a peça é cara de repor, pois passa dos R$ 600 (o par) ou R$ 400 (cada) em anúncios na internet."

dicionario_sentimento = sia.polarity_scores(text)

print("Dados da análise: ")
print(f"Positivo: {dicionario_sentimento['pos'] * 100} % ")
print(f"Negativo: {dicionario_sentimento['neg'] * 100} % ")
print(f"Neutro: {dicionario_sentimento['neu'] * 100} % ")
print(f"Compound: {dicionario_sentimento['compound']} ")
