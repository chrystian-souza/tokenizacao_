import  spacy

nlp = spacy.load("pt_core_news_sm")
texto = "Fantasia demorou 40 dias para ficar pronta, segundo a fabricante da roupa. Em publicação, Toko-san agradeceu a empresa: 'Graças a vocês," \
        " consegui realizar meu sonho de me tornar um animal. "\
        "Um japonês se transformou em um cão da raça Collie, investindo R$ 75 mil em uma fantasia ultrarrealista."\
        "Toko-san se apresentou como o Collie em abril, quando estreou seu canal no YouTube. " \
        "Lá,ele já publicou diversos vídeos. Nas imagens, ele reproduz ações do animal e aparece brincando" \
        " com sua pelúcia ou uma bolinha e, também, dá a patinha."



doc = nlp(texto)
for token in doc:
    print(f" {token.text}:{token.pos_} ")


