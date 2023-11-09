# pip install -U spacy
# python -m spacy download en_core_web_sm
# python -m spacy download es_core_news_sm
import csv

import spacy


def nlp_refactored(_nlp, _text):
    global entity
    # Load English tokenizer, tagger, parser and NER
    # Ahora cargo el tokenizer para español, el parser
    nlp = spacy.load(_nlp)
    # Process whole documents
    text = _text
    doc = nlp(text)
    # Analyze syntax
    print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
    print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])
    # Find named entities, phrases and concepts
    for entity in doc.ents:
        print(entity.text, entity.label_)


def nlp_sentiment_analysis(_text, model="es_core_news_sm"):
    # Comprobar si el texto es una cadena
    if not isinstance(_text, str):
        raise ValueError("El texto debe ser una cadena.")

    # Cargar el modelo de lenguaje
    nlp = spacy.load(model)

    # Cargar el texto
    text = _text

    # Analizar el texto
    doc = nlp(text)

    # Obtener los sentimientos de cada palabra
    sentiments = [token.sentiment for token in doc]

    # Imprimir los sentimientos
    print(sentiments)

    # Devolver más información sobre el análisis de sentimientos
    return sentiments, doc.sentiment


nlp = spacy.load("es_core_news_sm")


nlp_es = "es_core_news_sm"

nlp_uk = "en_core_web_sm"
# Process whole documents
text_uk = ("When Sebastian Thrun started working on self-driving cars at "
           "Google in 2007, few people outside of the company took him "
           "seriously. “I can tell you very senior CEOs of major American "
           "car companies would shake my hand and turn away because I wasn’t "
           "worth talking to,” said Thrun, in an interview with Recode earlier "
           "this week.")

# Cargar el texto
text_es = "La película fue increíble. La historia era muy emocionante y los personajes eran muy simpáticos. Me encantó el final."

sentiment_Es = nlp_sentiment_analysis(text_es, nlp_es)
print(sentiment_Es)
sentiment_UK = nlp_sentiment_analysis(text_uk, nlp_uk)
print(sentiment_UK)
