import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import models, layers
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import tensorflow as tf


def loadData():
    columnas = ['target', 'ids', 'date', 'flag', 'user', 'text']
    data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding="latin-1", names=columnas)
    # nos quedamos con una selección de 50000 tweets aleatorios
    data = data.sample(n=50000, random_state=42)
    data = data.reset_index(drop=True)  # reconstruimos los índices para que sean consecutivos
    # data.head()
    # print(data)
    plt.hist(data['target'], bins=5)

    # ahora la etiqueta para los tweets positivos es el 1
    data.loc[data['target'] == 4, 'target'] = 1
    plt.hist(data['target'], bins=3)

    return data


# codificar tweets -> corpus -> one-hot
# con limit podemos limitar el número de palabras del corpus
def construir_corpus(tweets, limit=10000):
    corpus = []
    i = 0
    for t in tweets:
        if i > limit:
            break
        for w in t.lower().split():
            # si la palabra no está en el corpus y no empieza por @ (usuario twitter)
            if w[0] not in ('@') and 'http' not in w and len(w) > 3 and not w in corpus:
                corpus.append(w)
                i = i + 1

    # codificar palabras como enteros
    t = Tokenizer()
    t.fit_on_texts(corpus)
    encoded_corpus = t.texts_to_matrix(corpus, mode='count')

    return corpus[:limit], encoded_corpus[:limit]


def codifica_tweets(tweets, corpus, corpus_size=10000):
    coded = np.zeros((len(tweets), corpus_size))
    for i, tweet_text in enumerate(tweets):
        words = tweet_text.lower().split()
        for w in words:
            if w in corpus:
                coded[i, corpus.index(w)] = 1

    return coded

'''
Este método trata de cargar un modelo preentrenado, si no lo está, lo entrena. 
Ahora mismo se tiene que refactorizar para que solo en caso de reentraniento, habría que cargar el dataset...
'''
def trainModel(data):
    global x_test, y_test
    x_train, x_val, y_train, y_val = loadTweetData(data)

    if os.path.exists('twitter_model_trained.h5'):
        # Si el modelo preentrenado existe, intenta cargarlo
        print("Cargando modelo preentrenado...")
        model = models.load_model('twitter_model_trained.h5')
    else:
        print("Entrenando nuevo modelo...")
        # print(x_train[0])
        # print(y_train[0])
        model = models.Sequential()
        model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(2, activation='softmax'))
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        train_log = model.fit(x_train, y_train,
                              epochs=5, batch_size=512,
                              validation_data=(x_val, y_val))
        acc = train_log.history['acc']
        val_acc = train_log.history['val_acc']
        loss = train_log.history['loss']
        val_loss = train_log.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        test_loss, test_accuracy = model.evaluate(x_test, y_test)
        print("test_accuracy: ", test_accuracy)
        # Guardar el modelo entrenado
        model.save('twitter_model_trained.h5')
        print("Modelo entrenado y guardado como 'model_trained.h5'.")

    return model, x_test, y_test


def loadTweetData(data):
    print("loadTweetData...")
    global x_test, y_test
    # Ahora usamos los tweets del dataset
    tweets = data['text']
    corpus, encoded_corpus = construir_corpus(tweets)
    # print(corpus)
    # print(encoded_corpus)  # esto no lo vamos a necesitar para nuestro ejemplo
    # echo 1 > /proc/sys/vm/overcommit_memory
    x_train = codifica_tweets(data['text'], corpus)
    y_train = to_categorical(data['target'])
    # barajamos el dataset
    np.random.seed(42)
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation]
    y_train = y_train[permutation]
    # obtenemos el conjunto de validación
    num_val = 10000
    x_val = x_train[:num_val]
    x_train = x_train[num_val:]
    y_val = y_train[:num_val]
    y_train = y_train[num_val:]
    # obtenemos el conjunto de test
    x_test = x_train[:num_val]
    x_train = x_train[num_val:]
    y_test = y_train[:num_val]
    y_train = y_train[num_val:]
    print("loadTweetData DONE!")

    return x_train, x_val, y_train, y_val


def main():
    # Probamos con tres frases para confirmar que todo va bien
    # tweets = ['Sólo sé que no sé nada', 'Pienso luego existo', 'viva @socrates y @descartes']

    # corpus, encoded_corpus = construir_corpus(tweets)
    # print(corpus)
    # print(encoded_corpus)
    data = loadData()
    model, x_test, y_test = trainModel(data)

    predictions = model.predict(x_test)
    print("predictions: ", predictions)
    print("clase más probable: ", np.argmax(predictions[1]))  # clase más probable
    print("y_test[1]: ", y_test[1])


if __name__ == "__main__":
    device = "/device:GPU:0"
    with tf.device(device):
        main()
