import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
import numpy as np


def loadMnistDataset():
    global dataset, info, fashion_mnist, X_train, y_train, X_test, y_test
    dataset, info = tfds.load('fashion_mnist', split='train',
                              data_dir='/Users/aironman/tensorflow_datasets/fashion_mnist/3.0.1/', with_info=True)
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    plt.figure(figsize=(20, 4))
    for index, img in zip(range(1, 9), X_train[:8]):
        plt.subplot(1, 8, index)
        plt.imshow(img, cmap=plt.cm.gray)
        plt.title('Ejemplo: ' + str(index))
    print("Longitud subconjunto de entrenamiento: ", len(X_train))
    print("Longitud subconjunto de pruebas: ", len(X_test))

def transformDataset():
    global X_train_prep, X_test_prep, y_train_prep, y_test_prep
    X_train_prep = X_train.astype('float32') / 255.0
    X_test_prep = X_test.astype('float32') / 255.0
    X_train_prep = X_train_prep.reshape((60000, 28, 28, 1))
    X_test_prep = X_test_prep.reshape((10000, 28, 28, 1))
    y_train_prep = to_categorical(y_train)
    y_test_prep = to_categorical(y_test)

def buildModel():
    checkpoint = tf.keras.callbacks.ModelCheckpoint("best_mnist_model.h5", monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    network = tf.keras.models.Sequential()
    network.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))
    network.add(tf.keras.layers.Dense(400, activation='relu'))
    network.add(tf.keras.layers.Dense(300, activation='relu'))
    network.add(tf.keras.layers.Dense(10, activation='softmax'))
    network.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'Precision'])
    history = network.fit(X_train_prep, y_train_prep, batch_size=32, epochs=20, validation_data=(X_test_prep, y_test_prep), callbacks=[checkpoint])

def showRandomImage(X_test_prep):
    random_index = random.randint(0, len(X_test_prep) - 1)
    random_image = X_test_prep[random_index]
    plt.imshow(random_image.squeeze(), cmap='gray')
    plt.title('Imagen Aleatoria')
    plt.axis('off')
    plt.show()

def preprocessImage(image):
    # Reshape the image to the correct input shape for the model
    image = np.reshape(image, (1, 28, 28, 1))
    return image

def makePrediction(image, model):
    # Preprocess the image
    image = preprocessImage(image)

    # Make the prediction
    predictions = model.predict(image)

    # Get the predicted class
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]

    # Print the prediction
    print(f'Predicción del modelo: {predicted_class}')

def loadModelAndCheckAccuracy():
    model = load_model('best_mnist_model.h5')
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (_, _), (X_test, y_test) = fashion_mnist.load_data()
    X_test_prep = X_test.astype('float32') / 255.0
    X_test_prep = X_test_prep.reshape((10000, 28, 28, 1))
    y_test_prep = to_categorical(y_test)
    accuracy = model.evaluate(X_test_prep, y_test_prep)
    print(f'Precisión actual del modelo: {accuracy}')

    return X_test_prep

def full():
    loadMnistDataset()
    transformDataset()
    # buildModel()
    X_test_prep = loadModelAndCheckAccuracy()
    showRandomImage(X_test_prep)
    model = load_model('best_mnist_model.h5')
    image = preprocessImage(X_test_prep[0])
    makePrediction(image, model)
    image = preprocessImage(X_test_prep[1])
    makePrediction(image, model)

    print("Done!")


def makeAnotherPrediction(X_test_prep, model):
    # Make a prediction on a random image from the test set
    random_index = random.randint(0, len(X_test_prep) - 1)
    image = X_test_prep[random_index]
    # Preprocess the image
    image = preprocessImage(image)
    # Make the prediction
    prediction = makePrediction(image, model)
    # Print the prediction
    print(f'Predicción del modelo: {prediction}')


if __name__ == "__main__":
    full()
