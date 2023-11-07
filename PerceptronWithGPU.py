import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Configurar para usar la GPU
device = "/device:GPU:0"

with tf.device(device):
    # Realiza un cálculo mínimo
    x = tf.constant(1.0, dtype=tf.float32)
    y = tf.constant(2.0, dtype=tf.float32)
    z = x + y
    # Imprime el resultado
    print("GPU disponible!",z)

# Cargar datos de MNIST utilizando scikit-learn
mnist = fetch_openml('mnist_784', as_frame=False)
X, y = mnist["data"], mnist["target"]

# Convertir los datos a un formato numérico si es necesario
X = X.astype('float32')  # Convertir a tipo float32 si no lo son

X = X / 255.0  # Normalizar los datos.
# Las imágenes en el conjunto de datos se convierten a números de punto flotante
# y se normalizan dividiéndolas por 255 (escalando los valores de píxeles entre 0 y 1).

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Convertir etiquetas a números enteros
y_train = y_train.astype('int')  # Asegúrate de que las etiquetas sean enteros

# Convertir etiquetas a formato one-hot
y_train = to_categorical(y_train, num_classes=10)
# Asegúrate de que el número de clases
# sea adecuado.
# Se dividen los datos en conjuntos de entrenamiento y prueba,
# y se convierten las etiquetas a formato one-hot, lo que significa que las etiquetas
# se convierten de un solo número a un vector de ceros y unos (indicando a qué
# clase pertenece cada imagen).

# Convertir etiquetas a números enteros
y_test = y_test.astype('int')  # Asegúrate de que las etiquetas sean enteros

# Convertir etiquetas a formato one-hot
y_test = to_categorical(y_test, num_classes=10)
# Asegúrate de que el número de clases sea adecuado, es decir, como estamos tratando de
# detectar los numeros del 0 al 9, num_classes vale 10.

# Creación del modelo de red neuronal
model_path = "my_perceptron_model"

try:
    # Cargar el modelo si ya está guardado
    print("Modelo", model_path, "no encontrado. Cargando...")
    model = tf.keras.models.load_model(model_path)
    print("Modelo cargado desde el archivo.")
except (OSError, ValueError):
    print("Modelo", model_path, "no cargado. Entrenando...")
    # Si el modelo no está guardado, entrenarlo
    with tf.device(device):
        # Aquí se crea un modelo de red neuronal simple, con una capa de entrada de 784
        # neuronas (correspondiente a la forma de las imágenes de MNIST),
        # una capa oculta con 10 neuronas (una para cada dígito) y la función de
        # activación softmax.
        # El modelo se compila y entrena con los datos de entrenamiento, y luego se guarda
        # en disco.
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        # Compilación del modelo
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])  # Usar 'categorical_crossentropy'

        # Entrenamiento del modelo
        model.fit(X_train, y_train, epochs=50, batch_size=32)

        # Guardar el modelo entrenado
        model.save(model_path)
        print("Modelo entrenado y guardado.")

# Evaluación del modelo
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {test_accuracy}")

with tf.device(device):
    # Predicción de las primeras 8 imágenes
    predictions = model.predict(X_test[:8])

plt.figure(figsize=(20, 4))
for i in range(8):
    plt.subplot(1, 8, i + 1)
    plt.imshow(np.reshape(X_test[i], (28, 28)), cmap=plt.cm.gray)
    plt.title(f"Predicted: {np.argmax(predictions[i])}")
plt.show()
'''
with tf.device(device):
    # Predicción de todas las imágenes del conjunto de datos de prueba
    all_predictions = model.predict(X_test)

# Visualización de las predicciones
num_images = X_test.shape[0]  # Número total de imágenes en el conjunto de datos de prueba

plt.figure(figsize=(20, 4 * (num_images // 8 + 1)))  # Asegura el espacio suficiente para todas las imágenes

for i in range(num_images):
    plt.subplot((num_images // 8) + 1, 8, i + 1)
    plt.imshow(np.reshape(X_test[i], (28, 28)), cmap=plt.cm.gray)
    plt.title(f"Predicted: {np.argmax(all_predictions[i])}")

plt.show()
'''