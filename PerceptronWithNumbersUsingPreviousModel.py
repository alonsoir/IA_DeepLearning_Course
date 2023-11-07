import os

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.datasets import mnist

# Obtener la ruta del directorio actual del script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Ruta para almacenar los datos MNIST
mnist_path = os.path.join(script_dir, 'mnist_images')

# Verificar si la carpeta mnist_images ya existe, de lo contrario, crearla
if not os.path.exists(mnist_path):
    os.makedirs(mnist_path)

# Comprobar si el conjunto de datos ya está descargado
if not os.path.exists(os.path.join(mnist_path, 'X_train.npy')):
    # Descargar MNIST si no está en la ruta especificada
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    np.save(os.path.join(mnist_path, 'X_train.npy'), X_train)
    np.save(os.path.join(mnist_path, 'y_train.npy'), y_train)
    np.save(os.path.join(mnist_path, 'X_test.npy'), X_test)
    np.save(os.path.join(mnist_path, 'y_test.npy'), y_test)
else:
    # Cargar datos desde la ruta especificada
    X_train = np.load(os.path.join(mnist_path, 'X_train.npy'))
    y_train = np.load(os.path.join(mnist_path, 'y_train.npy'))
    X_test = np.load(os.path.join(mnist_path, 'X_test.npy'))
    y_test = np.load(os.path.join(mnist_path, 'y_test.npy'))

# Asegurar que las imágenes estén en el formato correcto
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Remodelar las imágenes para que coincidan con las expectativas del modelo
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

# Ahora, los datos están listos para ser utilizados en el modelo


# Rutas del modelo y verificación
model_path = "perceptron_model.joblib"

if os.path.exists(model_path):
    # Si el modelo ya está guardado, cargarlo
    clf = load(model_path)
    print("Modelo cargado desde el archivo.")
else:
    # Si el modelo no está guardado, entrenarlo
    clf = Perceptron(max_iter=2000, random_state=40, n_jobs=-1)
    clf.fit(X_train, y_train)
    dump(clf, model_path)  # Guardar el modelo
    print("Modelo entrenado y guardado.")

# 5. Predicción con el conjunto de pruebas
y_pred = clf.predict(X_test)

# Convertir etiquetas de texto a números enteros
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)
y_pred_encoded = label_encoder.transform(y_pred)

# Mostrar el accuracy del modelo
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print(f"Accuracy: {accuracy}")

# Mostrar el f1_score resultante de la clasificación
f1 = f1_score(y_test_encoded, y_pred_encoded, average="weighted")
print(f"F1 Score: {f1}")
# 6. Mostrando las imágenes mal clasificadas
index = 0
index_errors = []
index_correct = []

for label, predict in zip(y_test, y_pred):
    if label != predict:
        index_errors.append(index)
    else:
        index_correct.append(index)
    index += 1

# Muestra las imágenes mal clasificadas
plt.figure(figsize=(20, 4))
for i, img_index in zip(range(1, 9), index_errors[:8]):
    plt.subplot(1, 8, i)
    plt.imshow(np.reshape(X_test[img_index], (28, 28)), cmap=plt.cm.gray)
    plt.title('Orig:' + str(y_test[img_index]) + ' Pred:' + str(y_pred[img_index]))
plt.show()

# Muestra las imágenes bien clasificadas
plt.figure(figsize=(20, 4))
for i, img_index in zip(range(1, 9), index_correct[:8]):
    plt.subplot(1, 8, i)
    plt.imshow(np.reshape(X_test[img_index], (28, 28)), cmap=plt.cm.gray)
    plt.title('Orig:' + str(y_test[img_index]) + ' Pred:' + str(y_pred[img_index]))
plt.show()
