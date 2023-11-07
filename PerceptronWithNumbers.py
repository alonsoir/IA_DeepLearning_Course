# Importamos el conjunto de datos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# 1. Lectura del conjunto de datos
# Añadimos as_frame=False para forzar la devolución de un array
mnist = fetch_openml('mnist_784',  as_frame=False)

print(mnist.data)

# 2. Visualización del conjunto de datos
plt.figure(figsize=(20, 4))

for index, digit in zip(range(1, 9), mnist.data[:8]):
    plt.subplot(1, 8, index)
    plt.imshow(np.reshape(digit, (28,28)), cmap=plt.cm.gray)
    plt.title('Ejemplo: ' + str(index))
plt.show()

# Conviertiendo el conjunto de datos en un DataFrame de Pandas

df = pd.DataFrame(mnist.data)
print(df)

# 3. División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.1)

# 4. Entrenamiento del algoritmo
clf = Perceptron(max_iter=2000, random_state=40, n_jobs=-1)
clf.fit(X_train, y_train)

# Número de parámetros que forman el modelo
clf.coef_.shape

# Parámetros bias/intercept
clf.intercept_

# 5. Predicción con el conjunto de pruebas
# Realizamos la predicción con el conjunto de datos de prueba
y_pred = clf.predict(X_test)

len(y_test)

# Mostramos el f1_score resultante de la clasificación

f1_score(y_test, y_pred, average="weighted")

# 6. Mostrando las imagenes mal clasificadas
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