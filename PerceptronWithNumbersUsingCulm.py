import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from cuml.linear_model import Perceptron
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import os
from joblib import dump, load

# 1. Lectura del conjunto de datos
mnist = fetch_openml('mnist_784', as_frame=False)

# 3. División del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.1)

# Rutas del modelo
model_path = "perceptron_model.joblib"

if os.path.exists(model_path):
    # Si el modelo ya está guardado, cargarlo
    clf = load(model_path)
    print("Modelo cargado desde el archivo.")
else:
    # Si el modelo no está guardado, entrenarlo
    clf = Perceptron(max_iter=2000, random_state=40)
    clf.fit(X_train, y_train)
    dump(clf, model_path)  # Guardar el modelo
    print("Modelo entrenado y guardado.")

# 5. Predicción con el conjunto de pruebas
y_pred = clf.predict(X_test)

# Mostrar el accuracy del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Mostrar el f1_score resultante de la clasificación
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"F1 Score: {f1}")

# Resto del código para mostrar imágenes clasificadas
# (sin cambios, ya que la predicción se realiza independientemente de la GPU)
