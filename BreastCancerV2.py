from sklearn.datasets import load_breast_cancer
from MPNeuron import MPNeuron
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

'''
1. Análisis exploratorio de datos
    
    Agrega visualizaciones y estadísticas descriptivas para comprender mejor los 
    datos antes de modelar:

    Visualizaciones: Histogramas, diagramas de dispersión, correlaciones, etc.
    Estadísticas descriptivas: Media, desviación estándar, cuartiles, etc.

2. Optimización del modelo

    Prueba con diferentes umbrales para el modelo MPNeuron y evalúa la 
    precisión en el conjunto de pruebas para seleccionar el mejor umbral.
    Considera otras métricas de evaluación del modelo, como precisión, recall, F1-score.

3. Comentarios y explicaciones
    
    Agrega comentarios explicativos al código para que sea más comprensible.
'''

# Cargar el conjunto de datos
breast_cancer = load_breast_cancer()

# Dividir en características y etiquetas
X = breast_cancer.data
Y = breast_cancer.target

# Crear un DataFrame para visualización
df = pd.DataFrame(X, columns=breast_cancer.feature_names)

# Dividir los datos en conjuntos de entrenamiento y pruebas
X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.2, random_state=42)

# Análisis exploratorio de datos (ejemplo: histograma de una característica)
plt.hist(X[:, 0], bins=20)
plt.title('Histograma de la primera característica')
plt.show()

# Instanciar y ajustar el modelo MPNeuron
mp_neuron = MPNeuron()

# Ajustar el modelo con diferentes umbrales y seleccionar el mejor
threshold_values = [10, 20, 30, 40]  # Ejemplo de diferentes valores de umbral
best_threshold = None
best_accuracy = 0

for threshold in threshold_values:
    mp_neuron.fit((X_train > threshold), y_train)
    y_pred = mp_neuron.predict(X_test > threshold)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print("Mejor threshold:", best_threshold)
print("Exactitud con el mejor threshold:", best_accuracy)

# Evaluar el modelo con el mejor threshold
y_pred_best = mp_neuron.predict(X_test > best_threshold)

# Métricas de evaluación
print("Exactitud:", accuracy_score(y_test, y_pred_best))
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred_best))
print("Precisión:", precision_score(y_test, y_pred_best))
print("Recall:", recall_score(y_test, y_pred_best))
print("F1-score:", f1_score(y_test, y_pred_best))
