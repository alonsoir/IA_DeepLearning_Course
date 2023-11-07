from sklearn.datasets import load_breast_cancer
from MPNeuron import MPNeuron
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
# Para transformar un valor a binario
import matplotlib.pyplot as plt

'''
Primero, se cargan los datos del conjunto de cáncer de mama. Luego se dividen en características (X) y etiquetas (Y). 
Después, se convierten las características a un valor binario mediante el uso de pd.cut y se separan los datos en 
conjuntos de entrenamiento y prueba.


En el código principal, se instancia la clase MPNeuron, se ajusta (fit) el modelo al conjunto de entrenamiento binarizado, 
se selecciona el umbral óptimo y se realiza la predicción sobre el conjunto de prueba. 
Finalmente, se calcula la precisión (accuracy_score) y la matriz de confusión (confusion_matrix).
'''
breast_cancer = load_breast_cancer()

X = breast_cancer.data
Y = breast_cancer.target

# print(breast_cancer)
df = pd.DataFrame(X, columns=breast_cancer.feature_names)
# print(df)
# df.plot()

# plt.show()

X_train, X_test, y_train, y_test = train_test_split(df, Y, stratify=Y)

# print("Tamaño del conjunto de datos de entrenamiento: ", len(X_train))
# print("Tamaño del conjunto de datos de pruebas: ", len(X_test))

# print(pd.cut([0.04, 2, 4, 5, 6, 0.02, 0.6], bins=2, labels=[0, 1]))

# plt.hist([0.04, 0.3, 4, 5, 6, 0.02, 0.6], bins=2)
# plt.show()

# Transformamos las caracteríticas de entrada a un valor binario
X_train_bin = X_train.apply(pd.cut, bins=2, labels=[1, 0])
X_test_bin = X_test.apply(pd.cut, bins=2, labels=[1, 0])
# Instanciamos el modelo MPNeuron
mp_neuron = MPNeuron()

# Encontramos el threshold óptimo
mp_neuron.fit(X_train_bin.to_numpy(), y_train)

# Threshold óptimo seleccionado
print("threshold: ",mp_neuron.threshold)

# Realizamos predicciones para ejemplos nuevos que no se encuentran en el conjunto de datos de entrenamiento
Y_pred = mp_neuron.predict(X_test_bin.to_numpy())

# Calculamos la exactitud de nuestra predicción
print("accuracy: ",accuracy_score(y_test, Y_pred))

# Calculamos la matriz de confusión

matrix = confusion_matrix(y_test, Y_pred)

print(matrix)

'''
El valor en la esquina superior izquierda (50) indica la cantidad de muestras que fueron clasificadas correctamente 
como la clase negativa (considerando este lado de la matriz como la clase negativa).
El valor en la esquina superior derecha (3) indica la cantidad de muestras que fueron clasificadas incorrectamente 
como la clase positiva (falsos positivos).
El valor en la esquina inferior izquierda (12) indica la cantidad de muestras que fueron clasificadas incorrectamente 
como la clase negativa (falsos negativos).
El valor en la esquina inferior derecha (78) indica la cantidad de muestras que fueron clasificadas correctamente 
como la clase positiva.
'''
