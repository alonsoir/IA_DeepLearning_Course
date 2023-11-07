import numpy as np
from sklearn.metrics import accuracy_score

'''
La clase MPNeuron es definida para representar el modelo de la neurona M-P (neurona de McCulloch-Pitts (MPNeuron) ).

Sirvió para mostrar el potencial pero no se usa en el mundo real. El accuracy no es adecuado.

__init__(self): Inicializa el modelo con un atributo threshold establecido en None.
model(self, x): Evalúa si la suma de las características es mayor o igual que el umbral (threshold). 
                Si es así, devuelve True, de lo contrario False.
predict(self, X): Realiza predicciones sobre el conjunto de datos proporcionado X utilizando el umbral establecido 
                    y devuelve una matriz de resultados.
fit(self, X, Y): Encuentra el mejor umbral que maximiza la precisión del modelo. 
                Evalúa diferentes umbrales, desde 0 hasta el número de características de entrada + 1, 
                y selecciona el umbral que proporciona la mayor precisión.
'''
class MPNeuron:

    def __init__(self):
        self.threshold = None

    def model(self, x):
        return (sum(x) >= self.threshold)

    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)

    def fit(self, X, Y):
        accuracy = {}
        # Seleccionamos un threshold entre el # de características de entrada
        for th in range(X.shape[1] + 1):
            self.threshold = th
            Y_pred = self.predict(X)
            accuracy[th] = accuracy_score(Y_pred, Y)
        # Seleccionamos el threshold que mejores resultados proporciona
        self.threshold = max(accuracy, key=accuracy.get)