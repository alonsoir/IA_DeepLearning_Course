import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Lee el fichero de texto como una lista de líneas
with open("SMSSpamCollection.txt", encoding='utf-8') as f:
    lines = f.readlines()

# Divide las líneas en etiquetas y texto del mensaje
labels = []
messages = []
for line in lines:
    parts = line.split('\t')
    labels.append(parts[0])
    messages.append(parts[1])

# Crea un DataFrame a partir de las etiquetas y mensajes
df = pd.DataFrame({'label': labels, 'message': messages})

# Convierte las etiquetas 'ham' y 'spam' a valores binarios (0 y 1)
df['spam'] = (df['label'] == 'spam').astype(int)

# Dividimos el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['spam'], test_size=0.25, random_state=42)

# Creamos un vectorizador TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Creamos un modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluamos el modelo en el conjunto de datos de prueba
score = model.score(X_test, y_test)

print('Score:', score)

# Tomamos 10 muestras aleatorias del DataFrame original para predecir si son spam o no
samples = df.sample(n=1000)

# Usamos el vectorizador TF-IDF ya ajustado en el modelo para transformar los textos de estos 10 emails
sample_features = vectorizer.transform(samples['message'])

# Hacemos predicciones
predictions = model.predict(sample_features)

# Mapeamos los valores predichos a 'Yes' o 'No'
predictions_text = ['Yes' if pred == 1 else 'No' for pred in predictions]

# Agregamos las predicciones al DataFrame de las muestras seleccionadas
samples['predicted_spam'] = predictions_text

print(samples[['message', 'predicted_spam']])
