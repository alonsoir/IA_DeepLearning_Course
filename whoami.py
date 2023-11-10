import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Input, Model
import numpy as np


def entrenar_modelo(num_frames):
    # Inicializa la cámara
    cap = cv2.VideoCapture(0)
    # Maximiza la ventana de la cámara
    cv2.namedWindow("Cámara", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cámara", 1000, 1000)
    # Lista para almacenar los fotogramas capturados
    frames = []

    # Captura el número especificado de fotogramas
    for _ in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    print("Training model. num_frames: ", num_frames)
    # Cierra la cámara
    cap.release()

    # Convierte los fotogramas a un formato de color adecuado
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

    # Detecta y guarda los rostros en los fotogramas
    rostros = []
    for frame in frames:
        faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml").detectMultiScale(frame, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            rostro = frame[y:y + h, x:x + w]
            rostros.append(rostro)
        # Si no se detectaron rostros en ningún fotograma, manejar el escenario
    if not rostros:
        print("No se detectaron rostros en ningún fotograma. Asegúrate de que la cámara está capturando frames.")
        return None

    # Entrena el modelo con los rostros capturados
    model = train_model(rostros)

    # Guarda el modelo entrenado en el mismo directorio que el script
    model.save("modelo.h5")

    # Devuelve el modelo entrenado
    return model


def test_input_shape():
    """
        Prueba que la función acepta imágenes de entrada con la forma especificada.
        """
    input_shape = (32, 32, 3)
    model = create_siamese_model(input_shape)
    assert model.input_shape == (None, input_shape)


def create_siamese_model(input_shape):
    input_layer = Input(shape=input_shape)

    # Define la rama de la red siamesa
    base_network = tf.keras.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu')
    ])

    # Conecta ambas entradas a la rama siamesa
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Calcula la distancia euclidiana entre las salidas de las ramas
    distance = tf.keras.layers.Lambda(lambda x: tf.keras.backend.abs(x[0] - x[1]))([processed_a, processed_b])

    # Crea el modelo siamesa completo
    model = Model(inputs=[input_a, input_b], outputs=distance)

    return model


def train_model(rostros):
    # Convierte las imágenes de rostros a arrays numpy y normaliza
    rostros = [tf.image.per_image_standardization(rostro) for rostro in rostros]
    rostros = [tf.image.resize(rostro, (96, 96)) for rostro in rostros]  # Ajusta el tamaño según tus necesidades

    # Crea pares de imágenes para el entrenamiento (imagen de entrada, imagen de la misma persona)
    pares_positivos = [(rostro, rostro) for rostro in rostros]

    # Etiquetas para los pares (1 indica que las imágenes son de la misma persona)
    labels = [1] * len(pares_positivos)

    # Duplica las imágenes para crear pares negativos (imágenes de personas diferentes)
    pares_negativos = [(rostro, rostros[i]) for i in range(len(rostros)) if i != len(rostros) - 1 for rostro in rostros]

    # Etiquetas para los pares negativos (0 indica que las imágenes son de personas diferentes)
    labels += [0] * len(pares_negativos)

    # Combina los pares positivos y negativos
    pares = pares_positivos + pares_negativos

    # Convierte a numpy arrays
    pairs = np.array([np.stack(pair, axis=0) for pair in pares])
    labels = np.array(labels)

    # Crea y compila el modelo siamesa
    input_shape = rostros[0].shape
    model = create_siamese_model(input_shape)

    # test_input_shape()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Agregar estas líneas en la función train_model, antes de model.fit
    model.summary()  # Para visualizar la arquitectura del modelo

    # Entrena el modelo
    model.fit([pairs[:, 0], pairs[:, 1]], labels, epochs=10, batch_size=64)

    return model


def main():
    # Entrenar el modelo con 10 fotogramas
    model = entrenar_modelo(num_frames=50)

    # Intenta cargar el modelo entrenado
    try:
        model = load_model("modelo.h5")
        print("Modelo cargado exitosamente.")
    except:
        print("Error al cargar el modelo.")

    # Inicializa la cámara nuevamente
    cap = cv2.VideoCapture(0)

    # Bucle infinito para la detección en tiempo real
    while True:
        ret, frame = cap.read()

        if ret:
            # Convierte el fotograma a un formato de color adecuado
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detecta los rostros en el fotograma
            faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml").detectMultiScale(frame, 1.3, 5)

            # Si se detecta un rostro
            for (x, y, w, h) in faces:
                # Recorta el rostro
                rostro = frame[y:y + h, x:x + w]
                # Ajusta el tamaño según las dimensiones esperadas por tu modelo
                rostro_redimensionado = cv2.resize(rostro, (96, 96))

                # Normaliza la imagen
                rostro_normalizado = tf.image.per_image_standardization(rostro_redimensionado)

                # Predice la identidad del rostro usando un par negativo
                prediccion = model.predict(
                    [tf.reshape(rostro_normalizado, (1, 96, 96, 3)), tf.reshape(rostro_normalizado, (1, 96, 96, 3))])

                print(prediccion[0][0])
                # Si la predicción es correcta
                if prediccion[0][0] > 0.5:
                    # Dibuja un cuadro sobre el rostro
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Escribe el texto "ADMIN" sobre el cuadro
                    cv2.putText(frame, "ADMIN", (x + 10, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Muestra el fotograma
            cv2.imshow("Cámara", frame)

            # Espera a que se presione una tecla
            key = cv2.waitKey(1)

            # Si se presiona la tecla q, sale del bucle
            if key == ord("q"):
                break

    # Cierra la cámara
    cap.release()

    # Destruye todas las ventanas abiertas
    cv2.destroyAllWindows()


if __name__ == "__main__":
    device = "/device:GPU:0"
    with tf.device(device):
        print("GPU disponible")
        main()
    print("Done!")
