import cv2
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Input, Model
import numpy as np


def train_model(num_frames):
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    # Maximize the camera window
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", 1000, 1000)
    # List to store the captured frames
    frames = []

    # Capture the specified number of frames
    for _ in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    print("Training model. num_frames: ", num_frames)
    # Close the camera
    cap.release()

    # Convert the frames to a suitable color format
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

    # Detect and store the faces in the frames
    faces = []
    for frame in frames:
        faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml").detectMultiScale(frame, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = frame[y:y + h, x:x + w]
            faces.append(face)
        # If no faces were detected in any frame, handle the scenario
    if not faces:
        print("No faces were detected in any frame. Make sure the camera is capturing frames.")
        return None

    # Convert the face images to numpy arrays and normalize
    faces = [tf.image.per_image_standardization(face) for face in faces]
    faces = [tf.image.resize(face, (96, 96)) for face in faces]  # Adjust the size according to your needs

    # Create pairs of images for training (input image, image of the same person)
    positive_pairs = [(face, face) for face in faces]

    # Labels for the pairs (1 indicates that the images are of the same person)
    labels = [1] * len(positive_pairs)

    # Duplicate the images to create negative pairs (images of different people)
    negative_pairs = [(face, faces[i]) for i in range(len(faces)) if i != len(faces) - 1 for face in faces]

    # Labels for the negative pairs (0 indicates that the images are of different people)
    labels += [0] * len(negative_pairs)

    # Combine the positive and negative pairs
    pairs = positive_pairs + negative_pairs

    # Convert to numpy arrays
    pairs = np.array([np.stack(pair, axis=0) for pair in pairs])
    labels = np.array(labels)

    # Create and compile the siamese model
    input_shape = faces[0].shape
    model = create_siamese_model(input_shape)

    # Add a callback to save the best iteration of the model
    checkpoint_path = "model.ckpt"
    epochs = 10
    batch_size = 32
    # Train the model
    # Add a callback to plot the accuracy and loss
    history = model.fit([pairs[:, 0], pairs[:, 1]], labels, epochs=epochs, batch_size=batch_size, callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir="./logs"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, min_lr=0.0001),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_best_only=True, monitor="val_accuracy"
        )
    ])

    # Plot the accuracy and loss
    plt.plot(history.history["accuracy"], label="Accuracy")
    plt.plot(history.history["loss"], label="Loss")
    plt.title("Training Accuracy and Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy/Loss")
    plt.legend()
    plt.show()

    return model


def test_input_shape():
    """
        Tests that the function accepts input images with the specified shape.
        """
    input_shape = (32, 32, 3)
    model = create_siamese_model(input_shape)
    assert model.input_shape == (None, input_shape)


def create_siamese_model(input_shape):
    input_layer = Input(shape=input_shape)

    # Define the siamese network branch
    base_network = tf.keras.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu')
    ])

    # Connects both inputs to the siamese branch
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    # Calculates the euclidean distance between the outputs of the branches
    distance = tf.keras.layers.Lambda(lambda x: tf.keras.backend.abs(x[0] - x[1]))([processed_a, processed_b])

    # Creates the complete siamese model
    model = Model(inputs=[input_a, input_b], outputs=distance)

    return model


def train_model(rostros):
    # Convert the face images to numpy arrays and normalize
    rostros = [tf.image.per_image_standardization(rostro) for rostro in rostros]
    rostros = [tf.image.resize(rostro, (96, 96)) for rostro in rostros]  # Adjust the size according to your needs

    # Create pairs of images for training (input image, image of the same person)
    pares_positivos = [(rostro, rostro) for rostro in rostros]

    # Labels for the pairs (1 indicates that the images are of the same person)
    labels = [1] * len(pares_positivos)

    # Duplicate the images to create negative pairs (images of different people)
    pares_negativos = [(rostro, rostros[i]) for i in range(len(rostros)) if i != len(rostros) - 1 for rostro in rostros]

    # Labels for the negative pairs (0 indicates that the images are of different people)
    labels += [0] * len(pares_negativos)

    # Combine the positive and negative pairs
    pares = pares_positivos + pares_negativos

    # Convert to numpy arrays
    pairs = np.array([np.stack(pair, axis=0) for pair in pares])
    labels = np.array(labels)

    # Create and compile the siamese model
    input_shape = rostros[0].shape
    model = create_siamese_model(input_shape)

    # Add a callback to save the best iteration of the model
    checkpoint_path = "model.ckpt"
    callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_best_only=True, monitor="val_accuracy"
    )

    # Train the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()  # To visualize the architecture of the model
    model.fit([pairs[:, 0], pairs[:, 1]], labels, epochs=10, batch_size=64, callbacks=[callback])

    return model


def main():
    # Train the model with 10 frames
    model = train_model(rostros=1000)

    # Try to load the trained model
    try:
        model = load_model("model.ckpt")
        print("Modelo cargado exitosamente.")
    except:
        print("Error al cargar el modelo.")

    # Initialize the camera again
    cap = cv2.VideoCapture(0)

    # Infinite loop for real-time detection
    while True:
        ret, frame = cap.read()

        if ret:
            # Convert the frame to a suitable color format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect the faces in the frame
            faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml").detectMultiScale(frame, 1.3, 5)

            # If a face is detected
            for (x, y, w, h) in faces:
                # Crop the face
                rostro = frame[y:y + h, x:x + w]
                # Resize the face to the expected dimensions of the model
                rostro_redimensionado = cv2.resize(rostro, (96, 96))

                # Normalize the face
                rostro_normalizado = tf.image.per_image_standardization(rostro_redimensionado)

                # Predict the identity of the face using a negative pair
                prediccion = model.predict(
                    [tf.reshape(rostro_normalizado, (1, 96, 96, 3)), tf.reshape(rostro_normalizado, (1, 96, 96, 3))])

                # If the prediction is correct
                if prediccion[0][0] > 0.5:
                    # Draw a rectangle around the face
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Write the text "ADMIN" above the rectangle
                    cv2.putText(frame, "ADMIN", (x + 10, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show the frame
            cv2.imshow("Cámara", frame)

            # Wait for a key press
            key = cv2.waitKey(1)

            # If the key q is pressed, exit the loop
            if key == ord("q"):
                break

    # Close the camera
    cap.release()


if __name__ == "__main__":
    main()
