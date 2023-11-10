import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def testGPU():
    # Realiza un cálculo mínimo
    x = tf.constant(1.0, dtype=tf.float32)
    y = tf.constant(2.0, dtype=tf.float32)
    z = x + y
    # Imprime el resultado
    print(z)

def loadMNIST(prefix,folder ):
    intType = np.dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile(folder + "/" + prefix + "-images-idx3-ubyte", dtype="ubyte")
    magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
    data = data[nMetaDataBytes:].astype(dtype="float32").reshape([nImages, width, height])

    labels = np.fromfile(folder + "/" + prefix + "-labels-idx1-ubyte",dtype="ubyte")[2 * intType.itemsize:]

    return data, labels
def show_image(image):
    image_array = image.get_array()
    image_array = image_array.astype(np.float32)
    plt.imshow(image_array, cmap="gray")
    plt.show()

def main():
    testGPU()
    images_tr, labels_tr = loadMNIST("train", "./data")
    images_te, labels_te = loadMNIST("t10k", "./data")
    # imagenes en array de 60000 x 28 x 28 -> 60000 imagenes de 28x28
    # coger una aleatoria del grupo de test
    i = random.randint(0, images_te.shape[0])
    img_test = images_te[i].flatten()
    label_test = labels_te[i]
    # buscamos los vecinos más cercanos (KNN)
    k = 5  # número de vecinos

    distances = []
    for i in range(images_tr.shape[0]):
        dist = np.sqrt(np.sum(np.square(images_tr[i].flatten() - img_test)))
        distances.append((dist, labels_tr[i]))  # guardamos las etiquetas y la distancia

    # ordenamos por distancia y nos quedamos con los k vecinos más cercanos
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    # contamos los votos para ver qué etiqueta gana. Hay 10 etiquetas, para predecir qué número del 0 al 9 es el votado.
    votes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for neighbor in neighbors:
        votes[neighbor[1]] = votes[neighbor[1]] + 1
    # obtenemos la etiqueta ganadora
    pred_label = votes.index(max(votes))
    print("Predicted: " + str(pred_label))
    print("Real: " + str(label_test))
    img = plt.imshow(img_test.reshape(28, 28))
    show_image(img)


if __name__ == "__main__":
    device = "/device:GPU:0"
    with tf.device(device):
        print("GPU disponible")
        main()
        print("DONE!")
