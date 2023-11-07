import tensorflow as tf
device = "/device:GPU:0"
with tf.device(device):
    print("GPU disponible")
# Realiza un cálculo mínimo
x = tf.constant(1.0, dtype=tf.float32)
y = tf.constant(2.0, dtype=tf.float32)
z = x + y
# Imprime el resultado
with tf.device(device):
    print(z)
