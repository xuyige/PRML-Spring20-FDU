import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10, 32,
                                batch_input_shape=[None, None])
    # tf.keras.layers.
])
model.summary()