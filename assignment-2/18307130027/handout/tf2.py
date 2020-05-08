
import numpy as np
import tensorflow as tf
import collections
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

from .data import prepare_batch, gen_data_batch, results_converter

import matplotlib.pyplot as plt
class myTFRNNModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.embed_layer = layers.Embedding(10, 32,
                                                    batch_input_shape=[None, None])
        self.rnncell = layers.SimpleRNNCell(64)
        self.rnn_layer = layers.RNN(self.rnncell, return_sequences=True)
        self.dense = layers.Dense(10)

    @tf.function
    def call(self, num1, num2):
        '''
        Please finish your code here.
        '''
        x1, x2 = self.embed_layer(num1), self.embed_layer(num2)
        y = layers.concatenate([x1, x2])
        z = self.rnn_layer(y)
        return self.dense(z)


class myAdvTFRNNModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.embed_layer = layers.Embedding(10, 32,
                                                    batch_input_shape=[None, None])
        self.rnncell = layers.LSTMCell(64)
        self.rnn_layer = layers.RNN(self.rnncell, return_sequences=True)
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.dense3 = layers.Dense(10)

    @tf.function
    def call(self, num1, num2):
        '''
        Please finish your code here.
        '''
        x1, x2 = self.embed_layer(num1), self.embed_layer(num2)
        y1, y2 = self.rnn_layer(x1), self.rnn_layer(x2)
        z = layers.concatenate([y1,y2])
        w1 = self.dense1(z)
        w2 = self.dense2(w1)
        return self.dense3(w2)


@tf.function
def compute_loss(logits, labels):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
    return tf.reduce_mean(losses)


@tf.function
def train_one_step(model, optimizer, x, y, label):
    with tf.GradientTape() as tape:
        logits = model(x, y)
        loss = compute_loss(logits, label)

    # compute gradient
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train(steps, model, optimizer):
    loss = 0.0
    accuracy = 0.0
    try:
        for step in range(steps):
            datas = gen_data_batch(batch_size=200, start=0, end=555555555)
            Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
            loss = train_one_step(model, optimizer, tf.constant(Nums1, dtype=tf.int32),
                                tf.constant(Nums2, dtype=tf.int32),
                                tf.constant(results, dtype=tf.int32))
            if step % 50 == 0:
                print('step', step, ': loss', loss.numpy())
                plt.plot(step,loss, 'rx')
    except KeyboardInterrupt:
        pass
    plt.show()
    return loss


def evaluate(model):
    datas = gen_data_batch(batch_size=2000, start=555555555, end=999999999)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
    logits = model(tf.constant(Nums1, dtype=tf.int32), tf.constant(Nums2, dtype=tf.int32))
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    for o in list(zip(datas[2], res))[:40]:
        print(o[0], o[1], o[0]==o[1])

    print('accuracy is: %g' % np.mean([o[0]==o[1] for o in zip(datas[2], res)]))


def tf_main():
    optimizer = optimizers.Adam(0.01)
    model = myTFRNNModel()
    train(3000, model, optimizer)
    model.summary()
    # keras.utils.plot_model(model, show_shapes= True)
    # keras.utils.model_to_dot(model).cre
    evaluate(model)


def tf_adv_main():
    optimizer = optimizers.Adam(0.01)
    model = myAdvTFRNNModel()
    train(3000, model, optimizer)
    model.summary()
    # keras.utils.plot_model(model, show_shapes= True)
    # keras.utils.model_to_dot(model).cre
    evaluate(model)