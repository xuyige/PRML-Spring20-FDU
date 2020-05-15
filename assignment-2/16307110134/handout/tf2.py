
import numpy as np
import tensorflow as tf
import collections
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

from .data import prepare_batch, gen_data_batch, results_converter
from .data import  adv_gen_data_batch
class myTFRNNModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.embed_layer = tf.keras.layers.Embedding(10, 32,
                                                    batch_input_shape=[None, None])

        self.rnncell = tf.keras.layers.SimpleRNNCell(64)
        self.rnn_layer = tf.keras.layers.RNN(self.rnncell, return_sequences=True)
        self.dense = tf.keras.layers.Dense(10)

    @tf.function
    def call(self, num1, num2):
        num1 = self.embed_layer(num1)
        num2 = self.embed_layer(num2)
        input_concat = tf.concat([num1,num2],2)
        x = self.rnn_layer(input_concat)
        logits = self.dense(x)
        return logits


class myAdvTFRNNModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.embed_layer = tf.keras.layers.Embedding(10, 32,
                                                    batch_input_shape=[None, None])
        self.gru_layer1 = tf.keras.layers.GRU(64, return_sequences = True)
        self.dense = tf.keras.layers.Dense(10)
        
        
    @tf.function
    def call(self, num1, num2):
        num1 = self.embed_layer(num1)
        num2 = self.embed_layer(num2)
        input_concat = tf.concat([num1,num2],2)
        
        x = self.gru_layer1(input_concat)
        
        logits = self.dense(x)
        return logits


@tf.function
def compute_loss(logits, labels):
    print(logits)
    print(labels)
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
    for step in range(steps):
        datas = gen_data_batch(batch_size=200, start=0, end=555555555)
        Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
        loss = train_one_step(model, optimizer, tf.constant(Nums1, dtype=tf.int32),
                              tf.constant(Nums2, dtype=tf.int32),
                              tf.constant(results, dtype=tf.int32))
        
        if step % 50 == 0:
            print('step', step, ': loss', loss.numpy())

    return loss


def evaluate(model):
    datas = gen_data_batch(batch_size=2000, start=555555555, end=999999999)
    Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
    logits = model(tf.constant(Nums1, dtype=tf.int32), tf.constant(Nums2, dtype=tf.int32))
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    for o in list(zip(datas[2], res))[:20]:
        print(o[0], o[1], o[0]==o[1])

    print('accuracy is: %g' % np.mean([o[0]==o[1] for o in zip(datas[2], res)]))
    

@tf.function
def adv_train_one_step(model, optimizer, x, y, label):
    with tf.GradientTape() as tape:
        logits = model(x, y)
        loss = compute_loss(logits, label)

    # compute gradient
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def adv_train(steps, model, optimizer,maxlen = 11,maxlen_test = 11):
    loss = 0.0
    maxnum = 0
    for i in range(maxlen-1):
        maxnum *= 10 
        maxnum += 1
    
    for step in range(steps):
        datas = adv_gen_data_batch(batch_size=200, start=0, end = maxnum * 5)
        Nums_1, Nums_2, results = prepare_batch(*datas, maxlen)
        loss = adv_train_one_step(model, optimizer, tf.constant(Nums_1, dtype=tf.int32),
                              tf.constant(Nums_2, dtype=tf.int32),
                              tf.constant(results, dtype=tf.int32))
        if step % 50 == 0:
            print('step', step, ': loss', loss.numpy())

    return loss


def adv_evaluate(model,maxlen = 11):
    maxnum = 0
    for i in range(maxlen-1):
        maxnum *= 10 
        maxnum += 1
    datas = adv_gen_data_batch(batch_size=2000, start=maxnum * 5, end= maxnum * 9)
    Nums_1, Nums_2, results_ = prepare_batch(*datas, maxlen)
    logits = model(tf.constant(Nums_1, dtype=tf.int32), tf.constant(Nums_2, dtype=tf.int32))
    logits = logits.numpy()
    pred = np.argmax(logits, axis=-1)
    res = results_converter(pred)
    for o in list(zip(datas[2], res))[:20]:
        print(o[0], o[1], o[0]==o[1])
    
    acc = np.mean([o[0]==o[1] for o in zip(datas[2], res)])
    print('accuracy is: %g\n' % acc)
    return acc

def tf_main():
    optimizer = optimizers.Adam(0.001)
    model = myTFRNNModel()
    train(3000, model, optimizer)
    evaluate(model)
    tf.compat.v1.reset_default_graph()


def tf_adv_main():
    maxlen_train = maxlen_test = 100
    optimizer = optimizers.Adam(0.01)
    model = myAdvTFRNNModel()
    adv_train(500,model,optimizer,maxlen_train,maxlen_test)
    adv_evaluate(model,maxlen_test)
    tf.compat.v1.reset_default_graph()
