
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, optimizers

from .data import generate_batch, row_to_string


class myTFRNNModel(keras.Model):
    def __init__(self, max_length):
        super().__init__()
        self.embed_layer = layers.Embedding(10, 64, input_length=max_length)
        self.rnn_layer = layers.RNN(layers.SimpleRNNCell(64), return_sequences=True)
        self.dense = layers.Dense(10, activation=tf.nn.softmax)

    @tf.function
    def call(self, num1, num2):
        logits1, logits2 = self.embed_layer(num1), self.embed_layer(num2)
        logits = tf.concat([logits1, logits2], axis=-1)
        logits = self.rnn_layer(logits)
        logits = self.dense(logits)
        return logits


class myAdvTFRNNModel(keras.Model):
    def __init__(self, max_length):
        super().__init__()
        self.embed_layer = layers.Embedding(10, 128, input_length=max_length)
        self.lstm_layer = layers.LSTM(64, return_sequences=True)
        self.dense = layers.Dense(10, activation=tf.nn.softmax)

    @tf.function
    def call(self, num1, num2):
        logits1, logits2 = self.embed_layer(num1), self.embed_layer(num2)
        logits = tf.concat([logits1, logits2], axis=-1)
        logits = self.lstm_layer(logits)
        logits = self.dense(logits)
        return logits


@tf.function
def compute_loss(logits, labels):
    y_true = tf.one_hot(labels, depth=10)
    losses = tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=logits, from_logits=True)
    return tf.reduce_mean(losses)


@tf.function
def train_one_step(model, optimizer, x, y, label):
    with tf.GradientTape() as tape:
        logits = model(x, y)
        loss = compute_loss(logits, label)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train(parser, model):
    if parser.optimizer == 'SGD':
        optimizer = optimizers.SGD(parser.learning_rate)
    elif parser.optimizer == 'Adam':
        optimizer = optimizers.Adam(parser.learning_rate)
    elif parser.optimizer == 'RMSprop':
        optimizer = optimizers.RMSprop(parser.learning_rate)
    else:
        raise RuntimeError
    parser.logger.info(''.join(['-'] * 30) + '\n' +
                       'Training process of {}:'.format(model.__class__.__name__))
    losses, steps = [], []
    if parser.plot:
        plt.figure()
        plt.clf()
        plt.ion()
        plt.title('Training Loss of {}'.format(model.__class__.__name__))

    for step in range(parser.steps):
        nums1, nums2, results = generate_batch(batch_size=parser.batch_size, max_len=parser.max_length)
        nums1, nums2, results = tf.constant(np.array(nums1, dtype=float)), tf.constant(
            np.array(nums2, dtype=float)), tf.constant(np.array(results, dtype=int))
        loss = train_one_step(model, optimizer, nums1, nums2, results)
        if step % 50 == 0:
            losses.append(loss)
            steps.append(step)
            if parser.plot:
                plt.cla()
                plt.plot(steps, losses)
                plt.title('Training Loss of {}'.format(model.__class__.__name__))
                plt.pause(0.0001)

            print('step', step, ': loss', loss.numpy())
            parser.logger.info('step ' + str(step) + ': loss ' + str(loss.numpy()))

    if parser.plot:
        plt.ioff()
        plt.show()


def evaluate(parser, model):
    parser.logger.info(''.join(['-'] * 30) + '\n' +
                       'Evaluate process of {}:'.format(model.__class__.__name__))

    nums1, nums2, results = [tf.constant(np.array(_, dtype=float)) for _ in
                             generate_batch(batch_size=parser.evaluate_batch_size, max_len=parser.max_length)]
    results = tf.keras.backend.get_value(results)
    logits = model(nums1, nums2)
    preds = np.argmax(logits, axis=-1).astype(int)
    accuracy = np.mean(np.prod(results == preds, axis=-1))
    total_accuracy = np.sum(results == preds) / results.size
    for i in range(parser.batch_size):
        result, pred = results[i, :], preds[i, :]
        line = row_to_string(result) + ' ' + row_to_string(pred) + ' ' + str((pred == result).all())
        print(line)
        parser.logger.info(line)

    print('Accuracy is : {}'.format(accuracy))
    print('Total accuracy is {}'.format(total_accuracy))
    parser.logger.info('Accuracy is : {}'.format(accuracy))
    parser.logger.info('Total accuracy is : {}'.format(total_accuracy))


def tf_main(parser):
    model = myTFRNNModel(parser.max_length)
    train(parser, model)
    evaluate(parser, model)
    parser.logger.info(''.join(['-'] * 30))


def tf_adv_main(parser):
    model = myAdvTFRNNModel(parser.max_length)
    train(parser, model)
    evaluate(parser, model)
    parser.logger.info(''.join(['-'] * 30))
