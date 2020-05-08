import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers

from .data import generate_batch, row_to_string


class myCascadeTFRNNModel(keras.Model):
    def __init__(self, max_length, unit_max_length):
        super().__init__()
        self.max_length = max_length
        self.unit_max_length = unit_max_length
        if max_length % unit_max_length is not 0:
            raise RuntimeError
        self.adder = self.adderUnit(unit_max_length)

    @tf.function
    def call(self, nums1, nums2, carrys=None, training=False):
        if training:
            if carrys is None:
                raise RuntimeError
            return self.adder(carrys, nums1, nums2)
        else:
            if carrys is not None:
                raise RuntimeError
            carrys = tf.zeros([nums1.shape[0], self.unit_max_length])
            n = self.max_length // self.unit_max_length
            logits = []
            for i in range(n):
                beg, end = i * self.unit_max_length, (i + 1) * self.unit_max_length
                logits_each_unit, carrys = self.adder(carrys, nums1[:, beg:end], nums2[:, beg:end])
                carrys = tf.expand_dims(tf.argmax(carrys, axis=-1), axis=1)
                carrys = tf.concat([carrys, tf.zeros([carrys.shape[0], self.unit_max_length - 1], dtype=tf.int64)], axis=-1)
                logits.append(logits_each_unit)
            logits = tf.concat(logits, axis=1)
            return carrys, logits

    class adderUnit(keras.Model):
        def __init__(self, max_length):
            super().__init__()
            self.max_length = max_length
            self.embed_layer1 = layers.Embedding(2, 128, input_length=max_length)
            self.embed_layer2 = layers.Embedding(10, 128, input_length=max_length)
            self.lstm1 = layers.LSTM(128, return_sequences=True)
            self.lstm2 = layers.LSTM(128, return_sequences=False)
            self.dense1 = layers.Dense(10, activation=tf.nn.softmax)
            self.dense2 = layers.Dense(2, activation=tf.nn.softmax)

        @tf.function
        def call(self, carrys, nums1, nums2):
            logits1 = self.embed_layer1(carrys)
            logits2 = self.embed_layer2(nums1)
            logits3 = self.embed_layer2(nums2)
            logits = tf.concat([logits1, logits2, logits3], axis=-1)
            logits1 = self.lstm1(logits)
            logits2 = self.lstm2(logits)
            logits1 = self.dense1(logits1)
            logits2 = self.dense2(logits2)
            return logits1, logits2


@tf.function
def compute_loss(results, carry_out, results1, carry_out1):
    y_true1 = tf.one_hot(results, depth=10)
    y_true2 = tf.one_hot(carry_out, depth=2)
    losses1 = tf.keras.losses.categorical_crossentropy(y_true=y_true1, y_pred=results1, from_logits=True)
    losses2 = tf.keras.losses.categorical_crossentropy(y_true=y_true2, y_pred=carry_out1, from_logits=True)
    losses = tf.reduce_mean(losses1) + 0.2 * tf.reduce_mean(losses2)
    return losses


@tf.function
def train_one_step(model, optimizer, carry_in, nums1, nums2, results, carry_out):
    with tf.GradientTape() as tape:
        results1, carry_out1 = model(nums1, nums2, training=True, carrys=carry_in)
        loss = compute_loss(results, carry_out[:, 0], results1, carry_out1)

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
        carry_in, nums1, nums2, results, carry_out = generate_batch(batch_size=parser.batch_size,
                                                                    max_len=parser.unit_max_length, return_carry=True)
        carry_in, nums1, nums2, results, carry_out = tf.constant(np.array(carry_in, dtype=int)), tf.constant(
            np.array(nums1, dtype=float)), tf.constant(np.array(nums2, dtype=float)), tf.constant(
            np.array(results, dtype=int)), tf.constant(np.array(carry_out, dtype=int))
        loss = train_one_step(model, optimizer, carry_in, nums1, nums2, results, carry_out)
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
    _, logits = model(nums1, nums2)
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


def tf_cascade_main(parser):
    model = myCascadeTFRNNModel(parser.max_length, parser.unit_max_length)
    train(parser, model)
    evaluate(parser, model)
    parser.logger.info(''.join(['-'] * 30))
