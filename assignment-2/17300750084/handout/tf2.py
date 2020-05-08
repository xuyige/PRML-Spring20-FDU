#
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import optimizers
#
# from handout.data import prepare_batch, gen_data_batch, results_converter
#
#
# class myTFRNNModel(keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.embed_layer = tf.keras.layers.Embedding(10, 32,
#                                                     batch_input_shape=[None, None])
#
#         self.rnncell = tf.keras.layers.SimpleRNNCell(64)
#         self.rnn_layer = tf.keras.layers.RNN(self.rnncell, return_sequences=True)
#         self.dense = tf.keras.layers.Dense(10)
#
#     @tf.function
#     def call(self, num1, num2):
#         '''
#         Please finish your code here.
#         '''
#         return logits
#
#
# class myAdvTFRNNModel(keras.Model):
#     def __init__(self):
#         '''
#         Please finish your code here.
#         '''
#         super().__init__()
#
#     @tf.function
#     def call(self, num1, num2):
#         '''
#         Please finish your code here.
#         '''
#         return logits
#
#
# @tf.function
# def compute_loss(logits, labels):
#     losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
#             logits=logits, labels=labels)
#     return tf.reduce_mean(losses)
#
#
# @tf.function
# def train_one_step(model, optimizer, x, y, label):
#     with tf.GradientTape() as tape:
#         logits = model(x, y)
#         loss = compute_loss(logits, label)
#
#     # compute gradient
#     grads = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     return loss
#
#
# def train(steps, model, optimizer):
#     loss = 0.0
#     accuracy = 0.0
#     for step in range(steps):
#         datas = gen_data_batch(batch_size=200, start=0, end=555555555)
#         Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
#         loss = train_one_step(model, optimizer, tf.constant(Nums1, dtype=tf.int32),
#                               tf.constant(Nums2, dtype=tf.int32),
#                               tf.constant(results, dtype=tf.int32))
#         if step % 50 == 0:
#             print('step', step, ': loss', loss.numpy())
#
#     return loss
#
#
# def evaluate(model):
#     datas = gen_data_batch(batch_size=2000, start=555555555, end=999999999)
#     Nums1, Nums2, results = prepare_batch(*datas, maxlen=11)
#     logits = model(tf.constant(Nums1, dtype=tf.int32), tf.constant(Nums2, dtype=tf.int32))
#     logits = logits.numpy()
#     pred = np.argmax(logits, axis=-1)
#     res = results_converter(pred)
#     for o in list(zip(datas[2], res))[:20]:
#         print(o[0], o[1], o[0]==o[1])
#
#     print('accuracy is: %g' % np.mean([o[0]==o[1] for o in zip(datas[2], res)]))
#
#
# def tf_main():
#     optimizer = optimizers.Adam(0.001)
#     model = myTFRNNModel()
#     train(3000, model, optimizer)
#     evaluate(model)
#
#
# def tf_adv_main():
#     '''
#     Please finish your code here.
#     '''
#     pass
