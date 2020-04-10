"""Model architecture of default (saved) LanguageModel"""

import tensorflow as tf
from tensorflow.contrib import rnn

class LanguageModel(object):
    def __init__(self, vocabulary_size, embedding_size, num_layers, num_hidden, mode='train'):
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden

        self.x = tf.compat.v1.placeholder(tf.int32, [None, None], name="x") # whole seq + seq len
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, [], name="keep_prob")
        self.batch_size = tf.compat.v1.shape(self.x)[0]

        if mode == 'train':
            self.lm_input = self.x[:, :-2]
            self.seq_len = self.x[:, -1]
        elif mode == 'predict':
            self.lm_input = self.x[:,:]
            self.seq_len = tf.reduce_sum(tf.sign(self.lm_input), 1)

        self.logits=tf.Variable(2.0, name="logits")

        # embedding, one-hot encoding
        # if embedding:
        with tf.name_scope("embedding"):
            init_embeddings = tf.random.uniform([vocabulary_size, self.embedding_size])
            embeddings = tf.compat.v1.get_variable("embeddings", initializer=init_embeddings)
            lm_input_emb = tf.nn.embedding_lookup(embeddings, self.lm_input)

        with tf.compat.v1.variable_scope("rnn"):
            def make_cell():
                cell = rnn.BasicRNNCell(self.num_hidden)
                cell = rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                return cell

            cell = rnn.MultiRNNCell([make_cell() for _ in range(self.num_layers)])

            self.initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

            # rnn_outputs: [batch_size, max_len, num_hidden(cell output)]
            rnn_outputs, self.last_state = tf.nn.dynamic_rnn(
                cell=cell, 
                initial_state=self.initial_state,
                inputs=lm_input_emb,
                sequence_length=self.seq_len, 
                dtype=tf.float32)

        # with tf.name_scope("output"):
        self.logits = tf.layers.dense(rnn_outputs, vocabulary_size)


        with tf.name_scope("loss"):
            if mode == "train":
                target = self.x[:, 1:-1]
            elif mode == "predict":
                target = self.x[:, :]

            self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=self.logits,
                targets=target,
                weights=tf.sequence_mask(self.seq_len, tf.shape(self.x)[1] - 2, dtype=tf.float32),
                average_across_timesteps=True,
                average_across_batch=True
            )

