"""Model architecture of default (saved) LanguageModel - TF2 version"""

import tensorflow as tf


class LanguageModel(tf.keras.Model):
    """Language model using Keras RNN layers for TF2."""

    def __init__(self, vocabulary_size, embedding_size, num_layers, num_hidden, mode='train'):
        super(LanguageModel, self).__init__()

        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.vocabulary_size = vocabulary_size
        self.mode = mode

        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=embedding_size,
            embeddings_initializer='uniform',
            name="embeddings"
        )

        # Build stacked RNN cells with dropout
        self.rnn_cells = []
        for i in range(num_layers):
            cell = tf.keras.layers.SimpleRNNCell(num_hidden, name=f"rnn_cell_{i}")
            self.rnn_cells.append(cell)

        if len(self.rnn_cells) > 1:
            self.stacked_cell = tf.keras.layers.StackedRNNCells(self.rnn_cells)
        else:
            self.stacked_cell = self.rnn_cells[0]

        self.rnn_layer = tf.keras.layers.RNN(
            self.stacked_cell,
            return_sequences=True,
            return_state=True,
            name="rnn"
        )

        # Output dense layer
        self.output_dense = tf.keras.layers.Dense(vocabulary_size, name="output")

        # Dropout (applied during call)
        self.dropout_layer = tf.keras.layers.Dropout(0.0)

    def call(self, x, seq_len=None, initial_state=None, keep_prob=1.0, training=None):
        """
        Forward pass of the language model.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor of shape (batch_size, seq_length) containing token indices
        seq_len : tf.Tensor or None
            Sequence lengths for masking
        initial_state : tuple or None
            Initial RNN state
        keep_prob : float
            Keep probability for dropout (1.0 - dropout_rate)
        training : bool or None
            Whether in training mode

        Returns
        -------
        logits : tf.Tensor
            Output logits of shape (batch_size, seq_length, vocabulary_size)
        last_state : tuple
            Final RNN state
        """
        # Update dropout rate
        self.dropout_layer.rate = 1.0 - keep_prob

        # Embedding lookup
        lm_input_emb = self.embedding(x)

        # Apply dropout to embeddings
        lm_input_emb = self.dropout_layer(lm_input_emb, training=training)

        # Create mask from sequence lengths if provided
        mask = None
        if seq_len is not None:
            mask = tf.sequence_mask(seq_len, maxlen=tf.shape(x)[1])

        # Run RNN
        if initial_state is not None:
            rnn_outputs = self.rnn_layer(lm_input_emb, mask=mask, initial_state=initial_state, training=training)
        else:
            rnn_outputs = self.rnn_layer(lm_input_emb, mask=mask, training=training)

        # rnn_outputs is (output, state) or (output, state1, state2, ...) for multi-layer
        outputs = rnn_outputs[0]
        last_state = rnn_outputs[1:]
        if len(last_state) == 1:
            last_state = last_state[0]

        # Apply dropout to outputs
        outputs = self.dropout_layer(outputs, training=training)

        # Dense layer to get logits
        logits = self.output_dense(outputs)

        return logits, last_state

    def compute_loss(self, x, target, seq_len):
        """
        Compute sequence loss.

        Parameters
        ----------
        x : tf.Tensor
            Input tensor
        target : tf.Tensor
            Target tensor
        seq_len : tf.Tensor
            Sequence lengths

        Returns
        -------
        loss : tf.Tensor
            Average cross-entropy loss
        """
        logits, _ = self.call(x, seq_len=seq_len, training=True)

        # Create mask
        mask = tf.sequence_mask(seq_len, maxlen=tf.shape(target)[1], dtype=tf.float32)

        # Compute cross-entropy loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            target, logits, from_logits=True
        )

        # Apply mask and average
        masked_loss = loss * mask
        loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)

        return loss


# Legacy interface for compatibility
class LanguageModelLegacy(object):
    """Legacy interface wrapper for backward compatibility."""

    def __init__(self, vocabulary_size, embedding_size, num_layers, num_hidden, mode='train'):
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden

        # Create the Keras model
        self.model = LanguageModel(vocabulary_size, embedding_size, num_layers, num_hidden, mode)

        # For compatibility with old code that accesses these attributes
        self.x = None
        self.keep_prob = None
        self.batch_size = None
        self.seq_len = None
        self.logits = None
        self.initial_state = None
        self.last_state = None
        self.loss = None

    def get_logits(self, x, keep_prob=1.0, initial_state=None):
        """Get logits for input x."""
        seq_len = tf.reduce_sum(tf.cast(tf.not_equal(x, 0), tf.int32), axis=1)
        logits, last_state = self.model(x, seq_len=seq_len, initial_state=initial_state,
                                        keep_prob=keep_prob, training=False)
        return logits, last_state
