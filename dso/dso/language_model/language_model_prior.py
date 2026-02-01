"""Language model to get prior"""

import pickle

import numpy as np
import tensorflow as tf

from .model.model_dyn_rnn import LanguageModel

class LanguageModelPrior(object):
    """
    Language model to get prior for DSO, given token.

    History of tokens of a sequence is holded as a state of language model.
    Usage: LanguageModelPrior.get_lm_prior(token)

    Parameters
    ----------
    dso_library: dso.library.Library
        Library used in main DSO model

    model_path: str
        Path to separately trained mathematical language model to use as prior

    lib_path: str
        Path to token library of mathematical language model

    embedding_size: int
    num_layers: int
    num_hidden: int
        Model architecture of loaded mathematical language model

    prob_sharing: bool
        Share probabilities among terminal tokens?
    """

    def __init__(self, dso_library,
                model_path="./language_model/model/saved_model",
                lib_path="./language_model/model/saved_model/word_dict.pkl",
                embedding_size=32, num_layers=1, num_hidden=256,
                prob_sharing=True
                ):

        self.dso_n_input_var = len(dso_library.input_tokens)
        self.prob_sharing = prob_sharing

        with open(lib_path, 'rb') as f:
            self.lm_token2idx = pickle.load(f)
        self.dso2lm, self.lm2dso = self.set_lib_to_lib(dso_library)

        # Create TF2 Keras model
        self.language_model = LanguageModel(
            len(self.lm_token2idx),
            embedding_size,
            num_layers,
            num_hidden,
            mode='predict'
        )

        # Load model weights
        self.load_model(model_path)

        self.next_state = None
        self._zero_state = np.zeros(num_hidden, dtype=np.float32)

    def load_model(self, saved_language_model_path):
        """Load model weights from checkpoint."""
        # Try to load from TF2 checkpoint first
        checkpoint = tf.train.Checkpoint(model=self.language_model)
        latest_checkpoint = tf.train.latest_checkpoint(saved_language_model_path)

        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            print(f"Restored language model from {latest_checkpoint}")
        else:
            # Try legacy TF1 checkpoint format
            try:
                # Build the model first by running a dummy input
                dummy_input = tf.constant([[0]], dtype=tf.int32)
                _ = self.language_model(dummy_input)

                # Load legacy weights if available
                legacy_path = tf.train.latest_checkpoint(saved_language_model_path)
                if legacy_path:
                    # For TF1 checkpoints, we need to manually map variables
                    print(f"Attempting to load legacy checkpoint from {saved_language_model_path}")
                    reader = tf.train.load_checkpoint(saved_language_model_path)
                    var_to_shape_map = reader.get_variable_to_shape_map()
                    print(f"Found variables: {list(var_to_shape_map.keys())}")
            except Exception as e:
                print(f"Warning: Could not load language model checkpoint: {e}")
                print("Language model will use random weights.")

    def set_lib_to_lib(self, dso_library):
        """match token libraries of DSO and lm (LanguageModel)"""

        # dso token -> lm token
        dso2lm = [self.lm_token2idx['TERMINAL']] * self.dso_n_input_var
        dso2lm += [self.lm_token2idx[t.name.lower()] for t in dso_library.tokens if t.input_var is None]

        # lm token -> DSO token
        lm2dso = {lm_idx: i for i, lm_idx in enumerate(dso2lm)}

        return dso2lm, lm2dso

    def get_lm_prior(self, next_input):
        """return language model prior based on given current token"""

        # set feed_dict
        next_input = np.array(self.dso2lm)[next_input]  # match library with DSO
        next_input_tensor = tf.constant([next_input], dtype=tf.int32)

        if self.next_state is None:  # first input of a sequence
            # Initialize the cell state
            self.next_state = np.atleast_2d(self._zero_state)

        # Convert state to tensor
        initial_state = tf.constant(self.next_state, dtype=tf.float32)

        # Get language model output (TF2 eager execution)
        lm_logit, last_state = self.language_model(
            next_input_tensor,
            initial_state=initial_state,
            keep_prob=1.0,
            training=False
        )

        # Update state for next call
        if isinstance(last_state, (list, tuple)):
            self.next_state = last_state[0].numpy() if len(last_state) > 0 else last_state.numpy()
        else:
            self.next_state = last_state.numpy()

        # Convert logits to numpy
        lm_logit = lm_logit.numpy()

        if self.prob_sharing is True:
            # sharing probability among tokens in same group (e.g., TERMINAL to multiple variables)
            lm_logit[:, :, self.lm_token2idx['TERMINAL']] = lm_logit[:, :, self.lm_token2idx['TERMINAL']] - np.log(self.dso_n_input_var)

        lm_prior = lm_logit[0, :, self.dso2lm]
        lm_prior = np.transpose(lm_prior)  # make its shape to (batch size, dso size)

        return lm_prior

    def reset_state(self):
        """Reset the RNN state for a new sequence."""
        self.next_state = None
