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

        self.language_model = LanguageModel(len(self.lm_token2idx), embedding_size, num_layers, num_hidden, mode='predict')
        self.lsess = self.load_model(model_path)
        self.next_state = None
        self._zero_state = np.zeros(num_hidden, dtype=np.float32)

    def load_model(self, saved_language_model_path):
        sess = tf.compat.v1.Session()
        saver = tf.train.Saver()
        saver.restore(sess,tf.train.latest_checkpoint(saved_language_model_path))
        return sess

    def set_lib_to_lib(self, dso_library):
        """match token libraries of DSO and lm (LanguageModel)"""

        # dso token -> lm token
        dso2lm = [self.lm_token2idx['TERMINAL']] * self.dso_n_input_var
        dso2lm += [self.lm_token2idx[t.name.lower()] for t in dso_library.tokens if t.input_var is None] # ex) [1,1,1,2,3,4,5,6,7,8,9], len(dso2lm) = len(library of dso)
        
        # lm token -> DSO token
        lm2dso = {lm_idx: i for i, lm_idx in enumerate(dso2lm)}

        # TODO: if DSO token missing in lm token library

        return dso2lm, lm2dso

    def get_lm_prior(self, next_input):
        """return language model prior based on given current token"""

        # set feed_dict
        next_input = np.array(self.dso2lm)[next_input]  # match library with DSO 
        next_input = np.array([next_input])

        if self.next_state is None: # first input of a sequence
            # For dynamic_rnn, not passing language_model.initial_state == passing zero_state.
            # Here, explicitly passing zero_state
            self.next_state = np.atleast_2d(self._zero_state) # initialize the cell
        
        feed_dict = {self.language_model.x: next_input, self.language_model.keep_prob: 1.0, self.language_model.initial_state: self.next_state}

        # get language model prior
        self.next_state, lm_logit = self.lsess.run([self.language_model.last_state, self.language_model.logits], feed_dict=feed_dict)
        
        if self.prob_sharing is True:
            # sharing probability among tokens in same group (e.g., TERMINAL to multiple variables)
            lm_logit[:, :, self.lm_token2idx['TERMINAL']] = lm_logit[:, :, self.lm_token2idx['TERMINAL']] - np.log(self.dso_n_input_var)
        lm_prior = lm_logit[0, :, self.dso2lm]
        lm_prior = np.transpose(lm_prior) # make its shape to (batch size, dso size)
        
        return lm_prior
