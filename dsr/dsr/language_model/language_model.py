"""
first input??
train

"""

# Load lang model

import os
import time

import argparse
import numpy as np
import tensorflow as tf
# from lm_utils import build_dataset_no_eos, batch_iter
from lm_utils import build_dataset_with_eos_padding, batch_iter


# from model.model_rnn import RNNLanguageModel
# from model.model_birnn import BiRNNLanguageModel
# from model.model_dyn_rnn_single import DynRNNLanguageModel
from model.model_dyn_rnn import DynRNNLanguageModel

# def log_print(str):
#     print(str)
#     with open('log.txt','a') as f:
#         f.write('{}\n'.format(str))




class LModel(object):
    def __init__(self, saved_model_path, lmodel_vocabulary_size, ):
        self.lmodel = DynRNNLanguageModel(lmodel_vocabulary_size, args, mode='predict')
        self.lsess = self.load_model(saved_model_path)
        self.next_state = None

        self.lib_dsr = {} # id to token
        self.lib_lmodel = {} # token to id

        # self.lib_dsr_to_lmodel = []

        # def _state_after_initial():

    def load_model(self, saved_model_path):
        sess = tf.compat.v1.Session()
        saver = tf.train.Saver()
        saver.restore(sess,tf.train.latest_checkpoint(saved_model_path))
        return sess

    def get_lm_prior(self, next_input):
        def _prep(next_input):
            """
            match library and return as lang model's input
            don't forget to handle terminals!
            """
            next_token = self.lib_dsr[next_input]
            next_x = self.lib_lmodel[next_token]
            return np.array([[next_x]])

        feed_dict = {self.lmodel.x: _prep(next_input), self.lmodel.keep_prob: 1.0}
        if self.next_state is not None: # not the first input
            feed_dict.update({self.lmodel.initial_state: self.next_state})

        self.next_state, lm_logit = self.lsess.run([self.lmodel.last_state, self.lmodel.logits], feed_dict=feed_dict)

        return lm_logit


def predict(saved_path, predict_data, vocabulary_size, args):
    with tf.compat.v1.Session() as sess:
        
        # if args.model == "rnn":
        #     model = RNNLanguageModel(vocabulary_size, args)
        # elif args.model == "birnn":
        #     model = BiRNNLanguageModel(vocabulary_size, args)
        # elif args.model == "dynrnn":
        #     model = DynRNNLanguageModel(vocabulary_size, args)
        # else:
        #     raise ValueError("Not Implemented {}.".format(args.model))

        model = DynRNNLanguageModel(vocabulary_size, args, mode='predict')


        # Load model
        saver = tf.train.Saver()
        saver.restore(sess,tf.train.latest_checkpoint(saved_path))

        test_batches = batch_iter(predict_data, args.batch_size, 1)

        for test_x in test_batches:
            # first
            feed_dict = {model.x: test_x[:,:1], model.keep_prob: 1.0}
            next_state, logits, lm_input, init_state = sess.run([model.last_state, model.logits, model.lm_input, model.initial_state], feed_dict=feed_dict)

            # second to last
            for i in range(1,2):
                feed_dict = {model.x: test_x[:,i:i+1], model.keep_prob: 1.0, model.initial_state: next_state}
                next_state, logits, lm_input, init_state = sess.run([model.last_state, model.logits, model.lm_input, model.initial_state], feed_dict=feed_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dynrnn", help="rnn | birnn")
    parser.add_argument("--embedding_size", type=int, default=32, help="embedding size.")
    parser.add_argument("--num_layers", type=int, default=1, help="RNN network depth.")
    parser.add_argument("--num_hidden", type=int, default=256, help="RNN network size.")
    parser.add_argument("--keep_prob", type=float, default=0.5, help="dropout keep prob.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate.")

    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--num_epochs", type=int, default=30, help="number of epochs.")

    parser.add_argument("--saved_path", type=str, default="/Users/kim102/OneDrive - LLNL/git/equation_language_model/results/dynrnn-(ep_10,la_1,kp_0.5,bs_64)-2001251717", help="trained model")
    args = parser.parse_args()


    # log_print(args)

    predict_data = np.array([[2,1], [2]])
    # train_file = "ptb_data/ptb.train.txt"
    # train_data, test_data = build_dataset(train_file, word_dict)

    # predict_, _, word_dict = build_dataset_no_eos(predict_input=predict_data, test_size=0.01)
    predict_, _, word_dict = build_dataset_with_eos_padding(predict_input=predict_data, test_size=0.01)
    
    # log_print(predict_.shape)


    predict(os.path.join(args.saved_path,'saved_model'), predict_, len(word_dict), args)
    
