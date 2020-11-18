import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from dsr.controller import Controller
from dsr.program import Program
from dsr.train import learn


class Dataset():
    def __init__(self, X, y):
        self.X_train = X
        self.y_train = y
        self.X_test = X
        self.y_test = y
        self.y_train_noiseless = y
        self.y_test_noiseless = y
        self.var_y_test = np.var(y)


class DSRRegressor(BaseEstimator, RegressorMixin):
    ''' DSR as scikit-learn regressor '''

    def __init__(self, n_samples=1000000, batch_size=500, verbose=True):
        self.config_controller = {'cell': 'lstm', 'num_layers': 1, 'num_units': 32, 'initializer': 'zeros',
                                  'embedding': False, 'embedding_size': 8, 'optimizer': 'adam', 'learning_rate': 0.001,
                                  'observe_action': False, 'observe_parent': True, 'observe_sibling': True,
                                  'constrain_const': True, 'constrain_trig': True, 'constrain_inv': True,
                                  'constrain_min_len': True, 'constrain_max_len': True, 'constrain_num_const': False,
                                  'min_length': 4, 'max_length': 30, 'max_const': 3, 'entropy_weight': 0.1,
                                  'ppo': False, 'ppo_clip_ratio': 0.2, 'ppo_n_iters': 10, 'ppo_n_mb': 4, 'pqt': False,
                                  'pqt_k': 10, 'pqt_batch_size': 1, 'pqt_weight': 200.0, 'pqt_use_pg': False}
        self.config_training = {'logdir': './log', 'n_epochs': None, 'n_samples': n_samples,
                                'batch_size': batch_size, 'reward': 'inv_nrmse', 'reward_params': [],
                                'complexity': 'length',
                                'complexity_weight': 0.0, 'const_optimizer': 'scipy', 'const_params': {}, 'alpha': 0.5,
                                'epsilon': 0.1, 'verbose': verbose, 'baseline': 'R_e', 'b_jumpstart': False,
                                'num_cores': 1, 'summary': False, 'debug': 0, 'output_file': 'dsr_Nguyen-1_0.csv',
                                'save_all_r': False, 'early_stopping': True, 'threshold': 1e-12}

    def fit(self, X, y):
        # Define the dataset and library
        dataset = Dataset(X, y)
        Program.clear_cache()
        Program.set_training_data(dataset)
        Program.set_library(['add', 'sub', 'mul', 'div', 'sin', 'cos', 'exp', 'log'], X.shape[1])

        tf.reset_default_graph()

        # Shift actual seed by checksum to ensure it's different across different benchmarks
        tf.set_random_seed(0)

        with tf.Session() as sess:
            # Instantiate the controller
            controller = Controller(sess, debug=False, summary=False,
                                    **self.config_controller)

            # Train the controller
            result: Program = learn(sess, controller, **self.config_training,
                                    return_estimator=True)  # r, base_r, expression, traversal
        self.result = result

    def predict(self, X, y=None):
        return self.result.execute(X)


if __name__ == '__main__':
    # load data
    origin_data = load_boston()
    data = pd.DataFrame(origin_data.data, columns=origin_data.feature_names)
    data['MEDV'] = pd.Series(origin_data.target)

    # process outliers
    data = data[~(data['MEDV'] >= 50.0)]

    # select important variable
    column_name = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
    x = np.array(data.loc[:, column_name])
    y = np.array(data['MEDV'])

    # train DSR
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    dsr = DSRRegressor(n_samples=10, batch_size=1)
    dsr.fit(x_train, y_train)

    # validation
    print(mean_squared_error(y_train, dsr.predict(x_train)))
    print(mean_squared_error(y_test, dsr.predict(x_test)))
