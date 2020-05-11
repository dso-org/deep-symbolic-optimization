import json

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from scipy.stats import gaussian_kde

from dsr.program import Program
from data import demo_utils

RESOLUTION = 200 # Number of points in KDE estimate

class MainPlot:
    def __init__(self):
        self.data_points = None
        self.data_range = [-1,1,0.5]
        
        self.done = False
        self.step = 0
        self.step_before = 0
        self.best_expr = None
        self.expr_info = {'expression': None, 'fitness': None}
        # self.diagnostics = []

        # Load offline data files
        with open("./data/traversals.txt", "r") as f:
            self.traversal_text = f.readlines()
        self.all_rewards = np.load("./data/dsr_Nguyen-5_0_all_r.npy")
        pd_all_tinfo = pd.read_csv("./data/dsr_Nguyen-5_0.csv")
        self.all_training_info = pd_all_tinfo.to_numpy()

    def scatter_data_points(self, data):
        self.data_points = data
        demo_utils.configure_program("./data/demo.json")

        data_x = Program.X_train.ravel()
        data_y = Program.y_train

        x_min = np.amin(data_x)
        x_max = np.amax(data_x)
        self.data_range = [x_min, x_max, x_max-x_min]

        scatter_data = [
            go.Scatter(
                x=data_x,
                y=data_y,
                mode='markers',
                marker=dict(
                    # color='#8c6bb1',
                    color='#1a76b3',
                    size=12,
                    line=dict(width=1.2,
                        color='#efedf5')
                        # color='#efedf5')
                )
            )
        ]

        graphJSON = json.dumps(scatter_data, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON

    def line_expression(self, expr_program):
        # Plot data points
        N = 1000
        x = np.linspace(self.data_range[0]-self.data_range[2]*0.1, self.data_range[1]+self.data_range[2]*0.1, N)
        y = expr_program.execute(x.reshape(N, -1))
        # df = pd.DataFrame({'x': x, 'y': y})
        self.expr_info['expression'] = repr(expr_program.sympy_expr)
        self.expr_info['fitness'] = expr_program.r

        line = [
            go.Scatter(
                x=x,
                # x=df['x'],
                y=y,
                # y=df['y'],
                mode='lines',
                name=self.expr_info['expression'],
                line=dict(color='#000', width=3)
                # line=dict(color='#3f007d', width=2.5)
            )
        ]

        graphJSON = json.dumps(line, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON

    def data_subplot(self):
        rng = range(self.step_before, self.step)
        best = self.all_training_info[rng,4] # base_r_best
        top = self.all_training_info[rng,7] # r_best
        mean = self.all_training_info[rng,6] # r_best
        reward = self.all_rewards[self.step] # Shape: (batch_size,)
        complexity = self.all_training_info[rng,12] # l_avg_full

        # Compute KDE for reward distribution
        kernel = gaussian_kde(reward, bw_method=0.25)
        reward_dist_x = np.linspace(0, 1, RESOLUTION)
        reward_dist_y = kernel(reward_dist_x)

        reward_dist_line = [
            go.Scatter(
                x=reward_dist_x,
                y=reward_dist_y,
                mode='lines',
                name=self.step,
                line=dict(color='#000', width=2)
            )
        ]
        graphJSON_reward_dist_line = json.dumps(reward_dist_line, cls=plotly.utils.PlotlyJSONEncoder)

        return {
            'subplot1': {
                'data': {
                    'x': [list(rng), list(rng), list(rng)], 
                    'y': [best.tolist(), top.tolist(), mean.tolist()]
                }
            },
            'subplot2': [{
                'reward': {
                    'data': {
                        'x': [reward_dist_x.tolist()],
                        'y': [reward_dist_y.tolist()],
                        'line': graphJSON_reward_dist_line
                    }
                }
            }],
            'subplot3': {
                'data': {
                    'x': [list(rng)], 
                    'y': [complexity.tolist()]
                }
            }
        }
        
    def plot_main_lines(self):
        try:
            if self.best_expr != self.traversal_text[self.step]:
                self.best_expr = self.traversal_text[self.step]
                best_p = demo_utils.make_program(self.traversal_text, self.step)

                response = {
                    'warn': False,
                    'done': self.done,
                    'plot': self.line_expression(best_p),
                    'info': self.expr_info,
                    'update': True
                }

            else:
                response = {
                    'warn': False,
                    'done': self.done,
                    'plot': None,
                    'update': False
                }
        except:
            self.done = True
            response = {
                'warn': False,
                'done': self.done,
                'plot': None,
                'update': False
            }

        response['subplot'] = self.data_subplot()
        self.step_before = self.step

        return json.dumps(response)