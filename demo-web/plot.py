import json

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from scipy.stats import gaussian_kde

from dsr.program import Program
from data import demo_utils

RESOLUTION = 50 # Number of points in KDE estimate


# def create_plot(): # example 

#     # Plot data points
#     N = 40
#     x = np.linspace(-2, 2, N)
#     y = np.exp(x)
#     df = pd.DataFrame({'x': x, 'y': y}) # creating a sample dataframe

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#             x=df['x'],
#             y=x,
#             mode='lines',
#             name='x',
#             line=dict(color='rgb(189,189,189)', width=2),
#             # hovertemplate = "%{label}"
#         ))
#     fig.add_trace(go.Scatter(
#             x=df['x'],
#             y=np.sin(x),
#             mode='lines',
#             name='sin',
#             line=dict(color='rgb(115,115,115)', width=2)
#         ))
#     fig.add_trace(go.Scatter(
#             x=df['x'],
#             y=2*np.sin(x),
#             mode='lines',
#             name='2sin',
#             line=dict(color='rgb(67,67,67)', width=2)
#         ))

#     annotations = []
#     # Title
#     annotations.append(dict(xref='paper', yref='paper', x=-0.05, y=1.1,
#                                 xanchor='left', yanchor='bottom',
#                                 text='Best expression: ',
#                                 font=dict(family='Arial',
#                                             size=15,
#                                             color='rgb(37,37,37)'),
#                                 showarrow=False))
    
#     fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
#     fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    
#     fig.update_layout(annotations=annotations)
#     # fig.update_layout(showlegend=False)
#     fig.update_layout(legend_orientation="h")
#     fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
#     fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')

#     # data = [
#     #     go.Scatter(
#     #         x=df['x'],
#     #         y=df['y'],
#     #         mode='lines',
#     #         name='exp'
#     #     )
#     # ]

#     graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

#     return graphJSON

class MainPlot:
    def __init__(self):
        self.data_points = None
        self.data_range = [-1,1,0.5]
        
        self.done = False
        self.step = 0
        self.step_before = 0
        self.best_expr = None
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
                    color='#8c6bb1',
                    size=7,
                    line=dict(width=1.2,
                        color='#efedf5')
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
        df = pd.DataFrame({'x': x, 'y': y})

        line = [
            go.Scatter(
                x=df['x'],
                y=df['y'],
                mode='lines',
                name=str(expr_program.sympy_expr),
                line=dict(color='#3f007d', width=2.5)
            )
        ]

        graphJSON = json.dumps(line, cls=plotly.utils.PlotlyJSONEncoder)

        return graphJSON

    def data_subplot(self):
        rng = range(self.step_before, self.step+1)
        # training = self.all_training_info[rng,0] # nmse
        training = self.all_training_info[rng,4] # r_best
        reward = self.all_rewards[self.step] # Shape: (batch_size,)

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
                line=dict(color='#000', width=1.3)
            )
        ]
        graphJSON_reward_dist_line = json.dumps(reward_dist_line, cls=plotly.utils.PlotlyJSONEncoder)

        return {
            'subplot1': [{
                'training': {
                    'data': {
                        'x': [list(rng)],
                        'y': [training.tolist()]
                    }
                }
            }],
            'subplot2': [{
                'reward': {
                    'data': {
                        'x': [reward_dist_x.tolist()],
                        'y': [reward_dist_y.tolist()],
                        'line': graphJSON_reward_dist_line
                    }
                }
            }]
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