import json

from flask import Flask, render_template, request

from plot import *

app = Flask(__name__)
main_plot = MainPlot()

@app.route('/', methods=['GET'])
def home():
    # bar_plot = create_plot()
    # return render_template('main.html', plot=bar_plot)
    return render_template('main.html')

  
@app.route('/data_points', methods=['POST'])
def data_points():
    data = request.data
    # validate data, if not valid, return warning
    
    return main_plot.scatter_data_points(data)

@app.route('/main_lines', methods=['POST'])
def main_lines():
    if main_plot.data_points == None:
        response = {
            'done': True,
            'plot': None,
            'warn': True # upload must be done ahead
        }
        return json.dumps(response)

    main_plot.step = request.json['step']
    return main_plot.plot_main_lines()


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blank')
def blank():
    return 'blank'

if __name__ == '__main__':
    app.run() # http://127.0.0.1:5000/