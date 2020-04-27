from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blank')
def blank():
    return 'blank'

if __name__ == '__main__':
    app.run() # http://127.0.0.1:5000/