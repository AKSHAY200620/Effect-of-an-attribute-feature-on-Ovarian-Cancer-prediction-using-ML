from flask import Flask, render_template, request
import pandas as pd
from project_internship import run_models

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = {}
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.xlsx'):
            data = pd.read_excel(file)
            results = run_models(data)
            return render_template('index.html', results=results)
        else:
            return render_template('error.html')  # ⬅ Render error page here
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
