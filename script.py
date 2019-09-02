import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request


app=Flask(__name__)
loaded_model = pickle.load(open("model.pkl","rb"))
@app.route('/')
@app.route('/index')
def index():
    return render_template('form_basico.html')



@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        result = request.form.to_dict()
        result = list(result.values())
        result = list(map(int,result))
        result = np.array(result).reshape(1,-1)     
        result = round(loaded_model.predict(result)[0],0)
    return render_template("resultado.html", prediction=result)

if __name__ == '__main__':
    app.run()
