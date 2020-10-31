# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 22:24:43 2020

@author: Admin
"""


from flask import Flask,request, url_for, redirect, render_template, jsonify,config
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model =  pickle.load(open('model.pkl','rb'))
cols = ['scaled_amount', 'scaled_time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
       'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16',
       'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26',
       'V27', 'V28']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = model.predict(data_unseen)
    prediction = int(prediction[0])
    if prediction==0:
        prediction1='Not Fraud'
    else:
        prediction1='Fraud'
    return render_template('home.html',pred='Transaction is Expected  to be {}'.format(prediction1))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = model.predict(data_unseen)
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)