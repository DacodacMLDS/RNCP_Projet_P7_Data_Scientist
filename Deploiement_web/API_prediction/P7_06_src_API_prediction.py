#Pour exécuter :
#en mode local :dans le dossier du fichier app.py faire python app.py 
# puis dans le navigateur aller à http://127.0.0.1:5000//predict?id_client=101908

# Import all packages and libraries
import pandas as pd
import numpy as np
from flask import Flask, render_template, url_for, request
import pickle
import math
import base64
from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv('data_predict_api.csv', sep=",")
list_id_client = list(data['SK_ID_CURR'].unique())

model = pickle.load(open('trained_gbc_model.pkl', 'rb'))
seuil = 0.52

app= Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods = ['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    ID = request.form['id_client']
    ID = int(ID)
    if ID not in list_id_client:
        prediction="Ce client n'est pas répertorié"
    else :
        X = data[data['SK_ID_CURR'] == ID]
        X = X.drop(['SK_ID_CURR'], axis=1)

        probability_default_payment = model.predict_proba(X)[:, 1]
        if probability_default_payment >= seuil:
            prediction = "Client défaillant: prêt refusé"
        else:
            prediction = "Client non défaillant:  prêt accordé"

    return render_template('index.html', prediction_html=prediction)

# Define endpoint for flask
app.add_url_rule('/predict', 'predict', predict)