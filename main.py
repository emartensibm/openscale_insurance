from flask import Flask, render_template, json, request
from flask_cors import CORS, cross_origin
from ibm_ai_openscale import APIClient
from ibm_ai_openscale.utils import get_instance_guid
import pandas as pd
import os


app = Flask(__name__)
CORS(app)
ai_client = None
subscription = None


def connect_wos_client():
    global ai_client
    global subscription
    try:
        version = ai_client.version
    except AttributeError:
        filename = os.path.join(app.root_path, 'credentials.json')
        cloud_api_key = json.load(open(filename))["CLOUD_API_KEY"]
        wos_guid = get_instance_guid(api_key=cloud_api_key)
        wos_credentials = {
            "instance_guid": wos_guid,
            "apikey": cloud_api_key,
            "url": "https://api.aiopenscale.cloud.ibm.com"
        }
        ai_client = APIClient(aios_credentials=wos_credentials)
        version = ai_client.version
        subscription = ai_client.data_mart.subscriptions.get(name='SKLearn Fraud Prediction')
    return version


def clean_factor_text(to_clean):
    cleaned = to_clean.replace("EXCESSIVE_CLAIM_AMOUNT", "Claim Amount")
    cleaned = cleaned.replace("EXPIRED_LICENSE", "License Status")
    cleaned = cleaned.replace("TOO_MANY_CLAIMS", "Number of Claims")
    cleaned = cleaned.replace("NO_POLICE", "Police Report Status")
    cleaned = cleaned.replace("SUSPICIOUS_CLAIM_TIME", "Time to Claim Filed")
    cleaned = cleaned.replace("LOW_MILES_AT_LOSS", "Miles at Loss")
    return cleaned


@app.route('/')
def index():
    return render_template('index.html', drivers=driver_data)


@app.route('/claim/<claim_id>')
def profile(claim_id):
    connect_wos_client()

    explain_table = subscription.explainability.get_table_content()
    try:
        raw_data = explain_table.loc[explain_table.transaction_id == claim_id + '-1'].to_dict(orient="list")['explanation'][0]['entity']
        predictions = raw_data['predictions']
        feature_values = raw_data['input_features']
        prediction = None
        for possibility in predictions:
            if "probability" in possibility and possibility["probability"] > 0.50:
                prediction = possibility
        prediction['probability'] = int(prediction['probability'] * 100)
        factors = []
        for factor in prediction['explanation']:
            if factor['weight'] > 0:
                factors.append({'name': clean_factor_text(factor['feature_name']), 'weight': int(factor['weight'] * 100)})

        for driver in driver_data:
            if driver["claim_id"] == claim_id:
                return render_template('claim.html', driver=driver, prediction=prediction, factors=factors, feature_values=feature_values)
        return "Claim ID not found"
    except IndexError:
        pass


@app.route('/store_feedback_positive', methods=['POST'])
def store_feedback_positive():
    print(request.form)
    pass


@app.route('/store_feedback_negative', methods=['POST'])
def store_feedback_negative():
    print(request.form)
    pass


driver_data = json.load(open(os.path.join(app.root_path, 'data.json')))

if __name__ == '__main__':
    PORT = 8080
    HOST = '0.0.0.0'
    print(connect_wos_client())
    app.run(host=HOST, port=PORT)