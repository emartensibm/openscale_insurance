from flask import Flask, render_template, json, request, redirect, url_for
from flask_cors import CORS, cross_origin
from ibm_ai_openscale import APIClient
from ibm_ai_openscale.utils import get_instance_guid
from ibm_ai_openscale.supporting_classes.enums import FeedbackFormat
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
def claim(claim_id):
    connect_wos_client()

    explain_table = subscription.explainability.get_table_content()
    try:
        raw_data = explain_table.loc[explain_table.transaction_id == claim_id + '-1'].to_dict(orient="list")['explanation'][0]['entity']
        predictions = raw_data['predictions']
        feature_values = raw_data['input_features']
        contrastive = raw_data['contrastive_explanations']
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
                return render_template('claim.html', driver=driver, prediction=prediction, factors=factors, feature_values=feature_values, contrastive=contrastive, claim_id=claim_id)
        return "Claim ID not found"
    except IndexError:
        pass


@app.route('/store_feedback', methods=['POST'])
def store_feedback():
    global subscription
    print('Storing feedback for claim', request.form.get('claim_id'))
    fields = ['SUSPICIOUS_CLAIM_TIME','EXPIRED_LICENSE','LOW_MILES_AT_LOSS','EXCESSIVE_CLAIM_AMOUNT','TOO_MANY_CLAIMS','NO_POLICE','FLAG_FOR_FRAUD_INV']
    SUSPICIOUS_CLAIM_TIME = request.form.get('SUSPICIOUS_CLAIM_TIME')
    EXPIRED_LICENSE = request.form.get('EXPIRED_LICENSE')
    LOW_MILES_AT_LOSS = request.form.get('LOW_MILES_AT_LOSS')
    EXCESSIVE_CLAIM_AMOUNT = request.form.get('EXCESSIVE_CLAIM_AMOUNT')
    TOO_MANY_CLAIMS = request.form.get('TOO_MANY_CLAIMS')
    NO_POLICE = request.form.get('NO_POLICE')
    FLAG_FOR_FRAUD_INV = request.form.get('FLAG_FOR_FRAUD_INV')
    data = [SUSPICIOUS_CLAIM_TIME,EXPIRED_LICENSE,LOW_MILES_AT_LOSS,EXCESSIVE_CLAIM_AMOUNT,TOO_MANY_CLAIMS,NO_POLICE,FLAG_FOR_FRAUD_INV]
    subscription.feedback_logging.store([data], fields=fields, feedback_format=FeedbackFormat.WML)
    subscription.feedback_logging.show_table()
    return redirect(url_for('claim', claim_id=request.form.get('claim_id')))


driver_data = json.load(open(os.path.join(app.root_path, 'data.json')))

if __name__ == '__main__':
    PORT = 8080
    HOST = '0.0.0.0'
    print(connect_wos_client())
    app.run(host=HOST, port=PORT)