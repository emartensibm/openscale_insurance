from flask import Flask, render_template, json, request, redirect, url_for
from flask_cors import CORS, cross_origin
from ibm_watson_openscale import APIClient
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson_openscale.supporting_classes.enums import *
from ibm_watson_openscale.supporting_classes import *
from time import sleep

# from ibm_ai_openscale import APIClient
# from ibm_ai_openscale.utils import get_instance_guid
# from ibm_ai_openscale.supporting_classes.enums import FeedbackFormat
import pandas as pd
import os


# Global variables for our web app. the ai_client and subscription variables will be used to connect to our
# Watson OpenScale instance and access the Python APIs.
app = Flask(__name__)
CORS(app)
ai_client = None
subscription = None
subscription_id = None
payload_data_set_id = None
feedback_data_set_id = None

def connect_wos_client():
    # Use the provided Cloud API key from the credentials.json file to connect to Watson OpenScale.
    global ai_client
    global subscription
    global subscription_id
    global payload_data_set_id
    global feedback_data_set_id
    try:
        version = ai_client.version
    except AttributeError:
        filename = os.path.join(app.root_path, 'credentials.json')
        cloud_api_key = json.load(open(filename))["CLOUD_API_KEY"]

        service_credentials = {
            "apikey": cloud_api_key,
            "url": "https://api.aiopenscale.cloud.ibm.com"
        }

        authenticator = IAMAuthenticator(apikey=service_credentials['apikey'])
        ai_client = APIClient(authenticator=authenticator)

        version = ai_client.version
        # Get the subscription for our specific Fraud Prediction model.
        subscriptions = ai_client.subscriptions.list().result.to_dict()['subscriptions']
        for sub in subscriptions:
            if sub['entity']['asset']['name'] == 'SKLearn Fraud Prediction':
                subscription_id = sub['metadata']['id']
        print('Getting subscription', subscription_id)
        subscription = ai_client.subscriptions.get(subscription_id)
        payload_data_set_id = ai_client.data_sets.list(type=DataSetTypes.PAYLOAD_LOGGING,
                                                       target_target_id=subscription_id,
                                                       target_target_type=TargetTypes.SUBSCRIPTION).result.data_sets[0].metadata.id
        feedback_data_set_id = ai_client.data_sets.list(type=DataSetTypes.FEEDBACK,
                                                       target_target_id=subscription_id,
                                                       target_target_type=TargetTypes.SUBSCRIPTION).result.data_sets[0].metadata.id
    return version


def clean_factor_text(to_clean):
    # Make the list of model features more human-readable.
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
    # Ensure that we have a valid client and subscription.
    connect_wos_client()
    records = ai_client.data_sets.get_list_of_records(data_set_id=payload_data_set_id, offset=0).result
    scoring_id = None
    for record in records['records']:
        if record["entity"]["values"]["customer_id"] == claim_id:
            scoring_id = record["entity"]["values"]["scoring_id"]
            break
    explanation_types = ["lime", "contrastive"]
    explain_task_result = ai_client.monitor_instances.explanation_tasks(scoring_ids=[scoring_id],
                                                                         explanation_types=explanation_types).result.to_dict()
    explain_task_id = explain_task_result['metadata']['explanation_task_ids'][0]
    task_state = 'in_progress'
    explanation_raw = None
    while task_state == 'in_progress':
        explanation_raw = ai_client.monitor_instances.get_explanation_tasks(explain_task_id).result.to_dict()
        task_state = explanation_raw['entity']['status']['state']
        if task_state == 'finished':
            break
        sleep(1)

    lime_explanation = None
    contrastive_explanation = None
    for explanation in explanation_raw['entity']['explanations']:
        if explanation['explanation_type'] == 'lime':
            lime_explanation = explanation
        elif explanation['explanation_type'] == 'contrastive':
            contrastive_explanation = explanation

    try:
        predictions = lime_explanation['predictions']
        prediction = None
        for possibility in predictions:
            if "probability" in possibility and possibility["probability"] > 0.50:
                prediction = possibility
                feature_values = possibility['explanation_features']
        prediction['probability'] = int(prediction['probability'] * 100)
        factors = []
        for factor in prediction['explanation_features']:
            if factor['weight'] > 0:
                factors.append({'name': clean_factor_text(factor['feature_name']), 'weight': int(factor['weight'] * 100)})

        # Get the other relevant claim, driver, weather, and location data from our data.json file.
        # In a production application, this would come from one or more database queries or RESTful API calls.
        # For simplicity in setting up the demo, we will use a static json file.
        for driver in driver_data:
            if driver["claim_id"] == claim_id:
                return render_template('claim.html', driver=driver, prediction=prediction, factors=factors, feature_values=feature_values, contrastive=contrastive_explanation, claim_id=claim_id)
        return "Claim ID not found"
    except IndexError:
        return "No explanation available"


@app.route('/store_feedback', methods=['POST'])
def store_feedback():
    # Use the OpenScale Python API to store feedback data so we can score our model for accuracy, and improve it
    # over time.
    global subscription
    global feedback_data_set_id
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

    feedback_payload = [{
        "fields": fields,
        "values": [data]
    }]
    print(feedback_payload)
    ai_client.data_sets.store_records(feedback_data_set_id, request_body=feedback_payload)
    # subscription.feedback_logging.store([data], fields=fields, feedback_format=FeedbackFormat.WML)
    # subscription.feedback_logging.show_table()
    return redirect(url_for('claim', claim_id=request.form.get('claim_id')))


# Get the static driver, policy, claim and other data from our json file.
driver_data = json.load(open(os.path.join(app.root_path, 'data.json')))

if __name__ == '__main__':
    PORT = 8080
    HOST = '0.0.0.0'
    print(connect_wos_client())
    app.run(host=HOST, port=PORT)