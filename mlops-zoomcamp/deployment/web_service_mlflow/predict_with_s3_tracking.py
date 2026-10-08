# model flask endpoint 

from multiprocessing.pool import RUN
import pickle
from flask import Flask, request, jsonify
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import mlflow

# loads model for consumption from mlflow registry in s3
RUN_ID = 'c4070475175040c181ee4a94d713de99'
MODEL_ID = 'm-00f0e0c415624084bb7d514d7604a4cc'

# get model
model_pipeline_uri_from_s3 = f's3://ride-duration-mlflow-bucket/3/models/{MODEL_ID}/artifacts'
model = mlflow.sklearn.load_model(model_pipeline_uri_from_s3)

# get features | transform features
def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (\
        ride['PULocationID'], ride['DOLocationID']
    )
    features['trip_distance'] = ride['trip_distance']
    return features

# serve/expose prediction

def predict(features):
    # X = dv.transform(features)
    preds = model.predict(features)
    return float(preds[0])


app = Flask('duration-prediction') # dev server (use gunicorn for prod)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)




