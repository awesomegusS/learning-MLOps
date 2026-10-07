# model flask endpoint 

import pickle
from flask import Flask, request, jsonify
from mlflow.tracking import MlflowClient
import mlflow

# loads model for consumption from mlflow registry
RUN_ID = 'd452b3da58064fe2acb675aacfbd5d09'
TRACKING_URI = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

# get dictv
path = client.download_artifacts(run_id=RUN_ID, path='dictvectorizer/preprocessor.b') 
print(f'Downloading the model vectorizer to {path}')
with open(path, 'rb') as f_out:
    dv = pickle.load(f_out)

# get model
logged_model = f'runs:/{RUN_ID}/ridedurationxgb'
model = mlflow.pyfunc.load_model(logged_model)

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
    X = dv.transform(features)
    preds = model.predict(X)
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




