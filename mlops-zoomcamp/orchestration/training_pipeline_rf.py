#!/usr/bin/env python
# coding: utf-8

import pickle
from pathlib import Path

import pandas as pd


from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

import mlflow
from pathlib import Path

# mlflow.set_tracking_uri('sqlite:///../mlflow.db')
mlflow.set_tracking_uri('http://127.0.0.1:5000')

mlflow.set_experiment('ride-duration-prediction-xgboost')

models_folder = Path('../models')
models_folder.mkdir(exist_ok=True)



def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df

def prepare_dictionaries(df: pd.DataFrame):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts


def train_model(dict_train, y_train, dict_val, y_val):
    with mlflow.start_run() as run:
        params = dict(max_depth=20, n_estimators=100, min_samples_leaf=10, random_state=0)
        mlflow.log_params(params)

        pipeline = make_pipeline(
        DictVectorizer(),
        RandomForestRegressor(**params, n_jobs=-1)
    )

        pipeline.fit(dict_train, y_train)
        y_pred = pipeline.predict(dict_val)

        rmse = root_mean_squared_error(y_pred, y_val)
        print(params, rmse)
        mlflow.log_metric('rmse', rmse)

        mlflow.sklearn.log_model(pipeline, artifact_path="model", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,)

        return run.info.run_id


def run(year, month):
    df_train = read_dataframe(year=year, month=month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1

    df_val = read_dataframe(year=next_year, month=next_month)

    dict_train = prepare_dictionaries(df_train)
    dict_val = prepare_dictionaries(df_val)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    run_id = train_model(dict_train, y_train, dict_val, y_val)
    print(f"MLflow run_id: {run_id}")
    return run_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a model to predict taxi trip duration.')
    parser.add_argument('--year', type=int, required=True, help='Year of the data to train on')
    parser.add_argument('--month', type=int, required=True, help='Month of the data to train on')
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)

    with open("./run_id.txt", "w") as f:
        f.write(run_id)