from datetime import datetime, timedelta

import hopsworks
from hsfs.feature_store import FeatureStore
import pandas as pd
import numpy as np

import src.config as config

def get_hopsworks_project() -> hopsworks.project.Project:

    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )

def get_feature_store() -> FeatureStore:
    
    project = get_hopsworks_project()
    return project.get_feature_store()

def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:

    predictions = model.predict(features)

    results = pd.DataFrame()
    results['pickup_location_id'] = features['pickup_location_id'].values
    results['predicted_demand'] = predictions.round(0)

    return results

def load_batch_of_features_from_store(
        current_date: pd.Timestamp,
) -> pd.DataFrame:
    
    feature_store = get_feature_store()

    n_features = config.N_FEATURES

    fetch_data_to = current_date - timedelta(hours=1) #an hour ago
    fetch_data_from = current_date - timedelta(days=28) #28 days ago    
    print(f'Fetching data from {fetch_data_from} to {fetch_data_to}')
    
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME,
        version=config.FEATURE_VIEW_VERSION
    )

    ts_data = feature_view.get_batch_data(
        start_time= (fetch_data_from - timedelta(days=1)),
        end_time = (fetch_data_to + timedelta(days=1))
    )

    pickup_ts_from = int(fetch_data_from.timestamp() * 1000)
    pickup_ts_to = int(fetch_data_to.timestamp() * 1000)
    ts_data = ts_data[ts_data.pickup_ts.between(pickup_ts_from, pickup_ts_to)]

    # ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]

    location_ids = ts_data['pickup_location_id'].unique()
    assert len(ts_data) == n_features*len(location_ids), "Time-series data is not complete"

    ts_data.sort_values(by=['pickup_location_id','pickup_hour'], inplace=True)
    print(f'{ts_data=}')

    x = np.ndarray(shape=(len(location_ids),n_features), dtypes=np.float32)
    for i, location_ids in enumerate(location_ids):
        ts_data_i = ts_data.loc[ts_data.pickup_location_id == location_ids, :]
        ts_data_i = ts_data_i.sort_values(by=['pickup_hour'])
        x[i,:] = ts_data_i['rides'].values

    features = pd.DataFrame(
        x,
        columns=[f'ride_previous_{i+1}_hour' for i in reversed(range(n_features))]
    )
    features['pickup_hour'] = current_date
    features['pickup_location_id'] = location_ids

    return features


def load_model_registry():

    import joblib
    from pathlib import Path

    project = get_hopsworks_project()
    model_registry = project.get_model_registry()

    model = model_registry.get_model(
        name = config.MODEL_NAME,
        version = config.MODEL_VERSION,
    )

    model_dir = model.download()
    model = joblib.load(Path(model_dir) / 'model.pkl')

    return model