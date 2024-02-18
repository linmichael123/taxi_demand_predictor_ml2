import os
from dotenv import load_dotenv
from src.paths import PARENT_DIR
from src.feature_store_api import FeatureGroupConfig, FeatureViewConfig

load_dotenv(PARENT_DIR / '.env')

HOPSWORKS_PROJECT_NAME = "taxi_demand_ml"

try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception('Create an .env file on the project root with the HOPSWORKS_API_KEY')

FEATURE_GROUP_NAME = 'time_series_hourly_feature_group'
FEATURE_GROUP_VERSION = 2
FEATURE_GROUP_METADATA = FeatureGroupConfig(
    name='time_series_hourly_feature_group',
    version=2,
    description='Feature group with hourly time-series data of historical taxi rides',
    primary_key=['pickup_location_id', 'pickup_ts'],
    event_time='pickup_ts',
    online_enabled=True,
)

FEATURE_VIEW_NAME = 'time_series_hourly_feature_view'
FEATURE_VIEW_VERSION = 2
FEATURE_VIEW_METADATA = FeatureViewConfig(
    name='time_series_hourly_feature_view',
    version=2,
    feature_group=FEATURE_GROUP_METADATA,
)


MODEL_NAME = "taxi_demand_predictor_next_hour"
MODEL_VERSION = 3

FEATURE_GROUP_MODEL_PREDICTIONS = 'model_predictions_feature_group'
FEATURE_GROUP_PREDICTIONS_METADATA = FeatureGroupConfig(
    name='model_predictions_feature_group',
    version=1,
    description="Predictions generate by our production model",
    primary_key = ['pickup_location_id', 'pickup_ts'],
    event_time='pickup_ts',
)


FEATURE_VIEW_MODEL_PREDICTIONS = 'model_predictions_feature_view'
FEATURE_VIEW_PREDICTIONS_METADATA = FeatureViewConfig(
    name='model_predictions_feature_view',
    version=1,
    feature_group=FEATURE_GROUP_PREDICTIONS_METADATA,
)

N_FEATURES = 24*28

