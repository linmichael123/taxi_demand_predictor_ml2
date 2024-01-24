from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR

def download_one_file_of_raw_data(year: int, month: int) -> Path:
    """"""
    URL = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    response = requests.get(URL)

    if response.status_code == 200:
        path = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        open(path, "wb").write(response.content)
        return path
    else:
        raise Exception(f'{URL} is not available')
    
def validate_raw_data(
        rides:pd.DataFrame,
        year: int,
        month: int,
) -> pd.DataFrame:
    
    this_month_start = f'{year}-{month:02d}-01'
    next_month_start = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
    rides = rides[rides.pickup_datetime >= this_month_start]
    rides = rides[rides.pickup_datetime < next_month_start]

    return rides

def load_raw_data(
        year:int,
        months: Optional[List[int]] = None
) -> pd.DataFrame:
    
    rides = pd.DataFrame()

    if months is None:
        months = list(range(1,13))
    elif isinstance(months, int):
        months = [months]

    for month in months:
        local_file = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        if not local_file.exists():
            try:
                print(f'Downloading file {year}-{month:02d}')
                download_one_file_of_raw_data(year,month)
            except:
                print(f'{year}-{month:02d} file is not available')
                continue
        else: 
            print(f'File {year}-{month:02d} was already in local storage')
        
        rides_one_month = pd.read_parquet(local_file)

        rides_one_month = rides_one_month[['tpep_pickup_datetime', 'PULocationID']]
        rides_one_month.rename(columns={
            'tpep_pickup_datetime': 'pickup_datetime',
            'PULocationID': 'pickup_location_id',
        }, inplace=True)

        rides_one_month = validate_raw_data(rides_one_month, year, month)

        rides = pd.concat([rides, rides_one_month])

    rides = rides[['pickup_datetime', 'pickup_location_id']]

    return rides

def add_missing_slots(
        rides: pd.DataFrame
) -> pd.DataFrame:
    
    location_ids = rides['pickup_location_id'].unique()
    full_range = pd.date_range(rides['pickup_hour'].min(),
                               rides['pickup_hour'].max(),
                               freq='h')
    output = pd.DataFrame()
    for location_id in tqdm(location_ids):

        rides_i = rides.loc[rides.pickup_location_id == location_id, ['pickup_hour','rides']]

        rides_i.set_index('pickup_hour', inplace=True)
        rides_i.index = pd.DatetimeIndex(rides_i.index)
        rides_i = rides_i.reindex(full_range, fill_value=0)

        rides_i['pickup_location_id'] = location_id

        output = pd.concat([output, rides_i])
    
    output = output.reset_index().rename(columns={'index': 'pickup_hour'})

    return output
            
def transform_raw_data_into_ts_data(
    rides: pd.DataFrame
) -> pd.DataFrame:
    
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('h')
    agg_rides = rides.groupby(['pickup_hour','pickup_location_id']).size().reset_index()
    agg_rides.rename(columns={0: 'rides'}, inplace=True)

    agg_rides_all_slots = add_missing_slots(agg_rides)

    return agg_rides_all_slots