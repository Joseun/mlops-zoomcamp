#!/usr/bin/env python
# coding: utf-8

import click
import pickle
import pandas as pd
import numpy as np
from pathlib import Path


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']


def read_data(year: int, month: int) -> pd.DataFrame:
    print(f"Downloading Yellow NYC Taxi rides for {year:04d}/{month:02d}")
    df = pd.read_parquet(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')
    print("Downloaded")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

@click.command()
@click.option('--year', prompt='Year to download', help='Year to download', type=int)
@click.option('--month', prompt='Month to download', help='Month to download', type=int)
def predict(year: int, month: int) -> None:
    print("Reading data")
    df = read_data(year, month)

    print("Generating features")
    dicts = df[categorical].to_dict(orient='records')

    print("Vectoring")
    X_val = dv.transform(dicts)

    print("Predicting")
    y_pred = model.predict(X_val)

    df_result = pd.DataFrame()

    df_result['ride_id'] = df['ride_id']
    df_result['duration'] = y_pred

    print("The mean of the predicted duration for this dataset: ", np.average(y_pred))

    # save_file(year, month, df_result)
    return

def save_file(year: int, month: int, df_result: pd.DataFrame)-> None:
    output_file = Path(f"../data/yellow/{year:04d}-{month:02d}.parquet")
    output_file.parents[0].mkdir(parents=True, exist_ok=True)

    print(f"Saving predictions into {output_file}")

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    return

if __name__ == "__main__":
    predict()