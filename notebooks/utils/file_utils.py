import pandas as pd
from hdfs3 import HDFileSystem
hdfs = HDFileSystem(user='ebouille')
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import heapq
import pickle
import math
from datetime import datetime

def day_of_week(x):
    # Monday is 0 and Sunday is 6
    date_time = datetime.strptime(x, "%d.%m.%Y")
    return float(date_time.weekday())

def week_of_year(x):
    date_time = datetime.strptime(x, "%d.%m.%Y")
    return float(date_time.isocalendar()[1])

def route_desc_to_product(x):
    if x == 'Bus':
        return u'bus'
    elif x == 'Tram':
        return u'tram'
    elif x == 'Taxi':
        return u''
    else:
        return 'zug'

def read_parquet_from_hdfs(filename, username=None):
    username = username or "moiseev"
    files = hdfs.glob(f'/user/{username}/{filename}.parquet/*.parquet')
    df = pd.DataFrame()
    for file in files:
        with hdfs.open(file) as f:
            df = df.append(pd.read_parquet(f))
    return df

def trip_to_product_func(username=None):
    routes = read_parquet_from_hdfs("routes", username)
    trips = read_parquet_from_hdfs("trips", username)
    routes['product_id'] = routes['route_desc'].map(route_desc_to_product)
    route_to_product = {row['route_id']: row['product_id'] for _, row in routes.iterrows()}
    trip_id_to_route = {row['trip_id']: row['route_id'] for _, row in trips.iterrows()}
    trip_to_product = {trip: route_to_product[trip_id_to_route[trip]] \
                  for trip in trips['trip_id'].unique()}
    return trip_to_product

product_id_map = {
    u'': 0,
    u'bus': 1,
    u'tram': 2,
    u'zug': 3
}