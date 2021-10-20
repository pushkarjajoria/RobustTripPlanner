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
import networkx as nx
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import math
import scipy.stats as sps

def read_scaler_from_hdfs(filename, username=None):
    delay_username = username or "moiseev"
    files = hdfs.glob(f'/user/{delay_username}/{filename}/data/*.parquet')
    df = pd.DataFrame()
    for file in files:
        with hdfs.open(file) as f:
            df = df.append(pd.read_parquet(f))
    
    scaler = StandardScaler()
    scaler.mean_ = df['mean'][0]['values']
    scaler.scale_ = np.maximum(df['std'][0]['values'], 1e-7)
    
    return scaler

def read_linear_regression_from_hdfs(filename, username=None):
    delay_username = username or "moiseev"
    files = hdfs.glob(f'/user/{delay_username}/{filename}/data/*.parquet')
    df = pd.DataFrame()
    for file in files:
        with hdfs.open(file) as f:
            df = df.append(pd.read_parquet(f))
    
    lr = LinearRegression()
    lr.intercept_ = df['intercept'][0]
    lr.coef_ = df['coefficients'][0]['values']
    
    return lr

def read_logistic_regression_from_hdfs(filename, username=None):
    delay_username = username or "moiseev"
    files = hdfs.glob(f'/user/{delay_username}/{filename}/data/*.parquet')
    df = pd.DataFrame()
    for file in files:
        with hdfs.open(file) as f:
            df = df.append(pd.read_parquet(f))
    
    log = LogisticRegression()
    log.classes_ = np.arange(df['numClasses'][0])
    log.intercept_ = df['interceptVector'][0]['values']
    log.coef_ = df['coefficientMatrix'][0]['values'].reshape(len(log.classes_), -1)   
    return log

def create_features(minute_of_day, day_of_week, week_of_year, product_id, stop_id, product_id_map, scaler):
    product_oh = [0] * 4
    product_oh[product_id_map[product_id]] = 1
    stop_oh = [0] * (scaler.mean_.shape[0] - 9)
    stop_oh[stop_id % len(stop_oh)] = 1
    features = np.array([minute_of_day, (24 * 30 - minute_of_day) ** 2, 
                         np.abs(24 * 30 - minute_of_day)] + product_oh + stop_oh + [week_of_year, day_of_week])
    return scaler.transform(features.reshape(1, -1))[0]

def predict_delay_probability(delay, minute_of_day, day_of_week, week_of_year,
                              product_id, stop_id, product_id_map, model, scaler):
    """Predicts probability that delay will be less or equal than requested value.
    
    So it allows to predict probability of success.
    """
    #print(delay, minute_of_day, day_of_week, week_of_year, product_id, stop_id)
    features = create_features(minute_of_day, day_of_week, week_of_year, product_id, stop_id, product_id_map, scaler)
    
    if isinstance(model, LinearRegression):
        delay_pred = model.predict(features.reshape(1, -1))[0]
        return sps.expon.cdf(delay, scale=1/delay_pred)
        
    elif isinstance(model, LogisticRegression):
        probs = model.predict_proba(features.reshape(1, -1))[0]
        bn = int(math.floor(delay))
        return probs[:bn + 1].sum()
    
def string_time_to_minutes(t):
    (hour_x, min_x) = tuple(t.split(':'))
    return 60*(int(hour_x)) + int(min_x)
    
def get_success_probability(edges, nodes, day_of_week, week_of_year, scaler, log, product_id_map, trip_to_product):
    prob = 1.
    for i, (prev, cur, node) in enumerate(zip(edges[:-1], edges[1:], nodes[1:])):
        if prev['trip_id'] == 'walk' or prev['trip_id'] == cur['trip_id']:
            continue
           
        if cur['trip_id'] == 'walk':
            if i + 2 < len(edges):
                nxt = edges[i + 2]
                max_delay = string_time_to_minutes(nxt['dep']) - string_time_to_minutes(prev['arr']) - cur['walking_time']
            else:
                continue
        else:
            max_delay = string_time_to_minutes(cur['dep']) - string_time_to_minutes(prev['arr']) - 2
            
        minute_of_day = string_time_to_minutes(prev['arr'])
            
        prob *= predict_delay_probability(max_delay, minute_of_day, day_of_week, week_of_year, 
                                          trip_to_product[prev['trip_id']], node, product_id_map, log, scaler)
    return prob
