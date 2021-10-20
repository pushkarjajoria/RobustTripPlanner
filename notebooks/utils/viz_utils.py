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
import plotly.graph_objects as go

def format_time(time):
    return "{0}:{1}".format(str(time).split(":")[0], str(time).split(":")[1].zfill(2))#.zfill(2)

def get_zoom(lons, lats):
    return np.clip(1/max(np.max(lons) - np.min(lons), np.max(lats) - np.min(lats)), 10, 14)

def add_path(fig, nodes, edges, trips, routes, stops, visible): 
    
    trip_ids = set()
    for e in edges:
        if e['trip_id'] != 'walk':
            trip_ids.add(e['trip_id'])
    trip_ids = list(trip_ids)

    trip_to_route = trips\
        .query("trip_id in {}".format(trip_ids))\
        .merge(routes, on='route_id', how='left')\
        [['trip_id', 'route_desc', 'route_short_name']]\
        .set_index('trip_id')\
        .apply (lambda row: row['route_desc']+' ' +row['route_short_name'], axis=1)
    trip_to_route = {trip_id: trip_to_route[trip_id] for trip_id in trip_to_route.index}
    
    stops_plt = stops.query("stop_id in {}".format(nodes)).set_index('stop_id')
    plt_lat = []
    plt_lon = []
    plt_name = []
    for n in nodes:
        s = stops_plt.loc[n]
        plt_lat.append(s['stop_lat'])
        plt_lon.append(s['stop_lon'])
        plt_name.append(s['stop_name'])
        
    trip_to_route['walk'] = "Walking"

    change_lat = [plt_lat[0]]
    change_lon = [plt_lon[0]]
    change_name = [plt_name[0]]

    curr_lat = [plt_lat[0]]
    curr_lon = [plt_lon[0]]
    curr_trip_id = edges[0]['trip_id']

    dep_time = edges[0]['dep']

    for e in range(len(edges)):
        if(edges[e]['trip_id']!=curr_trip_id):
            fig.add_trace(go.Scattermapbox(
                mode = "lines",
                lon = curr_lon,
                lat = curr_lat,
                hoverinfo='none',
                name = trip_to_route[curr_trip_id] + '\n Departure:'+ format_time(dep_time) + '\n Arrival:'+ format_time(edges[e-1]['arr']),
                line = dict(width=4),
                visible = visible
            ))
            dep_time = edges[e]['dep']
            curr_trip_id = edges[e]['trip_id']
            curr_lat = [plt_lat[e], plt_lat[e+1]]
            curr_lon = [plt_lon[e], plt_lon[e+1]]
            change_lat.append(plt_lat[e])
            change_lon.append(plt_lon[e])
            change_name.append(plt_name[e])
        else:
            curr_lat.append(plt_lat[e+1])
            curr_lon.append(plt_lon[e+1])
    fig.add_trace(go.Scattermapbox(
        mode = "lines",
        lon = curr_lon,
        lat = curr_lat,
        hoverinfo='none',
        name = trip_to_route[curr_trip_id] + '\n Departure:'+ format_time(dep_time) + '\n Arrival:'+ format_time(edges[-1]['arr']),
        line = dict(width=4),
        visible = visible
    ))
    change_lat.append(plt_lat[-1])
    change_lon.append(plt_lon[-1])
    change_name.append(plt_name[-1])

    fig.add_trace(go.Scattermapbox(
        mode = "markers",
        lon = change_lon,
        lat = change_lat,
        text = change_name,
        marker=go.scattermapbox.Marker(
                size=17,
                color='rgb(255, 0, 0)',
                opacity=1
            ),
        hoverinfo='text',
        name = 'Transfer stations',
        visible = visible
    ))

    fig.add_trace(go.Scattermapbox(
        mode = "markers",
        lon = plt_lon,
        lat = plt_lat,
        text = plt_name,
        marker=go.scattermapbox.Marker(
                size=8,
                color='rgb(255, 102, 102)',
                opacity=1
            ),
        hoverinfo='text',
        name = 'Intermediate stations',
        visible = visible
    ))

    fig.update_layout(
        margin ={'l':0,'t':0,'b':0,'r':0},
        mapbox = {
            'center': {'lon': np.mean(plt_lon), 'lat': np.mean(plt_lat)},
            'style': "open-street-map",
            'zoom': get_zoom(plt_lon, plt_lat)})
    
    return fig

def visualize(A, trips, routes, stops):
    fig = go.Figure()
    visibles = []
    prev_length = 0
    
    for i, ((nodes, edges), _) in enumerate(A):
        add_path(fig, nodes, edges, trips, routes, stops, i == 0)
        visibles.append([False] * prev_length + [True] * (len(fig.data) - prev_length))
        prev_length = len(fig.data)
    
    max_length = len(fig.data)
    for v in visibles:
        v += [False] * (max_length - len(v))
    
    initial_p = A[0][1]
    fig.update_layout(
        title = f"Path 1, confidence {initial_p:.3f}",
    )
    
    buttons = []
    for i, (v, (_, p)) in enumerate(zip(visibles, A)):
        buttons.append(dict(
            label=f"Path {i + 1}",
            method="update",
            args=[{"visible": v}, {"title": f"Path {i + 1}, confidence {p:.3f}"}]))
        
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                x=0.57,
                y=1.2,
                buttons=buttons,
            )
        ])
    return fig
