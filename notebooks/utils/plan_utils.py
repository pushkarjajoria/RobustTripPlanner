from .delay_utils import *
from .file_utils import *
from .graph_utils import *

def get_sort_key(route):
    trip_ids = set()
    n_walk = 0
    for edge in route[0][1]:
        if edge["trip_id"] == "walk":
            n_walk += 1
        else:
            trip_ids.add(edge["trip_id"])
    return (n_walk, len(trip_ids))

def sort_output(routes):
    return sorted(routes, key=get_sort_key)

def plan_travel(gt, origin, destination, arrival_time, date, n_routes=1, confidence_threshold=None, fast=True, 
                scaler=None, model=None, product_id_map=None, trip_to_product=None):
    if confidence_threshold and fast:
        if confidence_threshold < 0.8:
            wait_time = 1
        elif confidence_threshold < 0.9:
            wait_time = 2
        elif confidence_threshold < 0.95:
            wait_time = 3
        else:
            wait_time = 4
        threshold = None
    else:
        wait_time = 0
        threshold = confidence_threshold
        
    def _plan_travel(gt, origin, destination, arrival_time, date, n_routes, confidence_threshold, threshold, wait_time): 
        routes, _ = gt.get_k_shortest_path(origin, destination, None, arrival_time, int(day_of_week(date)) + 1, week_of_year(date), 
                                           threshold, wait_time, n_routes)
        if model:
            probs = [get_success_probability(route[1], route[0], day_of_week(date), week_of_year(date), scaler, model, product_id_map, trip_to_product)
                     for route in routes]
            routes = [(route, p) for route, p in zip(routes, probs) if confidence_threshold is None or p >= confidence_threshold]
            if len(routes) == n_routes:
                return routes
            else:
                return routes + _plan_travel(gt, origin, destination, arrival_time, date, n_routes - len(routes), 
                                             confidence_threshold, threshold, wait_time + 1)
                
        else:
            return [(route, None) for route in routes]
            
    return sort_output(_plan_travel(gt, origin, destination, arrival_time, date, n_routes, confidence_threshold, threshold, wait_time))
