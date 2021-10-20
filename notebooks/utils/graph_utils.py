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
import copy
from tqdm.notebook import tqdm
import random

from .delay_utils import predict_delay_probability

# Constants
OUT_OF_BOUNDS = "-1"
ORIGIN = "origin"
DESTINATION = "destination"
TRIP_ID = 'trip_id'
ARRIVAL = 'arr'
DEPARTURE = 'dep'
WALKING_TRIP = 'walk'
WALKING_TIME = 'walking_time'
SCHEDULE = 'weekly'

# Utility function
def time_gte(x, y):
    (hour_x, min_x) = tuple(x.split(':'))
    (hour_y, min_y) = tuple(y.split(':'))
    return (int(hour_x) > int(hour_y)) or (int(hour_x) == int(hour_y) and int(min_x) >= int(min_y))


def time_lte(x, y):
    if y is None:
        return True
    (hour_x, min_x) = tuple(x.split(':'))
    (hour_y, min_y) = tuple(y.split(':'))
    return (int(hour_x) < int(hour_y)) or (int(hour_x) == int(hour_y) and int(min_x) <= int(min_y))


def subtract_time(x, y):
    assert time_gte(x, y), "x cannot be smaller than y"
    (hour_x, min_x) = tuple(x.split(':'))
    (hour_y, min_y) = tuple(y.split(':'))
    return 60*(int(hour_x) - int(hour_y)) + int(min_x) - int(min_y)


def string_time_to_minutes(t):
    (hour_x, min_x) = tuple(t.split(':'))
    return 60*(int(hour_x)) + int(min_x)


def add_minutes(t: str, minutes: int):
    (hour_t, min_t) = tuple(map(int, t.split(':')))
    new_min = min_t + minutes
    if new_min > 59:
        new_min = new_min % 60
        hour_t = hour_t + 1
    return f"{hour_t}:{new_min}"


def subtract_minutes(t: str, minutes: int):
    (hour_t, min_t) = tuple(map(int, t.split(':')))
    new_min = min_t - minutes
    if new_min < 0:
        new_min = 60 + new_min
        hour_t = hour_t - 1
    return f"{hour_t}:{new_min}"


def edges_equal(edge_list1, edge_list2):
    if len(edge_list1) != len(edge_list2):
        return False

    if not all(map(
            lambda x: x[0][DEPARTURE] == x[1][DEPARTURE] and x[0][ARRIVAL] == x[1][ARRIVAL] and x[0][TRIP_ID] == x[1][TRIP_ID],
            zip(edge_list1, edge_list2))):
        return False

    return True


def edge_path_exists(edge_path_list, new_edge_path):
    return new_edge_path in edge_path_list


def path_weight_function(path):
    edges = path[1]
    departure_time = edges[0][DEPARTURE]
    arrival_time = edges[-1][ARRIVAL]
    return subtract_time(arrival_time, departure_time)


class GraphTraversals:
    def __init__(self, g = None, g_rev = None):
        self.g = g
        self.g_rev = self.g.reverse(copy=True) if g_rev is None else g_rev
        self.inaccessible_edges = []
        self.inaccessible_nodes = []

    def _restore_edges_nodes(self):
        self.inaccessible_edges = []
        self.inaccessible_nodes = []

    def _weight_func(self, edge_list, time_at_prev_node, end_time, prev_trip_id, day_of_week, week_of_year=None,
                     node=None, delay_threshold=None, wait_time=0, _reversed=False, is_destination=False):

        def is_valid(edge, _reversed=False):
            if edge in self.inaccessible_edges:
                return False
            starting_id = DESTINATION if _reversed else ORIGIN
            if edge[TRIP_ID] == WALKING_TRIP:
                if prev_trip_id == WALKING_TRIP:
                    return False
                else:
                    return True

            schedule = list(edge[SCHEDULE])

            if not int(schedule[day_of_week - 1]):
                return False

            if prev_trip_id == starting_id or edge[TRIP_ID] == prev_trip_id:
                time = time_at_prev_node
                is_stable = True
            else:
                time = subtract_minutes(time_at_prev_node, 2 + wait_time) if _reversed else add_minutes(time_at_prev_node, 2 + wait_time)
                if delay_threshold and edge[TRIP_ID] != 'walk':
                    delay_prob = predict_delay_probability(
                        string_time_to_minutes(time) - string_time_to_minutes(edge[ARRIVAL]),
                        string_time_to_minutes(edge[ARRIVAL]), day_of_week - 1, week_of_year, trip_to_product[edge['trip_id']],
                        node, product_id_map, log, scaler)
                    is_stable = delay_prob > delay_threshold
                else:
                    is_stable = True

            if _reversed:
                return time_lte(edge[ARRIVAL], time) and time_lte(edge[ARRIVAL], end_time) and is_stable
            else:
                return time_gte(edge[DEPARTURE], time) and time_lte(edge[ARRIVAL], end_time)

        def sorting_function(_edge, arrival_time_at_prev_node):
            if _edge[TRIP_ID] == WALKING_TRIP:
                return math.ceil(_edge[WALKING_TIME])
            else:
                return subtract_time(_edge[ARRIVAL], arrival_time_at_prev_node)

        def sorting_function_rev(_edge, arrival_time_at_prev_node):
            if _edge[TRIP_ID] == WALKING_TRIP:
                return math.ceil(_edge[WALKING_TIME])
            else:
                return subtract_time(arrival_time_at_prev_node, _edge[ARRIVAL])

        filtered_edges = [v for (k, v) in edge_list.items() if is_valid(v, _reversed)]

        if len(filtered_edges) == 0:
            return float('inf'), OUT_OF_BOUNDS, None

        if _reversed:
            edge = min(filtered_edges, key=lambda x: sorting_function_rev(x, time_at_prev_node))
        else:
            edge = min(filtered_edges, key=lambda x: sorting_function(x, time_at_prev_node))

        if _reversed:
            arr_time = subtract_minutes(time_at_prev_node, math.ceil(edge[WALKING_TIME])) if edge[TRIP_ID] == WALKING_TRIP else edge[DEPARTURE]
        else:
            arr_time = add_minutes(time_at_prev_node, math.ceil(edge[WALKING_TIME])) if edge[TRIP_ID] == WALKING_TRIP else edge[ARRIVAL]

        if edge[TRIP_ID] == WALKING_TRIP:
            if _reversed:
                edge[DEPARTURE] = arr_time
                edge[ARRIVAL] = time_at_prev_node
            else:
                edge[DEPARTURE] = time_at_prev_node
                edge[ARRIVAL] = arr_time

        if _reversed:
            if is_destination:
                cost = subtract_time(edge[ARRIVAL], edge[DEPARTURE])
            else:
                cost = subtract_time(time_at_prev_node, arr_time)
        else:
            cost = subtract_time(arr_time, time_at_prev_node)
        return cost, arr_time, edge

    def _get_fastest_path_to_destination(self, origin, destination, end_time, day_of_week, week_of_year, threshold,
                                         wait_time=0, subgraph=None, _prev_trip_id=None):
        delay_threshold = threshold ** (1/2) if threshold else None
        graph = subgraph if subgraph is not None else self.g_rev
        distance = {k: float('inf') for k in graph.nodes}
        visited_time = {k: "00:00" for k in graph.nodes}
        prev_node_dict = {k: None for k in graph.nodes}
        visited_trip_id = {k: None for k in graph.nodes}
        distance[destination] = 0
        visited_time[destination] = end_time
        visited_trip_id[destination] = DESTINATION if _prev_trip_id is None else _prev_trip_id
        visited = set()
        valid_nodes = list(filter(lambda x: x not in self.inaccessible_nodes, graph.nodes))
        unvisited_queue = [(0, 0, destination)]
        heapq.heapify(unvisited_queue)
        while len(unvisited_queue):
            _, n_changes, node = heapq.heappop(unvisited_queue)
            visited.add(node)
            if node == origin:
                break
            for adj in graph.neighbors(node):
                if adj in visited:
                    continue
                all_edges = graph[node][adj]
                distance_adj, time_at_adj, edge = self._weight_func(all_edges, visited_time[node], end_time,
                                                                    visited_trip_id[node], day_of_week, week_of_year,
                                                                    node, delay_threshold, wait_time,
                                                                    _reversed=True,
                                                                    is_destination=node == destination)
                if time_at_adj is OUT_OF_BOUNDS:
                    # Cannot reach this neighbour given the time limits
                    continue
                new_distance = distance[node] + distance_adj
                if new_distance < distance[adj]:
                    distance[adj] = new_distance
                    visited_time[adj] = time_at_adj
                    prev_node_dict[adj] = (node, edge)
                    visited_trip_id[adj] = edge[TRIP_ID]
                    heapq.heappush(unvisited_queue, (distance[adj], n_changes + (edge[TRIP_ID] != visited_trip_id[node]), adj))

        if prev_node_dict[origin] is None:
            raise nx.NetworkXNoPath(f"No path found between {origin} and {destination} with the current constraints")

        current_node = origin
        trip = [origin]
        trip_edges = []
        number_of_trip_ids = 0
        prev_trip_id = None

        while prev_node_dict[current_node] is not None:
            prev_node, prev_edge = prev_node_dict[current_node]
            trip.append(prev_node)
            trip_edges.append(prev_edge)
            if prev_trip_id != prev_edge[TRIP_ID]:
                number_of_trip_ids += 1
                prev_trip_id = prev_edge[TRIP_ID]
            current_node = prev_node
        return trip, trip_edges, number_of_trip_ids

    def get_shortest_path(self, origin, destination, start_time, end_time, day_of_week, week_of_year, threshold=None,
                          wait_time=0, subgraph=None, _prev_trip_id=None):
        """
        Returns the trip with the shortest time (weight) from u to v by implementing the Dijktra's algorithm.
        The travel time(weight of the edge) from an node x to y in the graph should not be negative as
        this algorithm is not capable to handling time-travel.
        :param subgraph (Deprecated): Used to pass a subgraph of the original graph to this function.
        :param origin: Origin station
        :param destination: Destination station
        :param start_time:
                - Time for the start of the journey. If Start Time in none, we compute the reversed path from
                    destination to source. Both start time and end time cannot be None.
                - The passenger is assumed to be at the station at this time and ready to board.
        :param end_time: Time before which the passenger must reach v. None for no limit
        :param day_of_week: int from [1, 7] denoting monday to sunday
        :param threshold: The required confidence threshold for a successful trip.
        :param wait_time: Addition waiting time to be considered at transfer stations.
        :param _prev_trip_id: The trip id for the trip so far. Used when this function is called from yen's algorithm.

        :return: A tuple of size 3
                    [0] A sequence of nodes of the graph in order.
                        The list is sorted based on travel time, fastest journey first.
                    [1] List of edges in order.
                    [2] Number of trip_ids in the journey, Returns 1 if a single trip Id takes
                        the passenger from origin to destination, Returns 0 if origin = destination
        :raises
            NodeNotFound – If u or v is not in G.
            NetworkXNoPath – If no path exists between source and target in the given constraints.
            ValueError - If both start time and end time are none.
        """
        graph = subgraph or self.g or self.g_rev

        if origin not in graph or destination not in graph:
            raise nx.NodeNotFound("Origin/Destination node is not a part of the provided graph")

        if origin == destination:
            return [], [], 0

        if start_time is None and end_time is None:
            raise ValueError("Both start time and end time cannot be None.")

        if not (1 <= day_of_week <= 7):
            raise ValueError("Day of the week needs to be in the interval [1, 7] denoting [Monday, Sunday]")

        if start_time is None:
            return self._get_fastest_path_to_destination(origin, destination, end_time, day_of_week, week_of_year,
                                                         threshold, wait_time, subgraph)
        # Stores the shortest distance from origin to node
        distance = {k: float('inf') for k in graph.nodes}
        # Used for filtering valid edges between any two nodes
        visited_time = {}
        # Used to traverse back from destination to origin.
        prev_node_dict = {k: None for k in graph.nodes}
        # Used to check if there is a transfer at a given node
        visited_trip_id = {k: None for k in graph.nodes}
        # Initialize the variable for starting node
        distance[origin] = 0
        visited_time[origin] = start_time
        visited_trip_id[origin] = ORIGIN if _prev_trip_id is None else _prev_trip_id
        visited = set()
        # Check if the node are accessible in the graph
        valid_nodes = list(filter(lambda x: x not in self.inaccessible_nodes, graph.nodes))
        # Dijkstra's heap
        unvisited_queue = [(0, origin)]
        heapq.heapify(unvisited_queue)
        while len(unvisited_queue):
            # Visit the node
            _, node = heapq.heappop(unvisited_queue)
            visited.add(node)
            if node == destination:
                break
            for adj in graph.neighbors(node):
                # Explore it's neighbours
                if adj in visited:
                    continue
                all_edges = graph[node][adj]
                # Compute the weight of the edge between the current node and its neighbour.
                distance_adj, time_at_adj, edge = self._weight_func(all_edges, visited_time[node] + wait_time, end_time,
                                                                    visited_trip_id[node], day_of_week,
                                                                    wait_time=wait_time)
                if time_at_adj is OUT_OF_BOUNDS:
                    # Cannot reach this neighbour given the time limits
                    continue
                new_distance = distance[node] + distance_adj
                # If a shorter distance is found, use this path instead.
                if new_distance < distance[adj]:
                    distance[adj] = new_distance
                    visited_time[adj] = time_at_adj
                    prev_node_dict[adj] = (node, edge)
                    visited_trip_id[adj] = edge[TRIP_ID]
                    heapq.heappush(unvisited_queue, (distance[adj], adj))

        if prev_node_dict[destination] is None:
            raise nx.NetworkXNoPath(f"No path found between {origin} and {destination} with the current constraints")

        current_node = destination
        trip = [destination]
        trip_edges = []
        number_of_trip_ids = 0
        prev_trip_id = None

        # Traverse back from destination to origin using the prev_node_dictionary
        while prev_node_dict[current_node] is not None:
            prev_node, prev_edge = prev_node_dict[current_node]
            trip.append(prev_node)
            trip_edges.append(prev_edge)
            if prev_trip_id != prev_edge[TRIP_ID]:
                number_of_trip_ids += 1
                prev_trip_id = prev_edge[TRIP_ID]
            current_node = prev_node
        # Reverse the list to have a chronological order
        trip.reverse()
        trip_edges.reverse()
        return trip, trip_edges, number_of_trip_ids

    def get_k_shortest_path(self, origin, destination, start_time, end_time, day_of_week, week_of_year, threshold,
                            wait_time=0, K=1,
                            only_spur_transfer_nodes=True, remove_nodes=False):
        """
        Returns K shortest trips from u to v by implementing the Yen's algorithm.
        Yen's algorithm internally uses Dijkstra's to find the shortest path between 2 nodes and the graph should
        not contain negative weight cycles.
        Please refer to https://en.wikipedia.org/wiki/Yen%27s_algorithm for details about the algorithm.

        :param origin: Origin station
        :param destination: Destination station
        :param start_time:
                - Time for the start of the journey. If Start Time in none, we compute the reversed path from
                    destination to source. Both start time and end time cannot be None.
                - The passenger is assumed to be at the station at this time and ready to board.
        :param end_time: Time before which the passenger must reach v. None for no limit
        :param day_of_week: int from [1, 7] denoting monday to sunday
        :param threshold: The required confidence threshold for a successful trip.
        :param wait_time: Addition waiting time to be considered at transfer stations.
        :param K: Number of paths required
        :param only_spur_transfer_nodes: Incase of a long trip, only the transfer nodes will the spurred
        :param remove_nodes: [Deprecated]
        :return: A tuple of size 2:
                    [0]: K shortest paths from source to destination
                    [1]: All the perturbed paths
        """
        assert K > 0, "Cannot have a negative value for k"
        """
        In case the start time is None, we use a separate function to compute the same result but we travel back from
        destination to the origin. In this case the graph is reversed but the overall code follows the same algorithm.
        """
        if start_time is None:
            return self._get_k_shortest_path_reversed(
                origin, destination, start_time, end_time, day_of_week, week_of_year, threshold, wait_time, K,
                only_spur_transfer_nodes, remove_nodes)
        # A[k] stores the kth shortest paths from source to destination
        A = []
        # B contains potential shortest paths from source to destination
        B = []

        nodes, edges, _ = self.get_shortest_path(origin, destination, start_time, end_time, day_of_week, week_of_year,
                                                 threshold, wait_time)
        A.append(copy.deepcopy((nodes, edges)))
        for k in tqdm(range(1, K)):
            prev_nodes, prev_edges = A[k - 1]
            index_list = list(range(len(prev_nodes) - 1))
            for i in tqdm(index_list):
                spur_node = prev_nodes[i]
                time_at_spur_node = start_time if i == 0 else prev_edges[i - 1][ARRIVAL]
                # root path includes spur node
                root_path = prev_nodes[:i + 1]
                root_path_edges = prev_edges[:i + 1]

                for k_nodes, k_edges in A:
                    if root_path == k_nodes[:i + 1]:
                        # Mark k_edges[i] as unusable in graph
                        self.inaccessible_edges.append(k_edges[i])

                if remove_nodes:
                    for root_path_node in root_path[:-1]:
                        self.inaccessible_nodes.append(root_path_node)

                try:
                    spur_path_nodes, spur_path_edges, _ = self.get_shortest_path(
                        spur_node, destination, time_at_spur_node, end_time, day_of_week, week_of_year, threshold,
                        wait_time,
                        subgraph=None, _prev_trip_id=root_path_edges[-1][TRIP_ID])
                except nx.NetworkXNoPath:
                    continue

                total_path_nodes = root_path[:-1] + spur_path_nodes
                total_path_edges = root_path_edges[:-1] + spur_path_edges

                if not edge_path_exists(B, total_path_edges):
                    B.append(copy.deepcopy((total_path_nodes, total_path_edges)))

                self._restore_edges_nodes()
            if len(B) == 0:
                break
            B = sorted(B, key=path_weight_function)
            # Need to copy to make sure the departure and arrival time of walking edges do not change.
            A.append(B.pop(0))

        return A, B

    def _get_k_shortest_path_reversed(self, origin, destination, start_time, end_time, day_of_week, week_of_year,
                                      threshold, wait_time, K,
                                      only_spur_transfer_nodes=True, remove_nodes=False):
        assert K > 0, "Cannot have a negative value for k"
        A = []
        B = []
        nodes, edges, _ = self.get_shortest_path(origin, destination, start_time, end_time, day_of_week, week_of_year,
                                                 threshold, wait_time)
        A.append(copy.deepcopy((nodes, edges)))
        graph = self.g_rev if start_time is None else self.g
        for k in tqdm(range(1, K)):
            prev_nodes, prev_edges = A[k - 1]
            index_list = list(range(1, len(prev_nodes)))
            # index_list.reverse()
            spur_nodes = []
            long_journey = False
            if len(prev_nodes) > 10:
                long_journey = True
                if only_spur_transfer_nodes:
                    for idx in range(len(prev_edges) - 1):
                        if prev_edges[idx][TRIP_ID] != prev_edges[idx + 1][TRIP_ID]:
                            spur_nodes.append(prev_nodes[idx])
                else:
                    spur_nodes = list(map(lambda x: prev_nodes[x], random.sample(index_list, 5)))

            for i in tqdm(index_list):
                spur_node = prev_nodes[i]

                if long_journey and spur_node not in spur_nodes:
                    continue

                time_at_spur_node = end_time if i == len(prev_nodes) - 1 else prev_edges[i][DEPARTURE]
                # root path includes spur node
                root_path = prev_nodes[i:]
                root_path_edges = prev_edges[i:]

                for k_nodes, k_edges in A:
                    if root_path == k_nodes[i:]:
                        # Mark k_edges[i] as unusable in graph
                        self.inaccessible_edges.append(k_edges[i - 1])

                # if remove_nodes:
                #     for root_path_node in root_path[:-1]:
                #         self.inaccessible_nodes.append(root_path_node)

                try:
                    _prev_trip_id = root_path_edges[0][TRIP_ID] if len(root_path_edges) else None
                    spur_path_nodes, spur_path_edges, _ = self.get_shortest_path(
                        origin, spur_node, None, time_at_spur_node, day_of_week, week_of_year, threshold, wait_time,
                        subgraph=None, _prev_trip_id=_prev_trip_id)
                except nx.NetworkXNoPath:
                    continue

                total_path_nodes = spur_path_nodes + root_path[1:]
                total_path_edges = spur_path_edges + root_path_edges

                if not edge_path_exists(B, total_path_edges):
                    B.append(copy.deepcopy((total_path_nodes, total_path_edges)))

                self._restore_edges_nodes()
            if len(B) == 0:
                break
            B = sorted(B, key=path_weight_function)
            # Need to copy to make sure the departure and arrival time of walking edges do not change.
            for i in range(len(B)):
                if B[i] not in A:
                    A.append(B.pop(i))
                    break

        return A, B
