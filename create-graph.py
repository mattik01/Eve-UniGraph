import requests
import networkx as nx
import time
from tqdm import tqdm
import os
import json

HEADERS = {'Accept': 'application/json'}
BASE_URL = 'https://esi.evetech.net/latest'

def get_all_system_ids():
    url = f'{BASE_URL}/universe/systems/'
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()

def get_system_data(system_id):
    url = f'{BASE_URL}/universe/systems/{system_id}/'
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 420:  # rate limited
        time.sleep(0)
        return get_system_data(system_id)
    response.raise_for_status()
    return response.json()

def get_stargate_data(gate_id):
    url = f'{BASE_URL}/universe/stargates/{gate_id}/'
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 420:
        time.sleep(0)
        return get_stargate_data(gate_id)
    response.raise_for_status()
    return response.json()

def build_graph():
    system_ids = get_all_system_ids()
    G = nx.Graph()
    system_cache = {}
    edge_set = set()

    for system_id in tqdm(system_ids, desc="Fetching systems"):
        system = get_system_data(system_id)
        system_name = system['name']
        G.add_node(system_id, name=system_name)
 
        system_cache[system_id] = system_name

        for gate_id in system.get('stargates', []):
            try:
                gate = get_stargate_data(gate_id)
                dest_system_id = gate['destination']['system_id']
                edge = tuple(sorted((system_id, dest_system_id)))
                if edge not in edge_set:
                    G.add_edge(*edge)
                    edge_set.add(edge)
            except requests.HTTPError:
                continue

    return G

def save_graph(G, path='eve_universe.graphml'):
    nx.write_graphml(G, path)
    print(f"Graph saved to {path}")

def load_graph(path='eve_universe.graphml'):
    return nx.read_graphml(path)

if __name__ == '__main__':
    graph_path = 'eve_universe.graphml'

    if os.path.exists(graph_path):
        print("Loading graph from disk...")
        G = load_graph(graph_path)
    else:
        print("Building graph from ESI API...")
        G = build_graph()
        save_graph(G, graph_path)

    print(f"Loaded graph with {len(G.nodes)} systems and {len(G.edges)} connections.")
