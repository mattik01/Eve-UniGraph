import networkx as nx
import requests
import time
from tqdm.notebook import tqdm
import json
import pandas as pd
import os
from datetime import datetime, timedelta
from collections import deque, defaultdict





##################################### HELPERS #######################################
HEADERS = {'Accept': 'application/json'}
BASE_URL = 'https://esi.evetech.net/latest'

### API REQUEST INTERVAL  -  BE CAREFULL REDUCING THIS ONE, LETS NOT ANNOY THE DEVELOPERS
request_interval = 0

def esi_get(url):
    """Performs a GET request to the ESI API with retry and delay."""
    while True:
        response = requests.get(url, headers=HEADERS)
        if response.status_code in [420, 429]:
            time.sleep(request_interval)
            continue
        response.raise_for_status()
        return response.json()
    
def print_node_info(node_id, data, G, max_len=3):
    def summarize(value):
        if isinstance(value, str) and ',' in value:
            parts = value.split(',')
            if len(parts) > max_len:
                return ', '.join(parts[:max_len]) + f", ... (+{len(parts)-max_len} more)"
            return value
        elif isinstance(value, (list, tuple)):
            if len(value) > max_len:
                return str(value[:max_len]) + f" ... (+{len(value)-max_len} more)"
            return str(value)
        elif isinstance(value, str) and len(value) > 100:
            return value[:100] + "..."
        return value

    print(f"Node {node_id}:")
    for key, value in data.items():
        print(f"  {key}: {summarize(value)}")

    neighbors = list(G.neighbors(str(node_id)))
    if neighbors:
        print("  Connected systems:")
        for neighbor_id in neighbors:
            neighbor_name = G.nodes[neighbor_id].get("system_name", G.nodes[neighbor_id].get("name", "Unknown"))
            print(f"    - {neighbor_name} (ID: {neighbor_id})")
    else:
        print("  No connections found.")
    print()


def print_node_info_from_df(system_id, G):
    node_id = str(system_id)
    if node_id in G.nodes:
        data = G.nodes[node_id]
        print_node_info(node_id, data, G)
    else:
        print(f"System ID {system_id} not found in graph.")


def classify_sec_type(sec):
    """Classify security status into HS/LS/NS/WH."""
    if sec >= 0.5:
        return "HS"
    elif sec >= 0.1:
        return "LS"
    elif sec >= -0.1:
        return "NS"
    else:
        return "WH"

def extract_station_services(station_table_path="util/stations_table.csv",
                              output_path="util/station_services.json"):
    """
    Extracts all unique services listed across stations into a small standalone JSON file.
    """
    df = pd.read_csv(station_table_path)

    all_services = set()
    for s in df["services"].dropna():
        all_services.update(s.split(','))

    service_list = sorted(all_services)

    with open(output_path, "w") as f:
        json.dump(service_list, f, indent=2)

    print(f"Saved {len(service_list)} unique station services to '{output_path}'.")



##################################### ROUTINES #######################################

def node_check(G, table_path="util/systems_table.csv"):
    df = pd.read_csv(table_path)
    print(f"Loaded table with {len(df)} systems.\n")

    # Dynamically classify security type
    df["sec_type"] = df["security_status"].apply(classify_sec_type)

    for sec_type in ["HS", "LS", "NS", "WH"]:
        subset = df[df["sec_type"] == sec_type]
        if not subset.empty:
            system = subset.sample(1).iloc[0]
            print(f"{sec_type}:")
            print_node_info_from_df(system["system_id"], G)
        else:
            print(f"{sec_type}: empty")


def update_system_table(G, output_path="util/systems_table.csv"):
    """
    Fetch system metadata from the EVE API and store it in a CSV file.
    """
    rows = []
    for node_id in tqdm(G.nodes, desc="Downloading system metadata"):
        try:
            system_data = esi_get(f"{BASE_URL}/universe/systems/{node_id}")
        except Exception as e:
            print(f"Failed to fetch system {node_id}: {e}")
            continue

        planets = system_data.get('planets', [])
        moon_count = sum(len(p.get('moons', [])) for p in planets)
        stations = system_data.get('stations', [])
        station_str = ','.join(str(s) for s in stations) if stations else ""

        rows.append({
            "system_id": node_id,
            "system_name": system_data.get("name"),
            "security_status": system_data.get("security_status", 0.0),
            "security_class": system_data.get("security_class", 'N/A'),
            "constellation_id": system_data.get("constellation_id"),
            "star_id": system_data.get("star_id", 'N/A'),
            "planet_count": len(planets),
            "moon_count": moon_count,
            "station_count": len(stations),
            "stations": station_str,
            "has_stargates": bool(system_data.get("stargates")),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved system metadata to '{output_path}'.")


def load_system_data_to_graph(G, table_path="util/systems_table.csv"):
    """
    Read system metadata from CSV and update graph nodes.
    """
    df = pd.read_csv(table_path)
    df = df.set_index(df["system_id"].astype(str))  # ensure node_id is string like in G

    for node_id in G.nodes:
        if node_id not in df.index:
            print(f"Warning: System {node_id} missing from table.")
            continue
        row = df.loc[node_id]
        node = G.nodes[node_id]

        node["system_name"] = row["system_name"]
        node["security_status"] = row["security_status"]
        node["security_class"] = row["security_class"]
        node["constellation_id"] = int(row["constellation_id"])
        node["star_id"] = row["star_id"]
        node["planet_count"] = int(row["planet_count"])
        node["moon_count"] = int(row["moon_count"])
        node["station_count"] = int(row["station_count"])
        node["stations"] = row["stations"]
        node["has_stargates"] = bool(row["has_stargates"])

    print("Graph updated with system metadata from CSV.")



def update_station_table(G, output_path="util/stations_table.csv"):
    """
    Creates a table with all known station IDs from the graph and enriches them with ESI data,
    including available services.
    """
    seen_station_ids = set()

    for _, data in G.nodes(data=True):
        station_str = data.get('stations', '')
        ids = station_str.split(',') if station_str else []
        seen_station_ids.update(int(s.strip()) for s in ids if s.strip().isdigit())

    print(f"Found {len(seen_station_ids)} unique station IDs.")

    rows = []
    for station_id in tqdm(seen_station_ids, desc="Fetching station data"):
        try:
            station = esi_get(f"{BASE_URL}/universe/stations/{station_id}")
        except Exception as e:
            print(f"Failed to fetch station {station_id}: {e}")
            continue

        # Fetch corp name
        corp_id = station.get("owner")
        try:
            corp_data = esi_get(f"{BASE_URL}/corporations/{corp_id}") if corp_id else {}
            corp_name = corp_data.get("name", "Unknown")
        except Exception:
            corp_name = "Unknown"

        pos = station.get("position", {})
        services = station.get("services", [])
        services_str = ','.join(services) if services else ""

        rows.append({
            "station_id": station_id,
            "name": station.get("name"),
            "system_id": station.get("system_id"),
            "type_id": station.get("type_id"),
            "owner_corp_id": corp_id,
            "owner_corp_name": corp_name,
            "race_id": station.get("race_id"),
            "reprocessing_efficiency": station.get("reprocessing_efficiency"),
            "reprocessing_stations_take": station.get("reprocessing_stations_take"),
            "max_dockable_ship_volume": station.get("max_dockable_ship_volume"),
            "x": pos.get("x"),
            "y": pos.get("y"),
            "z": pos.get("z"),
            "services": services_str
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved station table to '{output_path}'.")



def build_system_constellation_region_table(G, output_path="util/system_constellation_region.csv"):
    """
    Creates a table with system_id → constellation → region mapping and saves it to CSV.
    """
    rows = []
    constellation_ids = set()
    for node_id, data in G.nodes(data=True):
        constellation_ids.add(int(data.get("constellation_id", -1)))

    # Step 1: Get constellation info
    constellation_map = {}
    region_ids = set()
    for cid in tqdm(constellation_ids, desc="Fetching constellations"):
        try:
            c_data = esi_get(f"{BASE_URL}/universe/constellations/{cid}")
            constellation_map[cid] = {
                "constellation_name": c_data["name"],
                "region_id": c_data["region_id"]
            }
            region_ids.add(c_data["region_id"])
        except Exception as e:
            print(f"Constellation {cid} failed: {e}")

    # Step 2: Get region info
    region_map = {}
    for rid in tqdm(region_ids, desc="Fetching regions"):
        try:
            r_data = esi_get(f"{BASE_URL}/universe/regions/{rid}")
            region_map[rid] = r_data["name"]
        except Exception as e:
            print(f"Region {rid} failed: {e}")

    # Step 3: Build output rows
    for node_id, data in G.nodes(data=True):
        const_id = int(data.get("constellation_id", -1))
        const_info = constellation_map.get(const_id, {})
        region_id = const_info.get("region_id")
        rows.append({
            "system_id": node_id,
            "system_name": data.get("system_name", data.get("name", "Unknown")),
            "constellation_id": const_id,
            "constellation_name": const_info.get("constellation_name", "Unknown"),
            "region_id": region_id,
            "region_name": region_map.get(region_id, "Unknown")
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved system/constellation/region table to '{output_path}'.")


def update_graph_with_region_info(G, table_path="util/system_constellation_region.csv"):
    """
    Updates each graph node with 'region_id', 'region_name', and 'constellation_name'
    using the precomputed CSV lookup table.
    """
    df = pd.read_csv(table_path)
    df = df.set_index(df["system_id"].astype(str))  # ensure string index matches graph node IDs

    for node_id in G.nodes:
        if node_id in df.index:
            row = df.loc[node_id]
            G.nodes[node_id]["region_id"] = int(row["region_id"])
            G.nodes[node_id]["region_name"] = row["region_name"]
            G.nodes[node_id]["constellation_name"] = row["constellation_name"]
        else:
            print(f"Warning: system_id {node_id} not found in region table.")



def assign_factions_by_region(G, json_path="util/region_factions.json"):
    """
    Loads a region → faction map from JSON and assigns 'faction_name' to each system.
    Systems in unknown regions get faction_name = None.
    """
    with open(json_path, "r") as f:
        region_to_faction = json.load(f)

    for node_id, data in G.nodes(data=True):
        region_name = data.get("region_name")
        G.nodes[node_id]["faction_name"] = region_to_faction.get(region_name, "None")

    print("Updated 'faction_name' field for all systems.")


def update_graph_with_industry_indices(G):
    """
    Fetches live industry indices and adds each activity's cost index
    as a separate node attribute, e.g.:
        industry_manufacturing = 0.0016
        industry_copying = 0.0014
    This version avoids dicts entirely for GraphML compatibility.
    """
    print("Fetching industry index data...")
    industry_url = f"{BASE_URL}/industry/systems/"
    try:
        industry_data = esi_get(industry_url)
    except Exception as e:
        print(f"Failed to fetch industry data: {e}")
        return

    updated = 0
    skipped = 0

    for system in tqdm(industry_data, desc="Updating graph with industry data"):
        node_id = str(system["solar_system_id"])
        if node_id in G.nodes:
            for entry in system["cost_indices"]:
                activity = entry["activity"]
                cost = entry["cost_index"]
                attr_name = f"industry_{activity}"
                G.nodes[node_id][attr_name] = cost

            updated += 1
        else:
            skipped += 1

    print(f"Updated industry indices for {updated} systems. Skipped {skipped} systems (not in graph).")

def update_activity_snapshots(G):
    """
    Updates or appends a snapshot of activity for all systems.
    Missing systems in the ESI responses are recorded with 0/NaN.
    """
    os.makedirs("activity_data", exist_ok=True)
    timestamp = datetime.utcnow().isoformat()

    # Fetch data
    kills_data = esi_get(f"{BASE_URL}/universe/system_kills/")
    jumps_data = esi_get(f"{BASE_URL}/universe/system_jumps/")

    # Fast lookup
    kills_map = {entry["system_id"]: entry for entry in kills_data}
    jumps_map = {entry["system_id"]: entry["ship_jumps"] for entry in jumps_data}

    # Update all systems in graph
    for node_id in G.nodes:
        system_id = int(node_id)

        kill_entry = kills_map.get(system_id, {})
        data = {
            "timestamp": timestamp,
            "npc_kills": kill_entry.get("npc_kills", 0),
            "pod_kills": kill_entry.get("pod_kills", 0),
            "ship_kills": kill_entry.get("ship_kills", 0),
            "ship_jumps": jumps_map.get(system_id, 0)  # Can use 0 instead of NaN for consistency
        }

        path = f"activity_data/system_{system_id}.csv"
        df = pd.DataFrame([data])

        if os.path.exists(path):
            df.to_csv(path, mode='a', header=False, index=False)
        else:
            df.to_csv(path, mode='w', header=True, index=False)

    print(f"Updated activity data for {len(G.nodes)} systems.")


def average_activity(system_id, hours=6):
    path = f"activity_data/system_{system_id}.csv"
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, parse_dates=["timestamp"])
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=hours)
    recent = df[df["timestamp"] >= cutoff]

    if recent.empty:
        return None

    return recent.mean(numeric_only=True).to_dict()



def compute_quiet_factor(system_id, weight_vector=(1.0, 1.0, 1.0), days=1, activity_dir="activity_data"):
    """
    Computes a quietness score for a system.

    Parameters:
    - system_id: int or str (system ID)
    - weight_vector: tuple of 3 floats (ship+pod kills, ship jumps, npc kills)
    - days: int, how many past days to average over
    - activity_dir: str, folder where activity CSVs are stored

    Returns:
    - Tuple: (quietness score, ship+pod kills, ship jumps, npc kills), or None if no valid data
    """
    path = os.path.join(activity_dir, f"system_{system_id}.csv")
    if not os.path.exists(path):
        print(f"No activity file for system {system_id}.")
        return None

    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None

    # Filter to relevant time window
    cutoff = datetime.utcnow() - timedelta(days=days)
    df = df[df["timestamp"] >= cutoff]
    if df.empty:
        return None

    # Compute averages
    ship_kills = df["ship_kills"].mean() or 0
    pod_kills = df["pod_kills"].mean() or 0
    npc_kills = df["npc_kills"].mean() or 0
    jumps = df["ship_jumps"].mean() or 0

    total_kills = ship_kills + pod_kills
    quietness = (
        weight_vector[0] * total_kills +
        weight_vector[1] * jumps +
        weight_vector[2] * npc_kills
    )

    return (quietness, total_kills, jumps, npc_kills)



def displayed_sec(sec):
    return 0.1 if 0.0 <= sec < 0.05 else round(sec, 1)

def load_avoid_set(path="util/system_avoid_list.csv"):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    return set(df["system_name"].str.strip())



def load_trig_buyer_systems(csv_path="util/triglavian_buy_stations.csv"):
    if not os.path.exists(csv_path):
        print(f"Missing triglavian buy station CSV: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["system_name", "volume_remain"])

    # Group by system and sum the volumes
    grouped = df.groupby("system_name", as_index=False).agg({
        "region": "first",  # Keep one region (assumed consistent)
        "sec_status": "first",  # Keep one security status
        "volume_remain": "sum"  # Sum all buy volumes
    })

    return grouped


# Enhanced color mapping utility with more shades and optional non-linear scaling
COLORS = [
    "\033[92m",  # Bright Green
    "\033[32m",  # Dark Green
    "\033[96m",  # Teal
    "\033[93m",  # Yellow-Green
    "\033[33m",  # Yellow
    "\033[91m",  # Orange-Red
    "\033[31m"   # Red
]

def colorize(value, min_val, max_val, reverse=False, nonlinear_exp=None):
    if max_val == min_val:
        return f"{value:.2f}"  # Avoid divide by zero
    ratio = (value - min_val) / (max_val - min_val)
    if reverse:
        ratio = 1.0 - ratio
    if nonlinear_exp:
        ratio = ratio ** nonlinear_exp
    index = int(ratio * (len(COLORS) - 1))
    color = COLORS[min(index, len(COLORS) - 1)]
    return f"{color}{value:.2f}\033[0m"

def normalize_metric_ranges(results):
    jump_vals = [r["jumps"] for r in results]
    qf_vals = [r["quiet_factor"] for r in results]
    inc_jumps = [dist for r in results for _, _, dist, _, _ in r["inclusions"]]
    inc_qfs = [qf for r in results for _, _, _, qf, _ in r["inclusions"]]
    trig_jumps = [dist for r in results for _, _, _, dist, _, _ in r["trig_buyers"]]
    trig_qfs = [qf for r in results for _, _, _, _, qf, _ in r["trig_buyers"]]

    return {
        "jumps": (min(jump_vals), max(jump_vals)),
        "qf": (min(qf_vals), max(qf_vals)),
        "inc_jumps": (min(inc_jumps), max(inc_jumps)) if inc_jumps else (0, 1),
        "inc_qf": (min(inc_qfs), max(inc_qfs)) if inc_qfs else (0, 1),
        "trig_jumps": (min(trig_jumps), max(trig_jumps)) if trig_jumps else (0, 1),
        "trig_qf": (min(trig_qfs), max(trig_qfs)) if trig_qfs else (0, 1)
    }


def find_abyss_hubs(G,
                    start_system,
                    target_sec=0.6,
                    inclusion_vector=[0.5, 0.7],
                    inclusion_max_jumps=3,
                    use_avoid_list=True,
                    only_hs=True,
                    max_search_range=8,
                    quiet_weights=(0.4, 0.3, 0.3),
                    quiet_days=10000,
                    verbose=True):

    target_sec = round(target_sec, 1)
    inclusion_vector = [round(s, 1) for s in inclusion_vector]
    avoid_set = load_avoid_set() if use_avoid_list else set()
    trig_buyers = load_trig_buyer_systems()

    if isinstance(start_system, str):
        start_id = next((n for n, d in G.nodes(data=True) if d.get("system_name") == start_system), None)
    else:
        start_id = str(start_system)

    if start_id is None or start_id not in G.nodes:
        print(f"Start system '{start_system}' not found in graph.")
        return

    results = []
    visited = set()
    queue = deque([(start_id, 0, 0, [])])  # (system_id, jumps, lowsec_jumps, lowsec_names)

    if verbose:
        print(f"Starting search from system: {start_system} (ID: {start_id})")

    while queue:
        current_id, jumps, lowsec_jumps, low_names = queue.popleft()
        if jumps > max_search_range or current_id in visited:
            continue
        visited.add(current_id)

        data = G.nodes[current_id]
        sec = displayed_sec(data.get("security_status", -1))
        name = data.get("system_name")

        for nb in G.neighbors(current_id):
            nb_data = G.nodes[nb]
            nb_name = nb_data.get("system_name")
            nb_sec = nb_data.get("security_status", 0)
            if use_avoid_list and nb_name in avoid_set:
                continue
            if only_hs and nb_sec < 0.5:
                continue
            new_low_names = low_names + [nb_name] if nb_sec < 0.5 else low_names
            new_low = lowsec_jumps + (1 if nb_sec < 0.5 else 0)
            queue.append((nb, jumps + 1, new_low, new_low_names))

        if only_hs and lowsec_jumps > 0:
            continue
        if name in avoid_set:
            continue
        if sec != target_sec:
            continue

        inclusion_hits = {}
        for target_inc_sec in inclusion_vector:
            found = False
            inc_visited = set()
            inc_queue = deque([(current_id, 0)])
            while inc_queue and not found:
                nid, d = inc_queue.popleft()
                if d > inclusion_max_jumps or nid in inc_visited:
                    continue
                inc_visited.add(nid)

                s = displayed_sec(G.nodes[nid].get("security_status", -1))
                if s == target_inc_sec:
                    path = nx.shortest_path(G, current_id, nid)
                    if all(G.nodes[n].get("security_status", 0) >= 0.5 for n in path):
                        inclusion_hits[target_inc_sec] = (nid, d)
                        found = True
                for nb in G.neighbors(nid):
                    inc_queue.append((nb, d + 1))

        if len(inclusion_hits) != len(inclusion_vector):
            continue

        qf_main = compute_quiet_factor(current_id, quiet_weights, quiet_days)
        if qf_main is None:
            continue
        _, s_k, s_j, s_n = qf_main
        total_qf = quiet_weights[0]*s_k + quiet_weights[1]*s_j + quiet_weights[2]*s_n

        inc_info = []
        for inc_sec in inclusion_vector:
            sid, dj = inclusion_hits[inc_sec]
            qf = compute_quiet_factor(sid, quiet_weights, quiet_days)
            if qf is None:
                break
            _, i_k, i_j, i_n = qf
            inc_info.append((G.nodes[sid]["system_name"], displayed_sec(G.nodes[sid]["security_status"]), dj, i_k + i_j + i_n, (i_k, i_j, i_n)))
        else:
            # Find 3 closest trig buyer systems
            distances = []
            for _, row in trig_buyers.iterrows():
                buyer_name = row["system_name"]
                if buyer_name in avoid_set:
                    continue
                sid = next((n for n, d in G.nodes(data=True) if d.get("system_name") == buyer_name), None)
                if sid is None:
                    continue
                try:
                    path = nx.shortest_path(G, current_id, sid)
                    if only_hs and any(G.nodes[n].get("security_status", 1) < 0.5 for n in path):
                        continue
                    lowsec_on_path = [G.nodes[n]["system_name"] for n in path if G.nodes[n].get("security_status", 1) < 0.5]
                    qf = compute_quiet_factor(sid, quiet_weights, quiet_days)
                    if qf is None:
                        continue
                    _, bk, bj, bn = qf
                    dist = len(path) - 1
                    distances.append((buyer_name, displayed_sec(G.nodes[sid].get("security_status", 1)), row["volume_remain"], dist, bk + bj + bn, lowsec_on_path))
                except:
                    continue
            distances.sort(key=lambda x: x[3])

            results.append({
                "system_id": current_id,
                "name": name,
                "region": data.get("region_name", "Unknown"),
                "sec": sec,
                "jumps": jumps,
                "lowsec_jumps": lowsec_jumps,
                "lowsec_names": low_names,
                "quiet_factor": total_qf,
                "breakdown": (s_k, s_j, s_n),
                "inclusions": inc_info,
                "trig_buyers": distances[:3]
            })

    results.sort(key=lambda r: r["quiet_factor"])

    ranges = normalize_metric_ranges(results)
    avoid_message = "System Avoid List used ✓" if use_avoid_list else ""

    for r in results:
        print(f"\n--- {r['name']} ({r['region']}) ({r['sec']}) {avoid_message} ---")
        print(f"  Jumps from start: {colorize(r['jumps'], *ranges['jumps'], reverse=False)} (from {start_system}) | "
              f"Lowsec jumps: {r['lowsec_jumps']} [{', '.join(r['lowsec_names']) if r['lowsec_names'] else 'None'}]")
        sk, sj, sn = r["breakdown"]
        print(f"  Quiet: {colorize(r['quiet_factor'], *ranges['qf'], reverse=False, nonlinear_exp=0.5)}  "
              f"(s+pk: {sk:.1f}, j: {sj:.1f}, NPCk: {sn:.1f})")
        print("  Close Alt Sec Systems:")
        for name, sec, dist, qf, (ik, ij, inpc) in r["inclusions"]:
            print(f"    ↳ {name:20} (sec {sec:.1f}) | {colorize(dist, *ranges['inc_jumps'], reverse=False)} jumps | "
                  f"QF: {colorize(qf, *ranges['inc_qf'], reverse=False, nonlinear_exp=0.5)}  "
                  f"(s+pk: {ik:.1f}, j: {ij:.1f}, NPCk: {inpc:.1f})")
        print("  Closest Triglavian Buyer Systems:")
        for name, sec, volume, dist, qf, lowsec in r["trig_buyers"]:
            print(f"    → {name:20} (sec {sec:.1f}) | {colorize(dist, *ranges['trig_jumps'], reverse=False)} jumps | "
                  f"Buy vol: {volume} | QF: {colorize(qf, *ranges['trig_qf'], reverse=False, nonlinear_exp=0.5)} | "
                  f"LS: {', '.join(lowsec) if lowsec else 'None'}")
            
        print(" ")
        print("------------------------------------------------------------------------------------------------------------")
        print(" ")


def count_reachable_nodes(graph, start_node, max_jumps):
    visited = set()
    queue = deque([(start_node, 0)])
    while queue:
        nid, dist = queue.popleft()
        if nid in visited or dist > max_jumps:
            continue
        visited.add(nid)
        for neighbor in graph[nid]:
            if neighbor not in visited:
                queue.append((neighbor, dist + 1))
    return visited

def count_highsec_reachable_nodes(graph, start_node, max_jumps):
    visited = set()
    queue = deque([(start_node, 0)])
    while queue:
        nid, dist = queue.popleft()
        if nid in visited or dist > max_jumps:
            continue
        visited.add(nid)
        for neighbor in graph[nid]:
            if neighbor not in visited and graph.nodes[neighbor].get("security_status", 0) >= 0.5:
                queue.append((neighbor, dist + 1))
    return visited


def find_ded_farming_bases(
    graph,
    target_faction,
    spawn_radius=15,
    close_range=5,
    use_avoid_list=True,
    avoid_list_path="util/system_avoid_list.csv",
    quiet_weights=(0.1, 0.3, 0.6),
    quiet_days=20,
    avoid_multiplier=3.0,
    top_k=10
):
    avoid_set = load_avoid_set(avoid_list_path) if use_avoid_list else set()

    candidates = [
        nid for nid, data in graph.nodes(data=True)
        if data.get("security_status", 0) >= 0.5
        and data.get("faction_name", "").lower() == target_faction.lower()
    ]

    results = []

    for system_id in tqdm(candidates, desc="Evaluating candidates"):
        visited = set()
        queue = deque([(system_id, 0)])
        ls_penalty_score = 0
        ls_systems = []
        close_hs_qfs = []

        # Cache high-sec-only reachable systems from this system
        hs_reachable_set = count_highsec_reachable_nodes(graph, system_id, spawn_radius)

        while queue:
            nid, dist = queue.popleft()
            if nid in visited or dist > spawn_radius:
                continue
            visited.add(nid)

            node = graph.nodes[nid]
            sec = node.get("security_status", 0)
            name = node.get("system_name", "Unknown")

            ds = displayed_sec(sec)

            if sec >= 0.5 and dist <= close_range:
                qf = compute_quiet_factor(nid, quiet_weights, quiet_days)
                if qf:
                    close_hs_qfs.append(qf[0])

            elif 0.1 <= ds < 0.5:
                remaining_range = spawn_radius - dist
                from_ls_reachable = count_reachable_nodes(graph, nid, remaining_range)
                exposure = len(from_ls_reachable - hs_reachable_set)
                penalty = avoid_multiplier if name in avoid_set else 1.0
                ls_penalty_score += exposure * penalty
                ls_systems.append((name, dist, penalty))

            for neighbor in graph[nid]:
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))

        base_qf_data = compute_quiet_factor(system_id, quiet_weights, quiet_days)
        if not base_qf_data:
            continue

        avg_close_qf = sum(close_hs_qfs) / len(close_hs_qfs) if close_hs_qfs else None
        ls_dists = [d for _, d, _ in ls_systems]

        results.append({
            "id": system_id,
            "name": graph.nodes[system_id]["system_name"],
            "region": graph.nodes[system_id].get("region_name", "Unknown"),
            "sec": displayed_sec(graph.nodes[system_id].get("security_status", 0)),
            "score": ls_penalty_score,
            "quiet_factor": base_qf_data[0],
            "avg_surrounding_qf": avg_close_qf,
            "nearby_hs_count": len(close_hs_qfs),
            "lowsec_neighbors": ls_systems,
            "ls_count": len(ls_dists),
            "ls_avg_dist": sum(ls_dists) / len(ls_dists) if ls_dists else None
        })

    # Sort by score
    results.sort(key=lambda x: x["score"])

     # Normalize metrics for color scaling
    score_vals = [r["score"] for r in results[:top_k]]
    qf_vals = [r["quiet_factor"] for r in results[:top_k]]
    hs_counts = [r["nearby_hs_count"] for r in results[:top_k]]
    ls_counts = [r["ls_count"] for r in results[:top_k]]
    avg_ls_dists = [r["ls_avg_dist"] for r in results[:top_k] if r["ls_avg_dist"] is not None]

    min_score, max_score = min(score_vals), max(score_vals)
    min_qf, max_qf = min(qf_vals), max(qf_vals)
    max_hs = max(hs_counts)
    max_ls = max(ls_counts)
    min_ls_avg_dist, max_ls_avg_dist = (
        min(avg_ls_dists), max(avg_ls_dists)
    ) if avg_ls_dists else (0, 1)

    print(f"\nTop {top_k} candidate systems for DED farming ({target_faction}):")
    for r in results[:top_k]:
        print(f"\n--- {r['name']} ({r['region']}) ({r['sec']}) — {graph.nodes[r['id']].get('constellation_name', 'Unknown')} ---")
        print(f"Score: {colorize(r['score'], min_score, max_score)}")
        print(f"Quiet Factor: {colorize(r['quiet_factor'], min_qf, max_qf)}")
        if r['avg_surrounding_qf'] is not None:
            print(f"Avg Nearby HS Quietness (≤ {close_range} jumps): {colorize(r['avg_surrounding_qf'], min_qf, max_qf)}")
        print(f"Nearby HS Systems: {colorize(r['nearby_hs_count'], 0, max_hs)}")
        print(f"Nearby LS Systems in range: {colorize(r['ls_count'], 0, max_ls)}")

        if r['ls_avg_dist'] is not None:
            print(f"Avg LS Distance: {colorize(r['ls_avg_dist'], min_ls_avg_dist, max_ls_avg_dist, reverse=True)}")

        # Summarize LS system distances
        dist_summary = defaultdict(int)
        for _, dist, _ in r["lowsec_neighbors"]:
            dist_summary[dist] += 1
        if dist_summary:
            print("LS system distance distribution:")
            for dist in sorted(dist_summary):
                print(f"  - {dist_summary[dist]} at distance {dist}")

        avoid_ls = []

        # Create reverse lookup from system name to node ID
        name_to_id = {
            data.get("system_name", "").strip(): nid
            for nid, data in graph.nodes(data=True)
        }

        for avoid_name in avoid_set:
            nid = name_to_id.get(avoid_name)
            if not nid:
                continue  # Skip unknown names

            # BFS from the candidate system to find distance
            visited = set()
            q = deque([(r["id"], 0)])
            while q:
                current, d = q.popleft()
                if current == nid:
                    if d <= spawn_radius:
                        avoid_ls.append((avoid_name, d))
                    break
                if current in visited or d > spawn_radius:
                    continue
                visited.add(current)
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        q.append((neighbor, d + 1))

                # ✅ Final clean print
        if avoid_ls:
            print("Avoid-list systems (within radius):")
            for name, dist in sorted(avoid_ls, key=lambda x: x[1]):
                dist_col = colorize(dist, 0, spawn_radius, reverse=True)
                print(f"  - \033[91m{name}\033[0m (Distance: {dist_col})")

    return results
