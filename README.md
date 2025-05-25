# EVE Unigraph

**EVE Unigraph** is a Python-based toolkit and data pipeline for building, exploring, and analyzing a graph of the entire EVE Online universe. It empowers users to update the data to their desired degree of accuracy and run custom queries to answer their questions about New Eden.

## Components

### 🔷 The Graph

- Includes systems and stargate connections, and the most important system attributes. The loading functions print some example nodes from the graph for inspection.
- A backup graph, can be committed and restored if you break stuff.

### 🧰 Utility Folder (`util/`)

- Contains configuration and lookup tables that are too specific to be embedded in the graph directly.
- Includes (And more):
  - System avoidance lists (personalizable)
  - Lookup tables for reduced api calls
  - Region metadata
  - Security classification helpers

### 🛠️ `create-graph.py`
- Initializes a clean base graph using core EVE API data (e.g., stargate topology).
- Run this when you want a fresh start or are rebuilding from scratch.

### 🔄 `graph-refiner.ipynb`
- Refines and enriches the graph with optional and current data.
- Organized top-to-bottom:
  - **Top**: Lookup table updates (e.g., region names)
  - **Middle**: Static attributes (e.g., stations, security)
  - **Bottom**: Frequently changing data (e.g., cost indices, player activity)

### ❓ `graph-queries.ipynb`

- Notebook to load and run queries on the graph.
- Includes example queries, such as
  - Finding quiet Abyssal systems
  - Finding DED Farming Systems (avoid dangerous LS spawns)

### NOT YET IMPLEMENTED
- Player Structure Information
- Market Data
- Updating Abyssal NPC Buyer Information
- and much more... if it interests me or upon request I can implement it, otherwise feel free to make your own additions. 


## Philosophy

> ⚠️ **No automatic API polling included, and this is on purpose.**  
> The project is designed to minimize API Usage: only update the parts you care about, when you need them.  
> Be kind to the API — and keep your local cache relevant to your current gameplay goals.

## Requirements

Install required packages:

```bash
pip install -r requirements.txt
