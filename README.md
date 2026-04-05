# Incremental Dynamic Truck Platooning using Augmented Persistent Segment Trees

**A Master's project in Smart Mobility & Logistics + Data Structures and Algorithms**

This repository implements an efficient **incremental** system for dynamic truck platooning on real Indian highways under **non-FIFO time-dependent traffic**. 

The system processes a continuous stream of truck requests and forms small platoons (size ≤ 4) to maximize net fuel savings from aerodynamic drafting while respecting rendezvous costs, deadlines, heterogeneous truck types, and safety constraints in mixed-traffic conditions.

### Why This Matters
Truck platooning can reduce fuel consumption by 3–8% net (after formation costs) and lower emissions in India's massive freight sector. On corridors like Bengaluru–Mysore (NH275) or Golden Quadrilateral segments, frequent traffic changes and new orders make traditional full re-computation too slow.  

This project solves the problem incrementally in logarithmic time, making real-time platoon formation practical for logistics operators in heterogeneous, single-lane, and congested Indian highway conditions.

### Core DSA Innovation
- **Non-FIFO Time-Dependent Graph**: Uses a **label-correcting algorithm** (generalized Bellman-Ford) to compute feasible paths correctly even when FIFO is violated.
- **Augmented Persistent Segment Tree** (the main novelty): 
  - Stores time-interval coverage of truck routes with augmentations for maximum fuel-gain, platoon-size bonuses (≤4), and rendezvous penalties.
  - Supports **persistent versions** so that when traffic changes or new trucks arrive, only affected parts of the tree are updated in \(O(\log N)\) time.
- **Lightweight Greedy Insertion Heuristic**: Guided by fast tree queries to decide optimal platoon insertions/merges without full recomputation.
- **Incremental Re-optimization**: Only affected platoons are updated when conditions change.

This is a pure **DSA-centric solution**. The major innovation lies in the cross-domain application of persistent segment trees (from versioned databases) to dynamic overlap maximization in mobility networks — not just applying an existing algorithm.

### Key Features
- Streaming truck request processing with time windows
- Net fuel saving optimization: FuelSaved(Platoon) − RendezvousCost
- Hard constraint enforcement (platoon size ≤ 4, deadlines)
- Incremental updates for real-time traffic changes
- Comparison against two baselines:
  - Non-platooned individual routing
  - Static platooning with full re-computation
- Evaluation on realistic highway subgraph (Bengaluru–Mysore corridor)

### Project Highlights
- **Genuine real-world impact**: Helps logistics companies reduce fuel costs and emissions while improving delivery reliability.
- **Mathematical rigour**: Handles true complexity of non-FIFO time-dependent networks and dynamic streaming updates.
- **Real datasets**: OpenStreetMap highway network + Bengaluru Traffic Pulse (timestamped speeds) + freight pattern proxies.
- **Evaluation metrics**: Net fuel/emission savings, re-optimization time, deadline compliance, and platoon size adherence.
- **Scope**: Master's-level project designed to be completed in 2–3 weeks, with focus on DSA innovation.

### Repository Structure (planned)
- `src/` — Core DSA implementation (persistent segment tree, label-correcting paths, greedy heuristic)
- `data/` — Preprocessing scripts for OSM and traffic data
- `notebooks/` — Simulation, evaluation, and baseline comparison
- `visualization/` — Highway map with platoon visualization (optional animation)
- `results/` — Metrics tables, plots, and complexity analysis

### Technologies
- Python 3
- Custom persistent segment tree implementation (from scratch for learning/DSA focus)
- NetworkX or custom graph for time-dependent routing
- Pandas / NumPy for data handling
- Folium / Plotly for geospatial visualization (optional)

### Academic Context
Developed as part of a Master's in **Smart Mobility and Logistics**. The project strictly satisfies all requirements:
- Genuine societal/industry benefit
- Deep mathematical and algorithmic complexity (non-FIFO, incremental, constrained optimization)
- DSA as the optimal solution method
- Major innovation in the data structure layer (augmented persistent segment tree)
- Use of real-world datasets for testing
- No reliance on AI/LLM in the final system

---

**Star this repo** if you are interested in DSA applications to transportation, truck platooning, persistent data structures, or smart logistics!

Contributions, suggestions, and discussions on Indian highway platooning challenges are welcome.
