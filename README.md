# PlatoonMVP: Incremental Truck Platooning Optimization

**Course Project**: Introduction to Algorithms and Software Programming  
**Institution**: Indian Institute of Science (IISc), Bengaluru  
**Department**: Centre for infrastructure, Sustainable Transportation, and Urban Planning (CiSTUP)  
**Programme**: M.Tech in Smart Mobility and Logistics Systems  
**Author**: Swapnil Aggarwal (Student ID: 26040)

---

## 📌 Overview

**PlatoonMVP** is an online platooning coordination engine designed for hub-and-spoke logistics networks. Truck platooning—where multiple trucks travel in close formation to exploit aerodynamic drafting—can significantly reduce fuel consumption. However, matching trucks with different origins, destinations, and schedules in real-time is computationally complex (NP-Hard).

This project solves the coordination problem using:
1. A **Hub-and-Spoke Network Topology** modeled on the NH275 Bengaluru–Mysuru Expressway.
2. **Spatio-Temporal Indexing** using interval trees for fast overlap querying.
3. A **Greedy Online Algorithm** that matches trucks in real-time (< 50ms per truck) without expensive global optimization.
4. **Empirical Traffic Data Integration** using OpenStreetMap distances and Kaggle traffic datasets for time-dependent congestion modeling.

## ✨ Key Features

* **Real-time Truck Advisor**: A decision-support module (`truck_advisor.py`) that tells fleet managers exactly *when to depart, what speed to maintain, who to platoon with, and expected fuel savings*.
* **Aerodynamic Fuel Physics**: Calculates precise fuel savings based on a truck's position in the platoon (e.g., 10% for the first follower, 3% for the leader) while accounting for rendezvous fuel penalties.
* **Time-Dependent Travel Times**: Uses a 96-slot (15-minute resolution) matrix to model realistic highway congestion (rush hours vs. free-flow).
* **Monte Carlo Validation**: Proves the algorithm's robustness against randomized truck arrival sequences.

## 📂 Project Structure

```
PlatoonMVP/
├── platoonMVP/              # Core Algorithmic Engine
│   ├── coordinator.py       # Greedy platoon matching logic
│   ├── fuel_physics.py      # Aerodynamic drag & fuel savings model
│   ├── road_network.py      # Hub-spoke graph & interval tree indexing
│   ├── travel_time.py       # Time-dependent congestion matrices
│   ├── network_config.py    # Hub definitions & network parameters
│   ├── truck_advisor.py     # Per-truck recommendation engine
│   └── tests/               # Unit tests
├── integration/             # Real-world Data Pipelines
│   ├── osm_distances.py     # OpenStreetMap distance calculations
│   ├── traffic_data.py      # Kaggle dataset parsing & calibration
│   └── run_integration.py   # Integration pipeline runner
├── validation/              # Testing & Analytics
│   ├── run_validation.py    # Full validation suite
│   ├── monte_carlo.py       # Arrival-order robustness testing
│   ├── baseline.py          # Solo-driving baseline calculations
│   └── truck_generator.py   # Synthetic truck fleet generator
├── PlatoonMVPReport.pdf     # Compiled academic report
├── .gitignore
└── README.md
```

## 🚀 Setup & Installation

**Prerequisites**: Python 3.8+

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/PlatoonMVP.git
   cd PlatoonMVP
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install intervaltree networkx pandas osmnx
   ```

## 💻 Usage

### Run the Truck Advisor Demo
Generate comprehensive travel plans, fuel savings, and platoon assignments for a sample fleet:
```bash
python -m platoonMVP.truck_advisor
```

### Run the Full Validation Suite
Run Monte Carlo simulations and verify performance metrics:
```bash
python -m validation.run_validation
```

## 📊 Results Summary

When tested with 1,000 trucks over randomized arrival sequences, the system consistently achieves:

| Metric | Target | Achieved |
|--------|--------|----------|
| Net Fuel Savings | > 3% | ✅ Met |
| Platoon Participation | > 40% | ✅ Met |
| Average Query Latency | < 50ms | ✅ Met |
| Savings Std. Deviation | < 2% | ✅ Met |

## 📄 Report

The full academic report is available as [PlatoonMVPReport.pdf](PlatoonMVPReport.pdf) in the repository root.

---
