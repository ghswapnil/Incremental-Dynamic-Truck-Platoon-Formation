# PlatoonMVP: Incremental Truck Platooning Optimization

**Course Project**: Introduction to Algorithms and Software Programming  
**Institution**: Indian Institute of Science (IISc), Bengaluru  
**Department**: Centre for infrastructure, Sustainable Transportation, and Urban Planning (CiSTUP)  
**Programme**: M.Tech in Smart Mobility and Logistics Systems  
**Author**: Swapnil Aggarwal (Student ID: 26040)

---

## 📌 Overview

**PlatoonMVP** is a production-grade online platooning coordination engine designed for hub-and-spoke logistics networks. Truck platooning—where multiple trucks travel in close formation to exploit aerodynamic drafting—can significantly reduce fuel consumption. However, matching trucks with different origins, destinations, and schedules in real-time is computationally complex (NP-Hard).

This project solves the coordination problem using:
1. A **Hub-and-Spoke Network Topology** modeled on the NH275 Bengaluru--Mysuru Expressway.
2. **Spatio-Temporal Indexing** using interval trees for fast $O(k \log n)$ overlap querying.
3. A **Greedy Online Algorithm** that matches trucks in real-time ($< 50$ms per truck) without the need for expensive global optimization.
4. **Empirical Traffic Data Integration** using OpenStreetMap for accurate distances and Kaggle traffic datasets for time-dependent congestion modeling.

## ✨ Key Features

* **Real-time Truck Advisor**: A decision-support module (`truck_advisor.py`) that tells fleet managers exactly *when to depart, what speed to maintain, who to platoon with, and expected fuel savings*.
* **Aerodynamic Fuel Physics**: Calculates precise fuel savings based on a truck's position in the platoon (e.g., 10% savings for the first follower, 3% for the leader) while accounting for the fuel cost of speeding up/slowing down to rendezvous.
* **Time-Dependent Travel Times**: Uses a 96-slot (15-minute resolution) matrix to model realistic highway congestion (rush hours vs. free-flow).
* **Monte Carlo Validation**: Proves the algorithm's robustness against different, randomized truck arrival sequences.
* **Interactive Dashboard**: A Streamlit interface to visualize the corridor and metrics.

## 📂 Project Structure

```
PlatoonMVP/
├── platoon/                 # Core Algorithmic Engine
│   ├── coordinator.py       # The greedy matching logic
│   ├── fuel_physics.py      # Aerodynamic drag & fuel math
│   ├── road_network.py      # Graph topology & interval trees
│   ├── travel_time.py       # Time-dependent congestion matrices
│   └── truck_advisor.py     # Per-truck recommendation CLI
├── integration/             # Real-world Data Handlers
│   ├── osm_distances.py     # OpenStreetMap distance calculations
│   └── traffic_data.py      # Kaggle dataset parsing & calibration
├── validation/              # Testing & Analytics
│   ├── run_validation.py    # Generates thesis-grade metrics
│   ├── monte_carlo.py       # Arrival-order robustness testing
│   └── baseline.py          # Solo-driving baseline calculations
├── report/                  # Documentation
│   └── platoon_report.tex   # Comprehensive academic LaTeX report
├── Home.py                  # Streamlit Web App entry point
└── requirements.txt         # Project dependencies
```

## 🚀 Setup & Installation

**Prerequisites**: Python 3.8+

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/PlatoonMVP.git
   cd PlatoonMVP
   ```

2. **Create and activate a virtual environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   *(If you don't have a requirements.txt generated yet, you will need `intervaltree`, `networkx`, `pandas`, `osmnx`, and `streamlit`)*
   ```bash
   pip install intervaltree networkx pandas osmnx streamlit
   ```

## 💻 Usage

### 1. Run the Truck Advisor Demo
To see the system generate comprehensive travel plans, fuel savings, and platoon assignments for a sample fleet:
```bash
python -m platoon.truck_advisor
```

### 2. Run the Full Validation Suite
To run the Monte Carlo simulations and verify the algorithm's performance metrics ($>3\%$ fuel savings, $>40\%$ participation, $<50$ms latency):
```bash
python -m validation.run_validation
```

### 3. Launch the Dashboard
To open the interactive Streamlit UI:
```bash
streamlit run Home.py
```

## 📊 Results Summary

When tested with 1,000 trucks over randomized arrival sequences, the system consistently achieves:
* **Net Fuel Savings:** $> 3.0\%$ across the entire fleet.
* **Platoon Participation:** $> 40\%$ of trucks successfully matched.
* **Processing Latency:** $< 50$ms average query time per truck.
* **Robustness:** $< 2.0\%$ standard deviation in savings regardless of the order in which trucks enter the system.

---
