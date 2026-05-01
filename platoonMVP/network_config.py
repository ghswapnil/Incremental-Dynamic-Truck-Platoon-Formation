"""
network_config.py — Hardcoded Bengaluru Hub-and-Spoke Topology

Defines the 7-hub corridor along NH275 (Bengaluru–Mysuru Expressway)
and provides a factory function to build the default HubSpokeNetwork.

Hubs can be added/removed by editing HUBS and ROUTE_SEGMENTS below.
"""

from platoon.road_network import HubSpokeNetwork

# ---------------------------------------------------------------------------
# Hub definitions (lat/lon for future OSM integration in Phase 5)
# ---------------------------------------------------------------------------
HUBS = {
    'Peenya':        {'lat': 13.0358, 'lon': 77.5200},
    'Kengeri':       {'lat': 12.9144, 'lon': 77.4850},
    'Bidadi':        {'lat': 12.7994, 'lon': 77.3819},
    'Ramanagara':    {'lat': 12.7159, 'lon': 77.2810},
    'Mandya':        {'lat': 12.5244, 'lon': 76.8950},
    'Srirangapatna': {'lat': 12.4214, 'lon': 76.6936},
    'Mysuru':        {'lat': 12.2958, 'lon': 76.6394},
}

# ---------------------------------------------------------------------------
# Directed segments — (start_hub, end_hub, distance_km)
# Only the "forward" direction is listed here; build_default_network()
# automatically adds the reverse direction as well.
# ---------------------------------------------------------------------------
ROUTE_SEGMENTS = [
    ('Peenya',     'Kengeri',       18),   # km
    ('Kengeri',    'Bidadi',        25),
    ('Bidadi',     'Ramanagara',    15),
    ('Ramanagara', 'Mandya',        50),
    ('Mandya',     'Srirangapatna', 20),
    ('Srirangapatna', 'Mysuru',     25),
]


def build_default_network() -> HubSpokeNetwork:
    """
    Construct the full bidirectional Bengaluru–Mysuru hub-and-spoke network.

    Returns:
        A HubSpokeNetwork with 7 hubs and 12 directed segments
        (6 forward + 6 reverse).
    """
    network = HubSpokeNetwork()

    for start_hub, end_hub, distance_km in ROUTE_SEGMENTS:
        # Forward direction
        network.add_segment(start_hub, end_hub, distance_km)
        # Reverse direction (same distance)
        network.add_segment(end_hub, start_hub, distance_km)

    return network
