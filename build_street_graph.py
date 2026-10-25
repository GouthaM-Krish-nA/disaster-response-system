"""
build_street_graph.py

Run this ONCE (not every time the server starts) to download Chennai's real
street network and save it to a file. routing.py will load that saved file
instead of hitting the internet on every server restart.

This needs internet access and can take 30 seconds to a couple of minutes,
depending on your connection.
"""

import osmnx as ox

# Same center point and coverage area as data_generator.py, so the street
# graph covers the exact zone our synthetic reports are generated in.
CENTER_LAT = 13.0827
CENTER_LNG = 80.2707

# Radius in meters. SPREAD in data_generator.py is 0.08 degrees, which is
# roughly 8.9 km — 9000m comfortably covers that whole area from the center.
RADIUS_METERS = 9000

print("Downloading street network... this can take a minute.")
G = ox.graph_from_point(
    (CENTER_LAT, CENTER_LNG),
    dist=RADIUS_METERS,
    network_type="drive",   # only roads vehicles can use, not footpaths
)

ox.save_graphml(G, filepath="chennai_streets.graphml")
print(f"Saved! Graph has {len(G.nodes)} intersections and {len(G.edges)} road segments.")