"""
data_generator.py

Generates synthetic disaster reports so you don't need a real data source
(there isn't one you can access anyway). Two ways to use it:

  1. As a library: `generate_reports(n)` returns a list of report dicts.
  2. As a script: `python data_generator.py 50` writes 50 reports to
     sample_reports.json for inspection.

Coordinates are randomized around a fixed "disaster zone" center so reports
cluster realistically instead of being scattered across the whole planet
(makes the map demo look sane and lets duplicate-detection actually trigger).
"""

import json
import random
import sys

# Center of the simulated disaster zone (edit this to any city's coordinates)
CENTER_LAT = 13.0827   # Chennai, since VIT is nearby
CENTER_LNG = 80.2707
SPREAD = 0.08  # roughly a few km radius

REPORT_TYPES = ["road_block", "hospital_status", "victim_triage"]

DESCRIPTIONS = {
    "road_block": [
        "Main bridge collapsed, road impassable",
        "Debris blocking two-lane highway",
        "Flooding has submerged the underpass",
    ],
    "hospital_status": [
        "ICU beds full, only general ward available",
        "Emergency room operating at capacity",
        "Received extra generator, can take more patients",
    ],
    "victim_triage": [
        "Family of four trapped on rooftop",
        "Elderly person with breathing difficulty",
        "Minor injuries, ambulatory, needs transport",
    ],
}


def _random_location():
    return {
        "lat": round(CENTER_LAT + random.uniform(-SPREAD, SPREAD), 6),
        "lng": round(CENTER_LNG + random.uniform(-SPREAD, SPREAD), 6),
    }


def generate_reports(n: int) -> list[dict]:
    reports = []
    for _ in range(n):
        report_type = random.choice(REPORT_TYPES)
        reports.append({
            "type": report_type,
            "location": _random_location(),
            "severity": random.randint(1, 5),
            "description": random.choice(DESCRIPTIONS[report_type]),
        })
    return reports


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    data = generate_reports(count)
    with open("sample_reports.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {count} synthetic reports to sample_reports.json")
