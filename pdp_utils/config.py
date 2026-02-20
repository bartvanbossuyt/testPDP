# Default lane setup
DEFAULT_LANE_SETUP = {
    "lanes": 3,
    "lane_width": 3.0,
    "offset": 0.0,
    "bounds": None,
    "description": "Standard 3-lane road"
}

# Configuration for lane drawing per 'c' value
LANE_CONFIGURATIONS: dict[int, dict] = {c: {**DEFAULT_LANE_SETUP} for c in range(0, 11)}
# Apply offset to Config 1 to center cars in lanes (shift road up by half lane width)
LANE_CONFIGURATIONS[1]["offset"] = 1.25
# Config 4 also needs this offset
LANE_CONFIGURATIONS[4] = {**DEFAULT_LANE_SETUP, "offset": 1.25}
# Also add Config 11 with the same offset, as it's often used as default
LANE_CONFIGURATIONS[11] = {**DEFAULT_LANE_SETUP, "offset": 1.25}
# Config 3: overtaking scenario with wider lanes
LANE_CONFIGURATIONS[3] = {**DEFAULT_LANE_SETUP, "lane_width": 5.0, "description": "Overtaking (wide lanes)"}

LANE_CONFIGURATIONS[12] = {**DEFAULT_LANE_SETUP, "bounds": {"x": (20, 470), "y": (-5, 5)}, "description": "3-lane road (horizontal)"}
LANE_CONFIGURATIONS[13] = {**DEFAULT_LANE_SETUP, "bounds": {"x": (20, 470), "y": (55, 145)}, "offset": 1.5, "description": "3-lane road for overtaking (2 cars)"}
LANE_CONFIGURATIONS[14] = {**DEFAULT_LANE_SETUP, "bounds": {"x": (20, 160), "y": (90, 110)}, "description": "3-lane road for overtaking (3 cars)"}
LANE_CONFIGURATIONS[15] = {**DEFAULT_LANE_SETUP, "bounds": {"x": (40, 220), "y": (165, 195)}, "offset": 4.0, "lane_width": 4.0, "description": "S-curve with overtaking"}
LANE_CONFIGURATIONS[16] = {
    "mode": "intersection",
    "lanes_horizontal": 3,
    "lanes_vertical": 3,
    "lane_width": 3.0,
    "center": (200, 100),
    "horizontal_range": (20, 470),
    "vertical_range": (20, 320),
    "bounds": {"x": (20, 470), "y": (20, 320)},
    "description": "Intersection (2 cars crossing)",
    "offset_horizontal": 3.0,
    "offset_vertical": -3.0,
}
LANE_CONFIGURATIONS[17] = {**DEFAULT_LANE_SETUP, "bounds": {"x": (55, 115), "y": (40, 125)}, "description": "Curved overtaking maneuver"}
LANE_CONFIGURATIONS[68] = {
    **DEFAULT_LANE_SETUP,
    "lanes": 2,
    "lane_width": 2.0,
    "offset": 0.0,
    "force_horizontal": True,
    "description": "Straight 2-lane overtaking maneuver",
}

