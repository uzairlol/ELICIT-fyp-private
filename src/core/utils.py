#utils.py

import json
import logging
import re


def uses_climate_budget():
    """Shared check for whether the simulation is in climate/LDF budget mode."""
    from core import parameters
    scenario_name = str(getattr(parameters, 'SCENARIO', '')).lower()
    if scenario_name == 'climate':
        scenario_name = 'ldf'
    return (
        scenario_name == 'ldf'
        or bool(getattr(parameters, 'CLIMATE_SHOCK_ENABLED', False))
        or bool(getattr(parameters, 'LDF_ENABLED', False))
    )


def robust_json_loads(json_str):
    """
    Loads JSON robustly.

    Accepts:
    - a dict already returned by an LLM client
    - a JSON string
    - bytes/bytearray

    If direct parsing fails, attempts to extract the first {...} block and parse that.
    Raises ValueError on failure (preserving previous behavior).
    """
    try:
        # Already-parsed dict (some clients return Python objects)
        if isinstance(json_str, dict):
            return json_str

        # Decode bytes
        if isinstance(json_str, (bytes, bytearray)):
            json_str = json_str.decode('utf-8', errors='replace')

        # Coerce non-string to string
        if not isinstance(json_str, str):
            json_str = str(json_str)

        # Try direct JSON parse
        return json.loads(json_str)

    except Exception as e:
        logging.debug(f"robust_json_loads: direct json.loads failed: {e}")
        # Try to extract the first JSON-like object { ... }
        try:
            m = re.search(r'(\{(?:.|\n)*\})', json_str, flags=re.DOTALL)
            if m:
                candidate = m.group(1)
                return json.loads(candidate)
        except Exception as e2:
            logging.debug(f"robust_json_loads: extraction parse failed: {e2}")

        logging.error(f"JSON parse failure. Raw LLM string (first 400 chars): {json_str[:400]}")
        raise ValueError(f"Failed to parse explicitly requested JSON format. {e}")
