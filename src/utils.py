# src/utils.py
"""
Utility Functions

General-purpose helper functions for data processing and conversion.
"""

import isodate
import numpy as np

def iso_duration_to_minutes(duration):
    """
    Convert ISO 8601 duration format to minutes.
    
    Args:
        duration: ISO 8601 duration string (e.g., 'PT4M13S')
        
    Returns:
        Duration in minutes as float, or NaN if conversion fails
    """
    try:
        return isodate.parse_duration(duration).total_seconds() / 60
    except:
        return np.nan
