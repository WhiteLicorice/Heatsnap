"""
nws_labels.py
Module containing logic for NWS Heat Index Risk classification.

LIMITATIONS: 
Due to the rarity of Heat Index values above 125°F in the Skyfinder dataset, 
Categories 3 (Danger) and 4 (Extreme Danger) have been collapsed into a single
class (Class 3) to ensure statistical significance.
"""

from typing import List

# NWS Thresholds (Truncated to 4 bins: Safe, Caution, Ex. Caution, Danger+)
# Reference: NWS Heat Index Safety Guidelines. https://www.weather.gov/safety/heat-index
THRESHOLDS: List[int] = [80, 91, 104]

# Bin Centers for Virtual MAE calculation (Fahrenheit)
# These represent the 'average' temperature for each categorical prediction.
BIN_CENTERS: List[float] = [70.0, 85.5, 97.5, 115.0]

# Just four classes from 0-3 becaue we collapsed Categories 3 & 4 into one.
NUMBER_OF_CLASSES: int = 4

def get_nws_label(hi: float) -> int:
    """
    Maps continuous Heat Index to NWS Risk Categories. Note the limitation.
    
    0: Safe (<80)
    1: Caution (80-90)
    2: Extreme Caution (91-103)
    3: Danger/Extreme Danger (>=104)
    """
    for i, threshold in enumerate(THRESHOLDS):
        if hi < threshold:
            return i
    return len(THRESHOLDS)