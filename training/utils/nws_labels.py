"""
Module containing logic for Binary Heat Index Risk classification.
Maps continuous weather variables to actionable safety states.

LITERATURE CITATIONS:
- Physiological Threshold: Anderson, G. B., et al. (2013). "Methods to Calculate the 
  Heat Index as an Exposure Metric in Environmental Health Research." 
  Environmental Health Perspectives. https://ehp.niehs.nih.gov/doi/10.1289/ehp.1206273
- Operational Safety: Rothfusz, L. P. (1990). "The Heat Index Equation." 
  NWS Technical Attachment (SR 90-23). https://www.weather.gov/media/ffc/ta_htindx.PDF
"""

from typing import List, Final

# NWS Threshold for 'Caution': 80°F.
# In a Binary paradigm, any Heat Index >= 80.0 is 'Unsafe'.
THRESHOLDS: Final[List[int]] = [80]

# Bin Centers for Virtual MAE calculation (Fahrenheit)
# Used to ground-truth the 'semantic' error of the binary classifier.
BIN_CENTERS: Final[List[float]] = [70.0, 95.0]

NUMBER_OF_CLASSES: Final[int] = 2

def get_nws_label(hi: float) -> int:
    """
    Classifies Heat Index into binary categories.
    
    0: Safe (Apparent temperature < 80.0°F)
    1: Unsafe (Apparent temperature >= 80.0°F)
    """
    return 1 if hi >= THRESHOLDS[0] else 0