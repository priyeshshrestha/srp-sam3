"""Decision rule for the FLAIR band order, separated from I/O so it can be tested."""

from typing import Optional


def decide_band_order(soil_diff: Optional[float], water_diff: Optional[float],
                      margin: float = 3.0) -> Optional[str]:
    """Return 'rgb', 'bgr', or None when the evidence is missing, weak or contradictory.

    soil_diff / water_diff are the mean of (channel0 - channel2) over soil-like classes
    (bare soil, plowed land: reddish, so Red > Blue) and water-like classes (Blue >= Red),
    in 8-bit DN. A class group only votes when |diff| exceeds `margin`, and any disagreement
    between voters returns None: the caller must then ask a human to read the montage.
    Conservative on purpose, because a wrong automatic choice silently feeds every model a
    colour-swapped image.
    """
    votes = []
    if soil_diff is not None and abs(soil_diff) > margin:
        votes.append("rgb" if soil_diff > 0 else "bgr")
    if water_diff is not None and abs(water_diff) > margin:
        votes.append("rgb" if water_diff < 0 else "bgr")
    if votes and len(set(votes)) == 1:
        return votes[0]
    return None
