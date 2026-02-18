from dataclasses import dataclass


@dataclass
class Site:
    """Site location."""

    name: str
    """Name of site."""
    lat: float
    "Latitude of site"
    lon: float
    "Longitude of site"
    vs30: float
    "Average shear-wave velocity in the top 30m of soil (Vs30)"
