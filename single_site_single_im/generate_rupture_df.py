"""Pre-compute rupture dataframes for a given site."""

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import cyclopts
import numpy as np
import pandas as pd
import tqdm
from nshmdb.nshmdb import NSHMDB, Rupture

app = cyclopts.App()


def get_rupture_ids(db: NSHMDB) -> set[int]:
    """Get all rupture IDs with non-null rates.

    Parameters
    ----------
    db : NSHMDB
        The nshmdb to read from.

    Returns
    -------
    set[int]
        The set of rupture rates.
    """
    with db.connection() as conn:
        return {
            fault_id
            for (fault_id,) in conn.execute(
                "SELECT rupture_id FROM rupture where rate > 0"
            ).fetchall()
        }


def extract_all_ruptures(db: NSHMDB) -> list[Rupture]:
    """Extract all rupture objects from the nshmdb database.

    Parameters
    ----------
    db : NSHMDB
        Database to extract from.

    Returns
    -------
    list[Rupture]
        List of ruptures in the database.
    """
    rupture_ids = get_rupture_ids(db)
    ruptures = []
    failed_ids = []
    pbar = tqdm.tqdm(rupture_ids, desc="Extracting ruptures", unit="ruptures")
    for rupture in pbar:
        try:
            ruptures.append(db.get_rupture(rupture))
        except ValueError:
            failed_ids.append(rupture)
            pbar.set_postfix_str(f"{len(failed_ids)} failures")

    print(
        f"Could not extract ruptures from the following ids: {','.join(str(id) for id in failed_ids)}"
    )
    return ruptures


@dataclass
class Site:
    """Site location."""

    lat: float
    "Latitude of site"
    lon: float
    "Longitude of site"
    vs30: float
    "Average shear-wave velocity in the top 30m of soil (Vs30)"


def measure_site_distance(rupture: Rupture, site: Site) -> float:
    """Measure rupture-site distance.

    Parameters
    ----------
    rupture : Rupture
        Rupture to measure from.
    site : Site
        Site to measure to.

    Returns
    -------
    float
        Rrup from source to site (metres).
    """
    point = np.array([site.lat, site.lon, 0.0])  # Assumed at surface.
    return min(fault.rrup_distance(point) for fault in rupture.faults.values())


class RuptureRow(TypedDict):
    """Data transfer object for ruptures into DataFrame."""

    rupture_id: int
    mag: float
    vs30: float
    rrup: float


def compile_rupture_dataframe(db: NSHMDB, site: Site) -> pd.DataFrame:
    """Compile the rupture dataframe for a site.

    Parameters
    ----------
    db : NSHMDB
        The NSHMDB to extract from.
    site : Site
        The site to compute for.

    Returns
    -------
    pd.DataFrame
        A dataframe containing rupture id, magnitude, vs30, rrup and
        rupture rate.
    """
    rupture_rows = []
    ruptures = extract_all_ruptures(db)
    print(f"Found {len(ruptures)} ruptures to add")
    for rupture in tqdm.tqdm(ruptures, desc="Measuring distances", unit="ruptures"):
        rrup_metres = measure_site_distance(rupture, site)
        rrup_kilomtres = rrup_metres / 1000.0
        rupture_rows.append(
            RuptureRow(
                rupture_id=rupture.rupture_id,
                mag=rupture.magnitude,
                vs30=site.vs30,
                rrup=rrup_kilomtres,
            )
        )

    return pd.DataFrame(rupture_rows).set_index("rupture_id")


@app.default
def build_input_rupture_dataframe(
    nshmdb_path: Path,
    site_lat: float,
    site_lon: float,
    site_vs30: float,
    output_path: Path,
):
    """Build input rupture dataframe for the single site, im, model experiment.

    Parameters
    ----------
    nshmdb_path : Path
        Path to NSHMDB database.
    site_lat : float
        Site latitude.
    site_lon : float
        Site longitude.
    site_vs30 : float
        Site Vs30.
    output_path : Path
        Output path (parquet).
    """
    nshmdb = NSHMDB(nshmdb_path)
    site = Site(lat=site_lat, lon=site_lon, vs30=site_vs30)
    df = compile_rupture_dataframe(nshmdb, site)
    df.to_parquet(output_path)


if __name__ == "__main__":
    app()
