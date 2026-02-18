"""Pre-compute rupture dataframes for a given site."""

from functools import partial
from pathlib import Path
from typing import TypedDict

import cyclopts
import geopandas as gpd
import numpy as np
import pandas as pd
import tqdm
from nshmdb.nshmdb import NSHMDB, FaultInfo, Rupture
from tqdm.contrib.concurrent import process_map

from hazard_estimation.site import Site

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


def extract_fault_info(db: NSHMDB) -> dict[int, FaultInfo]:
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
    with db.connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""SELECT * FROM fault f""")
        fault_rows = cursor.fetchall()
        return {row[0]: FaultInfo(*row) for row in fault_rows}


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
    rrup: float
    area: float
    rake: float
    rate: float


def compile_rupture_dataframe(nshmdb: NSHMDB) -> pd.DataFrame:
    ruptures = extract_all_ruptures(nshmdb)
    fault_rows = extract_fault_info(nshmdb)
    rows = []
    for rupture in ruptures:
        rakes = [fault_rows[fault].rake for fault in rupture.faults]
        rows.append(RuptureRow())


@app.default
def build_input_rupture_dataframe(
    nshmdb_path: Path,
    sites_path: Path,
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
    df = compile_rupture_dataframe(nshmdb)
    df.to_parquet(output_path)


if __name__ == "__main__":
    app()
