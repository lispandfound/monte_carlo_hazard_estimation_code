"""Pre-compute rupture dataframes for a given site."""

from functools import partial
from pathlib import Path
from typing import TypedDict

import cyclopts
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
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


def extract_fault_info(db: NSHMDB) -> dict[str, FaultInfo]:
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
        cursor.execute(
            """SELECT p.name, f.* FROM fault f JOIN parent_fault p ON f.parent_id = p.parent_id"""
        )
        fault_rows = cursor.fetchall()
        return {row[0]: FaultInfo(*row[1:]) for row in fault_rows}


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
    area: float
    rake: float
    rate: float


class SiteRow(TypedDict):
    """Data transfer object for ruptures into DataFrame."""

    rupture_id: int
    site: str
    rrup: float


def _process_single_rupture(rupture: Rupture, sites: list[Site]) -> list[SiteRow]:
    """Worker function to process all sites for a specific rupture."""
    rows = []
    for site in sites:
        # Calculate minimum rrup across all faults for this site
        rrup = min(
            fault.rrup_distance(np.array([site.lat, site.lon, 0.0]))
            for fault in rupture.faults.values()
        )
        rrup /= 1000.0  # Convert to km
        rows.append(SiteRow(rupture_id=rupture.rupture_id, site=site.name, rrup=rrup))
    return rows


def compile_source_to_site_dataframe(nshmdb: NSHMDB, sites: list[Site]) -> pd.DataFrame:
    ruptures = extract_all_ruptures(nshmdb)

    # Use partial to fix the 'sites' argument for the worker function
    mapper = partial(_process_single_rupture, sites=sites)

    # process_map handles the Pool creation and the progress bar
    # chunksize can be adjusted based on the number of ruptures
    results = process_map(mapper, ruptures, chunksize=10)

    # Flatten the list of lists into a single list of SiteRow objects
    flattened_rows = [row for sublist in results for row in sublist]

    return pd.DataFrame(flattened_rows).set_index("rupture_id")


def compile_rupture_dataframe(nshmdb: NSHMDB) -> gpd.GeoDataFrame:
    ruptures = extract_all_ruptures(nshmdb)
    fault_info = extract_fault_info(nshmdb)
    rows = []
    geometries = []
    for rupture in ruptures:
        rakes = [fault_info[fault].rake for fault in rupture.faults]
        areas = [fault.area() for fault in rupture.faults.values()]
        rad_rakes = np.deg2rad(rakes)
        cos_rake = np.cos(rad_rakes)
        sin_rake = np.sin(rad_rakes)
        mean_rake_vec = np.average(
            np.array([cos_rake, sin_rake]), axis=-1, weights=areas
        )
        mean_rake = np.rad2deg(np.arctan2(mean_rake_vec[1], mean_rake_vec[0]))
        geometries.append(
            shapely.union_all([fault.geometry for fault in rupture.faults.values()])
        )
        rows.append(
            RuptureRow(
                rupture_id=rupture.rupture_id,
                mag=rupture.magnitude,
                area=rupture.area,
                rake=mean_rake,
                rate=rupture.rate,
            )
        )
    return gpd.GeoDataFrame(rows, geometry=geometries, crs="2193").set_index(
        "rupture_id"
    )


@app.command
def source_to_site(nshmdb_path: Path, site_path: Path, output_path: Path):
    nshmdb = NSHMDB(nshmdb_path)
    site_df = gpd.read_parquet(site_path).to_crs(4326)
    sites = [
        Site(
            name=name, vs30=prop["vs30"], lat=prop["geometry"].y, lon=prop["geometry"].x
        )
        for name, prop in site_df.iterrows()
    ]
    source_to_site_df = compile_source_to_site_dataframe(nshmdb, sites)
    source_to_site_df.to_parquet(output_path)


@app.command
def source_model(
    nshmdb_path: Path,
    output_path: Path,
):
    """Build input rupture dataframe for the single site, im, model experiment.

    Parameters
    ----------
    nshmdb_path : Path
        Path to NSHMDB database.
    output_path : Path
        Output path (parquet).
    """
    nshmdb = NSHMDB(nshmdb_path)
    df = compile_rupture_dataframe(nshmdb)
    df.to_parquet(output_path)


if __name__ == "__main__":
    app()
