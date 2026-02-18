import tomllib
from pathlib import Path
from typing import Any, NamedTuple

import cyclopts
import numpy as np
import pandas as pd
import tqdm
from nshmdb.nshmdb import NSHMDB
from sklearn.cluster import OPTICS
from source_modelling.sources import Plane

app = cyclopts.App()

Array1 = np.ndarray[tuple[int,], np.dtype[np.float64]]
EmbeddingMatrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]


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


def count_fault_planes(db: NSHMDB) -> int:
    """Count the number of fault planes in the database.

    Parameters
    ----------
    db : NSHMDB
        The nshmdb to read from.

    Returns
    -------
    int
        The number of fault planes in the DB.
    """
    with db.connection() as conn:
        return conn.execute("SELECT COUNT(fault_id) from fault_plane").fetchone()[0]


def extract_fault_vector(db: NSHMDB, vector_length: int, rupture_id: int) -> Array1:
    fault_vector = np.zeros(vector_length, dtype=np.float64)
    with db.connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT fs.*
                FROM fault_plane fs
                JOIN rupture_faults rf ON fs.fault_id = rf.fault_id
                JOIN fault f ON fs.fault_id = f.fault_id
                JOIN parent_fault p ON f.parent_id = p.parent_id
                WHERE rf.rupture_id = ?
                ORDER BY f.parent_id""",
            (rupture_id,),
        )
        fault_planes = cursor.fetchall()
        for (
            fault_id,
            top_left_lat,
            top_left_lon,
            top_right_lat,
            top_right_lon,
            bottom_right_lat,
            bottom_right_lon,
            bottom_left_lat,
            bottom_left_lon,
            top,
            bottom,
            _,
        ) in fault_planes:
            corners = np.array(
                [
                    [top_left_lat, top_left_lon, top],
                    [top_right_lat, top_right_lon, top],
                    [bottom_right_lat, bottom_right_lon, bottom],
                    [bottom_left_lat, bottom_left_lon, bottom],
                ]
            )
            plane = Plane.from_corners(corners)
            fault_vector[fault_id - 1] = plane.area
        return fault_vector


class Embedding(NamedTuple):
    rupture_ids: Array1
    embedding: EmbeddingMatrix


def read_embedding_from_file(file: Path) -> Embedding:
    datastore = np.load(file)
    return Embedding(datastore["rupture_ids"], datastore["embedding"])


def extract_all_fault_vectors(db: NSHMDB) -> Embedding:
    rupture_ids = list(get_rupture_ids(db))
    n_planes = count_fault_planes(db)
    vectors = []
    for rupture in tqdm.tqdm(rupture_ids, desc="Extracting fault vectors"):
        vectors.append(extract_fault_vector(db, n_planes, rupture))
    embedding_matrix = np.stack(vectors, axis=0)
    return Embedding(np.array(rupture_ids), embedding_matrix)


def cluster_embedding(embedding: Embedding, **kwargs: Any) -> pd.DataFrame:
    clustering = OPTICS(metric="braycurtis", n_jobs=-1, **kwargs)
    clustering.fit(embedding.embedding)
    return pd.DataFrame(
        {
            "label": clustering.labels_,
            "reachability": clustering.reachability_,
            "ordering": clustering.ordering_,
        },
        index=embedding.rupture_ids,
    )


def read_cluster_settings(cluster_settings: Path) -> dict["str", Any]:
    with open(cluster_settings, "rb") as f:
        return tomllib.load(f)


@app.command
def generate(db_path: Path, output_path: Path) -> None:
    db = NSHMDB(db_path)
    embedding = extract_all_fault_vectors(db)
    np.savez_compressed(
        output_path, rupture_ids=embedding.rupture_ids, embedding=embedding.embedding
    )


@app.command
def cluster(embedding_path: Path, cluster_settings: Path, output_path: Path) -> None:
    embedding = read_embedding_from_file(embedding_path)
    settings = read_cluster_settings(cluster_settings)
    clustering = cluster_embedding(embedding, **settings)
    clustering.to_parquet(output_path)


if __name__ == "__main__":
    app()
