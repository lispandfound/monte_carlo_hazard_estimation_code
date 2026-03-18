import pandas as pd
import geopandas as gpd
import numpy as np
import re
from pathlib import Path

def load_hazard_file(file_path: Path) -> gpd.DataFrame:
    """
    Loads an individual OpenQuake hazard curve CSV, converts POE to Rate,
    and extracts the SA period as a float.
    """
    filename = file_path.name
    
    # 1. Extract metadata
    # Identifies 'mean' or 'rlz-###'
    rlz_match = re.search(r'hazard_curve-(mean|rlz-\d+)', filename)
    rlz = rlz_match.group(1) if rlz_match else "unknown"
    
    # Extract the numeric value inside SA(...)
    # e.g., "hazard_curve-mean-SA(0.12)_10.csv" -> 0.12
    sa_match = re.search(r'SA\((.*?)\)', filename)
    if not sa_match:
        return None # Should not happen with current glob filter
    
    period_float = float(sa_match.group(1))

    # 2. Read and Process
    # skip_rows=1 to bypass the OQ metadata header
    df = pd.read_csv(file_path, skiprows=1)
    
    # Identify POE columns
    values = [col for col in df.columns if col.startswith('poe-')]
    
    # Melt to long format
    df_melted = df.melt(
        id_vars=['lon', 'lat'], 
        value_vars=values, 
        var_name='threshold', 
        value_name='poe'
    )
    
    # Calculate Rate and clean threshold
    # Note: 50 is the investigation_time from your sample
    df_melted['rate'] = -np.log(np.maximum(1 - df_melted['poe'], 1e-6)) / 50
    df_melted['threshold'] = df_melted['threshold'].str.removeprefix('poe-').astype(float)
    
    # Add extracted metadata
    df_melted['rlz'] = rlz
    df_melted['period'] = period_float
    
    return df_melted

def consolidate_hazard_curves(directory: str, output_name: str = "consolidated_sa_hazard.parquet"):
    path = Path(directory)
    
    # Ignore PGA files by only globbing for SA patterns
    all_files = list(path.glob("hazard_curve-*-SA(*)*.csv"))
    
    if not all_files:
        print("No SA files found matching the pattern.")
        return None

    print(f"Found {len(all_files)} SA files. Ignoring PGA. Starting processing...")
    
    processed_data = []
    for i, f in enumerate(all_files):
        try:
            temp_df = load_hazard_file(f)
            if temp_df is not None:
                processed_data.append(temp_df)
            
            if i % 20 == 0:
                print(f"Processed {i}/{len(all_files)}...")
        except Exception as e:
            print(f"Error processing {f.name}: {e}")

    # Combine
    final_df = pd.concat(processed_data, ignore_index=True)
    
    # GeoDataFrame conversion
    gdf = gpd.GeoDataFrame(
        final_df, 
        geometry=gpd.points_from_xy(final_df['lon'], final_df['lat']),
        crs=4326
    ).drop(columns=['lon', 'lat'])

    # Save
    gdf.to_parquet(output_name)
    print(f"Successfully consolidated {len(all_files)} files into {output_name}")
    return gdf

if __name__ == "__main__":
    consolidate_hazard_curves(".")
