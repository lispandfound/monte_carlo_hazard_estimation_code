single_site_cbgs_inputs:
    uv run single_site_single_im/generate_rupture_df.py data/nshmdb.db "-43.52934701" 172.6198762 197 data/CBGS_inputs.parquet
single_site_cbgs_gmm im:
    uv run single_site_single_im/atkinson_model.py data/CBGS_inputs.parquet {{im}} data/GBGS_gmm.parquet
