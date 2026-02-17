
rupture_inputs:
    uv run hazard_estimation/generate_rupture_df.py source-model data/nshmdb.db data/inputs.parquet
source_to_site:
    uv run hazard_estimation/generate_rupture_df.py source-to-site data/nshmdb.db data/selected_stations.parquet data/source_to_site.parquet

single_site_cbgs_gmm im:
    uv run hazard_estimation/atkinson_model.py data/CBGS_inputs.parquet {{im}} data/CBGS_gmm.parquet
test:
    uv run --extra test pytest -s tests/

poisson_strategy:
    uv run hazard_estimation/monte_carlo_experiment.py poisson data/CBGS_gmm.parquet figures/poisson_strategy.png --length 200000

naive_strategy:
    uv run hazard_estimation/monte_carlo_experiment.py naive data/CBGS_gmm.parquet figures/naive_strategy.png --n 10


compare_strategies strategy="strategies.toml" output="figures/strategy_comparison.png":
    uv run hazard_estimation/monte_carlo_experiment.py compare {{strategy}} data/CBGS_gmm.parquet {{output}} --subwidth 6 --subheight 6
