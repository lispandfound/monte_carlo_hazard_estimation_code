
single_site_cbgs_inputs:
    uv run single_site_single_im/generate_rupture_df.py data/nshmdb.db "-43.52934701" 172.6198762 197 data/CBGS_inputs.parquet
single_site_cbgs_gmm im:
    uv run single_site_single_im/atkinson_model.py data/CBGS_inputs.parquet {{im}} data/CBGS_gmm.parquet
test:
    uv run --extra test pytest -s tests/

poisson_strategy:
    uv run single_site_single_im/monte_carlo_experiment.py poisson data/CBGS_gmm.parquet figures/poisson_strategy.png --length 200000

naive_strategy:
    uv run single_site_single_im/monte_carlo_experiment.py naive data/CBGS_gmm.parquet figures/naive_strategy.png --n 10


compare_strategies:
    uv run single_site_single_im/monte_carlo_experiment.py compare data/strategies.toml data/CBGS_gmm.parquet figures/strategy_comparison.png --subwidth 6 --subheight 6
