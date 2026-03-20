
rupture_inputs:
    python hazard_estimation/generate_rupture_df.py source-model data/nshmdb.db data/inputs.parquet
source_to_site stations="data/selected_stations.parquet":
    python hazard_estimation/generate_rupture_df.py source-to-site data/nshmdb.db {{stations}} data/inputs_source_to_site.parquet

single_site_cbgs_gmm im:
    hazard_estimation/atkinson_model.py data/CBGS_inputs.parquet {{im}} data/CBGS_gmm.parquet
test:
    --extra test pytest -s tests/

poisson_strategy:
    hazard_estimation/monte_carlo_experiment.py poisson data/CBGS_gmm.parquet figures/poisson_strategy.png --length 200000

naive_strategy:
    hazard_estimation/monte_carlo_experiment.py naive data/CBGS_gmm.parquet figures/naive_strategy.png --n 10


compare_strategies strategy="strategies.toml" output="figures/strategy_comparison.png":
    hazard_estimation/monte_carlo_experiment.py compare {{strategy}} data/CBGS_gmm.parquet {{output}} --subwidth 6 --subheight 6


plot_all_site_metrics:
    hazard_estimation/disaggregation.py evaluate-monte-carlo-sample ./data/monte_carlo_hazard_30000_100r_cs_nz_density.h5 'Cybershake NZ' ./data/ruptures.parquet data/south_island_sites.parquet ./data/total_hazard_recalc_mar_13.h5 figures/monte_carlo_sample_30000_cs APPS CBGS DCDS ICCS KIKS NELS QTPS
    hazard_estimation/disaggregation.py evaluate-monte-carlo-sample ./data/monte_carlo_hazard_30000_100r_equal_density.h5 'Equal Density' ./data/ruptures.parquet data/south_island_sites.parquet ./data/total_hazard_recalc_mar_13.h5 figures/monte_carlo_sample_30000_equal APPS CBGS DCDS ICCS KIKS NELS QTPS
    hazard_estimation/disaggregation.py evaluate-monte-carlo-sample ./data/monte_carlo_hazard_30000_100r_optimal_density.h5 'Optimal Density' ./data/ruptures.parquet data/south_island_sites.parquet ./data/total_hazard_recalc_mar_13.h5 figures/monte_carlo_sample_30000_optimal APPS CBGS DCDS ICCS KIKS NELS QTPS
    hazard_estimation/disaggregation.py evaluate-monte-carlo-sample ./data/monte_carlo_hazard_30000_100r_rate_density.h5 'Rate Density' ./data/ruptures.parquet data/south_island_sites.parquet ./data/total_hazard_recalc_mar_13.h5 figures/monte_carlo_sample_30000_rate APPS CBGS DCDS ICCS KIKS NELS QTPS
    hazard_estimation/disaggregation.py evaluate-monte-carlo-sample ./data/monte_carlo_hazard_30000_100r_scec_density.h5 'SCEC Density' ./data/ruptures.parquet data/south_island_sites.parquet ./data/total_hazard_recalc_mar_13.h5 figures/monte_carlo_sample_30000_scec APPS CBGS DCDS ICCS KIKS NELS QTPS

compare_methods:
    streamlit run hazard_estimation/spatial_comp.py -- ./data/south_island_sites.parquet ./data/monte_carlo_30000_100r_error_all.h5
