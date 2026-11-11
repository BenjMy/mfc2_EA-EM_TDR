- [x]  ROPE inversion. How does it works? is it MonteCarlo as well??

- [x]  Currently no beta lateral smoothing implemented. do it. beta has no equivalent wiring at all — not in scenario_loader's schema, not in PARAM_GRID, nowhere. so lateral (within-epoch) smoothing via beta would actually take effect immediately once plumbed through — no restructuring needed, just the same kind of field-declaration + invert_kwargs addition alpha already has

- [x]  gamma is "wired" as a config field, but dead as an inversion parameter. scenario_loader.py declares it, validates it (gamma >= 0), loads it into cfg.gamma, and run_pipeline.py picks it up as GAMMA = cfg.gamma — it's even included in the ensemble-cache fingerprint (so changing gamma correctly invalidates cached runs) and printed in the scenario summary. But GAMMA is never actually put into invert_kwargs:

- [x] can you add a plot showing the % of differences between water_content_xz_2d_dense_tdr.png and water_content_xz_2d_dense_tdr_truth.png


- when MCMC raise error if beta and gamma are set since emagpy MCMC does not accept them	

- why even dense TDR has success with quality gates warmings? it should be perfectly resolve isnt it?

- Looks like results are super dependant from initial model. Can we vary it during the ensmble generation? | `initial conductivity ranges` | `[[0.01, 60.0], [0.01, 60.0], [0.01, 60.0], [0.01, 60.0]]` |



- can you plot the initial model used for the inversion versus the true ECa in the same style than: water_content_xz_2d.png . let me know if you understand the question


