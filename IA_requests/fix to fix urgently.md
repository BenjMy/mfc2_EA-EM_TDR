# Propagating EC-ensemble uncertainty to θ (volumetric water content)

Currently COverage and CV have nothing to deal with TDR sensors! they are just proxies of ensemble spread and model reliability

# Need a "how many EM points are calibrated near this cell"
extract_sample_positions_from_survey computes, for every TDR probe, a SamplePositionResult with an indices field — literally "the EM measurement array indices that fall within window_m of this probe" (i.e. the actual EM points already treated as "this probe's data" for calibration purposes, not just the probe's own coordinate). That's a hard window, not a smooth kernel, and it's exactly "how many EM EC points are calibrated [i.e. matched to real TDR data] around here":
That's arguably the more natural proxy for this pipeline, because it reuses the exact same window-matching logic that decides which EM points get folded into calibrate_petrophysical_relationship in the first place — no new distance metric to justify or tune a bandwidth for. My kernel version is really just a smoothed generalization of that (hard cutoff → tapered falloff), useful mainly if you want a continuous map instead of a step function at window_m. Want me to add the hard-window count version too (or swap it in as the primary one, since it's more directly tied to what actually feeds the calibration)?


# 2d map fix, does Emagpy account for xy variation (since inverison is 1d)
see .md corresponding file
