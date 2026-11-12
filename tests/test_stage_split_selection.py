"""Core claim behind --stage map: running Steps 2-5 again on ONLY the members the
calibration selected (on a denser point set that still contains the probe points)
re-derives the same K and the same Waxman-Smits parameters.

Uses the real step2/step3/step4 functions on a synthetic ensemble (no emagpy)."""
import itertools

import numpy as np
import pytest

import em_methodology as EMmethod
import stage_split as ss
import step2_petro
import step3_average
import step4_select

DEPTHS0 = [0.225]
WS_TRUE = {0: (0.6, 1.5, 300.0, 10.0), 1: (0.7, 1.6, 250.0, 20.0)}   # a, m, sigma_w, sigma_s
FIXED = {"a": True, "m": True, "sigma_w": False, "sigma_s": False}
ALPHAS = [0.01, 0.07, 0.5]
FMS = ["FSlin", "CS"]
CONDS = [[0.01, 0.01], [300.0, 300.0]]
N_T, N_PROBE = 2, 8


def _ws_sigma(theta, a, m, sw, ss):
    return (theta ** m / a) * (sw + ss)


def _run_steps_2_to_5(models, pos, times, theta_obs, layers, outdir):
    kw = dict(sample_positions=pos, sample_times=times, theta_obs=theta_obs,
              probe_layer_indices=layers, depths_are_fixed=True, fixed_params=FIXED)
    fits2 = step2_petro.run_petro(models=models, depths0=DEPTHS0, output_dir=str(outdir),
                                  generate_plots=False, **kw)
    averaged, ranking = step3_average.run_average(models=models, fits_step2=fits2, depths0=DEPTHS0,
                                                  output_dir=str(outdir), generate_plots=False)
    fits4 = EMmethod.recalibrate_petrophysical_relationship(
        averaged_models=averaged, depths0=DEPTHS0, **kw)
    sel = step4_select.run_select_assess(models=models, averaged_models=averaged, ranking=ranking,
                                         fits_step2=fits4, depths0=DEPTHS0, output_dir=str(outdir),
                                         generate_plots=False, sample_positions=pos,
                                         sample_times=times, theta_obs=theta_obs,
                                         probe_layer_indices=layers)
    return fits2, ranking, fits4, sel


def _ensemble(seed, n_points):
    """(members, T, n_points, 2 layers); probe points are the first N_PROBE columns,
    members differ in noise level so the ranking is meaningful."""
    rng = np.random.default_rng(seed)
    combos = [(a, 0.0, 0.0, "ROPE", f, list(c))
              for a, f, c in itertools.product(ALPHAS, FMS, CONDS)]
    n_m = len(combos)
    pos = list(range(N_PROBE)) * N_T
    times = [t for t in range(N_T) for _ in range(N_PROBE)]
    layers = [i % 2 for i in range(N_PROBE)] * N_T
    theta_obs = rng.uniform(0.12, 0.38, len(pos))
    noise = rng.uniform(0.01, 0.35, n_m)                 # per-member quality
    models = rng.uniform(5.0, 80.0, (n_m, N_T, n_points, 2))
    for i, (p, t, l) in enumerate(zip(pos, times, layers)):
        truth = _ws_sigma(theta_obs[i], *WS_TRUE[l])
        models[:, t, p, l] = truth * (1 + noise * rng.standard_normal(n_m))
    return combos, models, np.array(pos), np.array(times), np.array(layers), theta_obs


def _meta(combos):
    return [dict(alpha=a, beta=b, gamma=g, method=m, forwardModel=f, conds0=c)
            for a, b, g, m, f, c in combos]


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_map_run_on_selected_members_rederives_calibration(seed, tmp_path):
    combos, models, pos, times, layers, theta_obs = _ensemble(seed, n_points=N_PROBE)
    fits2, ranking, fits4, sel = _run_steps_2_to_5(models, pos, times, theta_obs, layers, tmp_path)

    rec = ss.build_calibration_record(
        scenario="s", run_id="r", output_dir=str(tmp_path), ranking=ranking,
        param_metadata=_meta(combos), rmse_by_member=[f.rmse for f in fits2],
        selected_K=sel.K, selected_rmse=sel.rmse,
        ws_by_horizon=ss.ws_rows_from_fit(fits4[sel.K - 1].horizon_fits),
        localized_bandwidth=None, bandwidth_from_sweep=False, depths0=DEPTHS0,
        cond_ranges=[[0.01, 300.0]] * 2, em_files=["a.csv", "b.csv"],
        em_subsample={"mode": "probes", "radius_m": 5.0}, quick=False, n_probe_obs=len(theta_obs))

    # ---- map run: only the selected members, on a MUCH denser point set that still
    # contains the probe points at the same indices (extra points are non-probe).
    kept, missing = ss.select_members(combos, rec)
    assert missing == [] and len(kept) == sel.K
    keep_idx = [combos.index(c) for c in kept]
    extra = np.random.default_rng(99).uniform(5.0, 80.0, (len(keep_idx), N_T, 500, 2))
    models_map = np.concatenate([models[keep_idx], extra], axis=2)
    fits2m, rankingm, fits4m, selm = _run_steps_2_to_5(models_map, pos, times, theta_obs, layers, tmp_path)

    assert selm.K == sel.K
    ws_new = ss.ws_rows_from_fit(fits4m[selm.K - 1].horizon_fits)
    drift = ss.drift_report(rec, selm.K, ws_new, rel_tol=1e-6)
    assert not drift["warn"], drift
    assert selm.rmse == pytest.approx(sel.rmse, rel=1e-9)
    # same members in the same rank order
    assert [combos.index(kept[i]) for i in rankingm[:sel.K]] == [
        keep_idx[i] for i in rankingm[:sel.K]]


def test_selection_is_not_trivial_across_seeds(tmp_path):
    """Guard against a degenerate test: K should not be 1 for every seed."""
    Ks = []
    for seed in range(6):
        combos, models, pos, times, layers, theta_obs = _ensemble(seed, N_PROBE)
        *_, sel = _run_steps_2_to_5(models, pos, times, theta_obs, layers, tmp_path)
        Ks.append(sel.K)
    assert max(Ks) > 1, Ks
