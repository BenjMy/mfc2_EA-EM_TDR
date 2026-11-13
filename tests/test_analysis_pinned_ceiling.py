import os
import numpy as np
import analysis_pinned_ceiling as apc


def _ens(tradeoff, N=60, T=2, P=6, L=4, seed=1):
    rng = np.random.default_rng(seed)
    truth = 50 + 10 * rng.random((T, P, L))
    e0 = rng.normal(0, 8, (N, 1, 1))
    noise = rng.normal(0, 2, (N, T, P, L))
    d = noise.copy()
    d[..., 0] += e0
    d[..., 1] += (-1.0 if tradeoff else 0.0) * e0    # layer-1 error mirrors layer 0
    return truth[None] + d, truth


def test_tradeoff_detected_and_absent():
    m, t = _ens(True)
    cell, memb = apc.layer0_vs_deep_correlation(m, t)
    assert cell[1] < -0.8 and memb[1] < -0.8
    assert abs(cell[2]) < 0.3                       # untouched layer: no correlation
    m, t = _ens(False)
    cell, _ = apc.layer0_vs_deep_correlation(m, t)
    assert abs(cell[1]) < 0.3


def test_oracle_beats_random_on_layer0_and_k_logic():
    m, t = _ens(True)
    rank = np.arange(len(m)); mr = np.linspace(0.01, 0.05, len(m))
    res, band, top, orc = apc.stack_comparison(m, t, rank, mr, K=10, n_random=300)
    assert res["oracle"][0][0] < res["random"][0][0]          # layer 0 selected on truth
    assert res["oracle"][0][1] < res["random"][0][1]          # trade-off -> helps layer 1 too
    assert len(top) == len(orc) == 10
    assert (band[0] <= res["random"][0] + 1e-9).all() and (res["random"][0] <= band[1] + 1e-9).all()


def test_stack_all_equals_mean_and_k_clipped():
    m, t = _ens(False, N=5)
    res, *_ = apc.stack_comparison(m, t, np.arange(5), np.ones(5), K=99, n_random=5)
    np.testing.assert_allclose(res["all"][0], res["tdr"][0], rtol=1e-9)   # K clipped to N, equal weights


def test_run_and_write_outputs(tmp_path):
    m, t = _ens(True, N=20)
    om = tmp_path / "output_model"; om.mkdir()
    np.savez(om / "ensemble_members.npz", models_clean=m)
    np.save(om / "synthetic_time_lapse_ec.npy", t[None])
    np.save(om / "member_ranking.npy", np.arange(20)); np.save(om / "member_rmse.npy", np.linspace(.01, .03, 20))
    np.save(om / "K.npy", np.array(5))
    r = apc.run(str(tmp_path), n_random=50)
    out = apc.write_outputs(r, str(tmp_path))
    for f in ("stack_errors.csv", "correlations.csv", "summary.md", "fig_pinned_ceiling.png"):
        assert os.path.getsize(os.path.join(out, f)) > 0
