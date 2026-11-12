import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from data_real import clean_tdr_swc


def _df():
    t = pd.date_range("2024-05-01", periods=200, freq="h")
    rng = np.random.default_rng(0)
    x = 0.25 + 0.005 * rng.standard_normal(200)
    x[50] = 0.55          # spike inside range -> Hampel
    x[80] = -0.01         # below range
    x[120] = 90.0         # 0.90 after /100 -> above range
    x[150:160] = 25.0     # percent-unit samples = 0.25 -> must be KEPT
    return pd.DataFrame({"Timestamp": t, "Unit": "Unit 1", "SWC_15cm_Avg": x})


def test_flags_and_unit_fix():
    d, rep = clean_tdr_swc(_df(), {0.15: "SWC_15cm_Avg"})
    v = d["SWC_15cm_Avg"]
    assert v.iloc[[50, 80, 120]].isna().all()
    assert v.iloc[150:160].notna().all() and np.allclose(v.iloc[150:160], 0.25)
    assert int(rep.n_range.iloc[0]) == 2 and int(rep.n_hampel.iloc[0]) == 1
    assert v.notna().sum() == 197


def test_disabled():
    d, rep = clean_tdr_swc(_df(), {0.15: "SWC_15cm_Avg"}, vmin=None, vmax=None, window=None)
    assert rep.n_range.iloc[0] == 0 and rep.n_hampel.iloc[0] == 0
