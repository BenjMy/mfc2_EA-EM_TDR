## `beta` (lateral smoothing) is silently ignored in `Problem.invert()`

### Summary

Passing any non-zero `beta` to `Problem.invert()` has no effect on the
inversion results. The value passed in is never actually used by the
solver for any row after the first — every row is inverted with the
lateral-smoothing weight hard-coded to `0`, regardless of the `beta`
argument.

### Where

`emagpy/Problem.py`, inside `invert()`'s per-row inversion loop:

```python
for j in range(nrows):
    ...
    # define previous profile in case we want to lateral constrain
    if j == 0:
        pn = np.zeros(np.sum(np.r_[vd, vc]))
        b = 0
        iinverted[j] = True
    else:
        if threed: # mean of 3 closest inverted survey
            ipt = iorder[j]
            dist = np.sqrt(np.sum((xy-xy[ipt,:])**2, axis=1))
            icloseInverted = np.argsort(dist[iinverted])[:3]
            val = np.c_[depth[icloseInverted,:][:,vd],
                        model[icloseInverted,:][:,vc]]
            pn = np.mean(val, axis=0)
            iinverted[ipt] = True
        else: # previous survey
            pn = np.r_[depth[j-1,:][vd], model[j-1,:][vc]]

    ...
    params.append((obs, pn, spn, alpha, b, g, ini0))
```

`b` is set to `0` once, in the `j == 0` branch, and is **never reassigned**
in the `else` branch (`j > 0`). Since Python variables persist across loop
iterations, `b` simply stays `0` for the rest of the profile. The
`params.append(...)` call then passes this stale `b` — not the `beta`
argument the user supplied — into `solve(...)`, which forwards it straight
into `objfunc`'s regularization term:

```python
beta*np.sum((p - pn)**2)/len(p)   # l2
beta*np.sum(np.abs(p - pn))/len(p)  # l1
```

So the lateral-smoothing penalty is always `0 * (...)`, no matter what
`beta` is set to.

### Reproduction

Run `invert()` twice on the same survey/config, once with `beta=0.0` and
once with a large value, e.g. `beta=10.0`. The resulting `k.models` /
`k.depths` are effectively identical (differences are only from stochastic
solvers like `ROPE`, not from `beta`), even though a `beta` this large
should visibly pull neighboring profiles together.

### Expected behavior

`b` should be reassigned to the actual `beta` argument for every row where
a real neighboring profile (`pn`) is available (i.e. `j > 0`), so the
lateral-smoothing term in `objfunc` actually reflects the requested `beta`.

### Suggested fix

```diff
                     else: # previous survey
                         pn = np.r_[depth[j-1,:][vd], model[j-1,:][vc]]
+                    b = beta  # b was never re-assigned from beta for j > 0,
+                              # so lateral smoothing was silently a no-op.
                     
                 # define profile from previous survey for time-lapse constrain
```

Note `b = 0` at `j == 0` is correct and should stay as-is: there is no real
neighboring profile at the first row (`pn` is just a zero-vector
placeholder there), so smoothing toward it would be wrong. The bug is
specifically that `b` never gets updated once a real `pn` exists.

### Environment

- `emagpy` version: 1.4.2 (latest on PyPI at time of writing)
- Confirmed via static trace of `emagpy/Problem.py`'s `invert()`; not yet
  confirmed against a live before/after numerical comparison with the patch
  applied.

Happy to open a PR with this change if useful.
