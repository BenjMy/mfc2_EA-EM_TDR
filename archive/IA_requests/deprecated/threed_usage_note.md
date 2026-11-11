# Using `threed` for `--mode 2d` and real-data scenarios

## Decision

Use `threed=True` (emagpy's quasi-3D lateral-smoothing mode) for any scenario
run with `--mode 2d` and for real-data scenarios (`--data-source real`).
Do **not** use it for `--mode 1d` synthetic transect runs.

## Why

`emagpy`'s `Problem.invert()` builds the lateral-smoothing "previous profile"
(`pn`) differently depending on `threed`:

- `threed=False` (default): `pn` is simply row `j-1` of the dataframe — i.e.
  "the previous point in the CSV." This only makes physical sense when rows
  are ordered along a single transect, where row `j-1` really is the nearest
  spatial neighbor. That matches our `1d` synthetic scenarios.
- `threed=True`: `pn` is the mean of the 3 spatially nearest *already
  inverted* points (real XY distance via `xy`), which is the correct notion
  of "neighbor" once points are laid out on a 2D grid (`--mode 2d`) or come
  from a real field survey (`--data-source real`), where dataframe row order
  has no relationship to spatial adjacency.

So for our 2D and real-data scenarios, `threed=True` is the conceptually
correct setting if lateral smoothing (`beta > 0`) is being used at all.

## Prerequisite: the `beta` bug must be fixed first

`threed` only changes anything through the `pn` term, which enters the
objective function multiplied by `beta`:

```python
beta*np.sum((p - pn)**2)/len(p)
```

We found that `emagpy` 1.4.2 has a bug where `b` (the value actually passed
as `beta` into the per-row solve) is set to `0` at `j == 0` and never
reassigned to the real `beta` argument for `j > 0` — so `beta` is silently
zeroed out on every row today. Until that is patched, `threed` has **no
effect on results either**, regardless of what value we pass.

Fix (in `emagpy/Problem.py`, inside the per-row loop):

```diff
                     else: # previous survey
                         pn = np.r_[depth[j-1,:][vd], model[j-1,:][vc]]
+                    b = beta  # FIX: b was never re-assigned from beta for j > 0,
+                              # so lateral smoothing was silently a no-op (always 0).
                     
                 # define profile from previous survey for time-lapse constrain
```

Apply this (via a vendored/forked `emagpy` install, not by hand-editing
site-packages) before relying on `threed`.

## Caution: validate before trusting `threed`

Beyond the `beta` bug, tracing the `threed` branch surfaced an indexing
pattern that looks suspicious and has **not** been empirically confirmed
correct or broken (static trace only, not run):

- Model results are written to `depth[j,:]` / `model[j,:]` — dataframe row
  order (`j`) — on every iteration, regardless of `threed`.
- `iinverted[ipt] = True` marks completion at the spatially-sorted index
  `ipt = iorder[j]`, not at `j`.
- `icloseInverted = np.argsort(dist[iinverted])[:3]` returns positions
  *within the filtered/masked subset* (`0..sum(iinverted)-1`), which are
  then used directly as row indices into the full `depth`/`model` arrays —
  without translating back to original indices.

Unless `j == ipt` by coincidence, these may not line up, which could mean
`pn` averages the wrong points or references rows not yet filled.

**Before adopting `threed` for a real run:** apply the `beta` fix, then run
a small sanity check (e.g. `--mode 2d --nx 3 --ny 3` with an obviously large
`--beta`, like `5.0`) and manually inspect whether the resulting models look
spatially smoothed in a sensible way. If they don't, this indexing issue
needs a second patch before `threed` can be trusted for production runs.

## Action items

- [ ] Apply the `beta` fix to our `emagpy` fork/vendored copy.
- [ ] Wire `threed` into `em_methodology.py`'s `invert_kwargs` and expose it
      as a scenario-config field / CLI flag (`--threed`), following the same
      pattern as `--beta`/`--gamma`. Not currently exposed anywhere in the
      pipeline — `emagpy` always runs with its default `threed=False` today.
- [ ] Run the 3×3 grid sanity check above and confirm `pn` neighbor selection
      is correct before enabling `threed=True` by default for `2d` / real
      scenarios.
