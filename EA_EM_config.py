"""EA_EM_config.py -- single place where the INPUT data paths of the EA-EM pipeline are defined.

Nothing else in the repository should hard-code ``./data/...``: the pipeline
(``run_pipeline.py``), the launchers (``run_one_scenario.sh``) and the scenario
loader ask this module.  To run the pipeline on a new site / survey that is not
the MFC2 catchment, you only touch THIS file (or a JSON next to it, see below).

A *dataset* bundles everything the real-data branch reads:

    em_dir        directory of EM survey CSV files (one file per survey date)
    tdr_dir       directory of TDR / soil-moisture spreadsheets (.xlsx)
    coord_file    sensor (probe) coordinates CSV
    probe_xyz     optional explicit probe x/y/z CSV (the ``--probe-xyz`` option)

Selecting a dataset (first match wins)
--------------------------------------
    1. ``--em-dir / --tdr-dir / --coord-file``   explicit path on the command line
    2. ``--dataset NAME``                         command line
    3. ``em_data_dir / tdr_data_dir / coord_file`` written in the scenario JSON
    4. ``EA_EM_DATASET=NAME``                     environment variable
    5. ``DEFAULT_DATASET``                        below ("mfc2")

A scenario JSON may therefore simply OMIT the three path keys; they then come
from the selected dataset.  The existing MFC2 scenarios keep their explicit
paths and behave exactly as before.

Adding a new dataset
--------------------
Either edit ``DATASETS`` below, or (no code change) create
``EA_EM_datasets.json`` next to this file (or point ``EA_EM_DATASETS_FILE`` to
one):

    {
      "my_site": {
        "em_dir":     "./data/my_site/EM/",
        "tdr_dir":    "./data/my_site/TDR/",
        "coord_file": "./data/my_site/probe_coordinates.csv",
        "description": "Site X, 2025 campaign"
      }
    }

Relative paths are relative to the working directory (run from the repository
root, as for the rest of the pipeline).  ``EA_EM_DATA_ROOT`` moves the whole
default ``./data`` tree (e.g. onto a shared drive) without editing anything.

Command line
------------
    python EA_EM_config.py --list                      # known datasets
    python EA_EM_config.py --check [--dataset NAME]    # do the paths exist?
    python EA_EM_config.py --dataset NAME --field em_dir   # one value (used by the .sh launchers)
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

#: Root of the input data tree. Override with the EA_EM_DATA_ROOT env variable.
DATA_ROOT: str = os.environ.get("EA_EM_DATA_ROOT", "./data").rstrip("/") or "."

#: Root folder for everything the pipeline writes (run folders, run_log.csv, ...).
#: Override with EA_EM_OUTPUT_ROOT; the default keeps the historical name.
OUTPUT_ROOT: str = os.environ.get("EA_EM_OUTPUT_ROOT", "mfc2_outputs").rstrip("/") or "."

#: Central run log shared by run_pipeline.py and the comparison / study scripts.
RUN_LOG: str = OUTPUT_ROOT + "/run_log.csv"

#: Dataset used when nothing else selects one (kept = MFC2 for backward compatibility).
DEFAULT_DATASET: str = "mfc2"

#: Optional JSON with extra datasets (see module docstring).
USER_DATASETS_FILE: str = os.environ.get(
    "EA_EM_DATASETS_FILE", os.path.join(_HERE, "EA_EM_datasets.json")
)


def _p(*parts: str) -> str:
    """Join under DATA_ROOT, always with forward slashes (the launchers are bash)."""
    return "/".join([DATA_ROOT, *parts])


# ---------------------------------------------------------------------------
# Dataset definition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetPaths:
    """Input files/directories of one field dataset."""

    name: str
    em_dir: str
    tdr_dir: str
    coord_file: str
    probe_xyz: str | None = None
    description: str = ""

    def problems(self) -> list[str]:
        """Human-readable list of missing paths (empty list = all good)."""
        out = []
        if not os.path.isdir(self.em_dir):
            out.append(f"EM directory not found: {self.em_dir}")
        if not os.path.isdir(self.tdr_dir):
            out.append(f"TDR directory not found: {self.tdr_dir}")
        if not os.path.isfile(self.coord_file):
            out.append(f"Coordinates file not found: {self.coord_file}")
        if self.probe_xyz and not os.path.isfile(self.probe_xyz):
            out.append(f"probe_xyz file not found: {self.probe_xyz}")
        return out

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Registered datasets  <-- ADD NEW SITES HERE (or in EA_EM_datasets.json)
# ---------------------------------------------------------------------------
DATASETS: dict[str, DatasetPaths] = {
    "mfc2": DatasetPaths(
        name="mfc2",
        em_dir=_p("EMI_MFC2") + "/",
        tdr_dir=_p("daily_MFC") + "/",
        coord_file=_p("SOILNET_MFC2_coordinates_sensors.csv"),
        description="MFC2 catchment (Alento): GEM-2 EM surveys + SoilNet TDR",
    ),
}


def register_dataset(name: str, em_dir: str, tdr_dir: str, coord_file: str,
                     probe_xyz: str | None = None, description: str = "") -> DatasetPaths:
    """Add (or replace) a dataset at runtime, e.g. from a notebook or a test."""
    ds = DatasetPaths(name, em_dir, tdr_dir, coord_file, probe_xyz, description)
    DATASETS[name] = ds
    return ds


def _load_user_datasets(path: str = USER_DATASETS_FILE) -> None:
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        for name, d in raw.items():
            if name.startswith("_"):          # allow "_comment" keys
                continue
            register_dataset(
                name, d["em_dir"], d["tdr_dir"], d["coord_file"],
                d.get("probe_xyz"), d.get("description", ""),
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"EA_EM_config: cannot read datasets file '{path}': {exc}") from exc


_load_user_datasets()


# ---------------------------------------------------------------------------
# Lookup / resolution
# ---------------------------------------------------------------------------
def selected_dataset_name(name: str | None = None) -> str:
    """Explicit name > EA_EM_DATASET env variable > DEFAULT_DATASET."""
    return name or os.environ.get("EA_EM_DATASET") or DEFAULT_DATASET


def get_dataset(name: str | None = None) -> DatasetPaths:
    """Return the DatasetPaths for ``name`` (see ``selected_dataset_name``)."""
    key = selected_dataset_name(name)
    try:
        return DATASETS[key]
    except KeyError:
        raise KeyError(
            f"Unknown dataset '{key}'. Known datasets: {', '.join(sorted(DATASETS))}. "
            f"Add it to DATASETS in EA_EM_config.py or to {os.path.basename(USER_DATASETS_FILE)}."
        ) from None


@dataclass
class ResolvedPaths:
    em_dir: str | None
    tdr_dir: str | None
    coord_file: str | None
    dataset: str


def resolve_real_paths(
    cli_em_dir: str | None = None,
    cli_tdr_dir: str | None = None,
    cli_coord_file: str | None = None,
    cli_dataset: str | None = None,
    scenario_em_dir: str | None = None,
    scenario_tdr_dir: str | None = None,
    scenario_coord_file: str | None = None,
) -> ResolvedPaths:
    """Apply the precedence documented in the module docstring, field by field.

    explicit CLI path > --dataset > scenario JSON path > EA_EM_DATASET / DEFAULT_DATASET
    """
    if cli_dataset:                       # --dataset beats what the scenario JSON hard-codes
        ds = get_dataset(cli_dataset)
        em = cli_em_dir or ds.em_dir
        tdr = cli_tdr_dir or ds.tdr_dir
        coord = cli_coord_file or ds.coord_file
    else:
        ds = get_dataset(None)
        em = cli_em_dir or scenario_em_dir or ds.em_dir
        tdr = cli_tdr_dir or scenario_tdr_dir or ds.tdr_dir
        coord = cli_coord_file or scenario_coord_file or ds.coord_file
    return ResolvedPaths(em, tdr, coord, ds.name)


def add_path_args(parser) -> None:
    """Add --dataset/--em-dir/--tdr-dir/--coord-file (all optional) to an argparse parser."""
    parser.add_argument("--dataset", default=None, metavar="NAME",
                        help="named input dataset from EA_EM_config.py "
                             f"(default: $EA_EM_DATASET or '{DEFAULT_DATASET}')")
    parser.add_argument("--em-dir", default=None, help="EM survey directory (overrides the dataset)")
    parser.add_argument("--tdr-dir", default=None, help="TDR directory (overrides the dataset)")
    parser.add_argument("--coord-file", default=None, help="sensor coordinates CSV (overrides the dataset)")


def paths_from_args(args, cfg=None) -> ResolvedPaths:
    """ResolvedPaths from parsed ``add_path_args`` options (+ an optional loaded scenario)."""
    return resolve_real_paths(
        cli_em_dir=args.em_dir, cli_tdr_dir=args.tdr_dir,
        cli_coord_file=args.coord_file, cli_dataset=args.dataset,
        scenario_em_dir=getattr(cfg, "em_data_dir", None),
        scenario_tdr_dir=getattr(cfg, "tdr_data_dir", None),
        scenario_coord_file=getattr(cfg, "coord_file", None),
    )


# ---------------------------------------------------------------------------
# Other input files that used to be hard-coded
# ---------------------------------------------------------------------------
#: Reference EM file used by the stand-alone ``step1_inversion.py`` demo.
SYNTHETIC_EM_FILE: str = _p("synthetic", "data_Model 1.csv")

#: Explicit probe layouts for the synthetic 2-D runs (``--probe-xyz``).
PROBE_XYZ_FILES: dict[str, str] = {
    "2d_2sensors": _p("coord_TDR_mode2d_2sensors.csv"),
    "2d_8sensors": _p("coord_TDR_mode2d_8sensors.csv"),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Show / check the EA-EM input-data configuration.")
    ap.add_argument("--list", action="store_true", help="list known datasets")
    ap.add_argument("--dataset", default=None, help="dataset name (default: env / DEFAULT_DATASET)")
    ap.add_argument("--check", action="store_true", help="verify that the dataset's paths exist")
    ap.add_argument("--probe-xyz", default=None, metavar="KEY", choices=sorted(PROBE_XYZ_FILES),
                    help="print the path of a registered synthetic probe layout (PROBE_XYZ_FILES)")
    ap.add_argument("--output-root", action="store_true", help="print OUTPUT_ROOT")
    ap.add_argument("--field", default=None,
                    choices=["em_dir", "tdr_dir", "coord_file", "probe_xyz", "name"],
                    help="print one value (for shell scripts)")
    args = ap.parse_args(argv)

    if args.probe_xyz:
        print(PROBE_XYZ_FILES[args.probe_xyz])
        return 0
    if args.output_root:
        print(OUTPUT_ROOT)
        return 0

    if args.list:
        cur = selected_dataset_name(args.dataset)
        for k, ds in sorted(DATASETS.items()):
            print(f"{'*' if k == cur else ' '} {k:<16} {ds.description}")
            print(f"    em_dir     {ds.em_dir}\n    tdr_dir    {ds.tdr_dir}\n    coord_file {ds.coord_file}")
        return 0

    try:
        ds = get_dataset(args.dataset)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        return 3

    if args.field:
        print(getattr(ds, args.field) or "")
        return 0
    if args.check:
        probs = ds.problems()
        print(f"dataset '{ds.name}':", "OK" if not probs else "")
        for p in probs:
            print("  -", p)
        return 1 if probs else 0

    print(json.dumps(ds.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
