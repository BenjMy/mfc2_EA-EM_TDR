#!/usr/bin/env bash
# Runs the test plan (TEST_PLAN.md) for branch todo/train-val-synthetic.
#
# Usage (from anywhere; it cd's to the repo root = folder of this script):
#   ./run_test_plan.sh                 # all steps 0-6 (about 55 min without WITH_FULL)
#   ./run_test_plan.sh 1 2 3           # only some steps
#   ./run_test_plan.sh 6               # only compare + bundle (runs must already exist)
#   WITH_FULL=1 ./run_test_plan.sh 5 6 # also the full-depth reference runs (+ ~25 min)
#   CHECK=/some/dir ./run_test_plan.sh # where logs/results go (default ~/mfc2_check, NOT /tmp)
#
# Steps: 0 git info | 1 pytest | 2 scenarios load | 3 regression baseline
#        4 train/val | 5 pinned experiment (uniform + hetero) | 6 compare tables/plots + tgz
#
# Needs the conda env with emagpy active. A failing step does not stop the rest;
# exit codes are written to $CHECK/exit_codes.txt.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

CHECK="${CHECK:-$HOME/mfc2_check}"
mkdir -p "$CHECK"
OUT="$(python EA_EM_config.py --output-root)"
[ -n "$OUT" ] || { echo "could not get the output root (is the conda env active?)"; exit 1; }
STEPS=("$@"); [ ${#STEPS[@]} -gt 0 ] || STEPS=(0 1 2 3 4 5 6)
WITH_FULL="${WITH_FULL:-0}"
NPOS=5

has()    { local s; for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }
say()    { printf '\n=== %s (%s)\n' "$1" "$(date +%H:%M:%S)"; }
rc_log() { echo "$1: exit $2" | tee -a "$CHECK/exit_codes.txt"; }
latest() { ls -d "$OUT"/field_survey_"$1"/[0-9]*_* 2>/dev/null | tail -1; }
runp()   { # runp <name> <logfile> <cmd...>: run, log to file, record exit code
  local name="$1" log="$2"; shift 2
  say "$name"; "$@" > "$log" 2>&1; local rc=$?
  rc_log "$name" "$rc"
  echo "traceback count in $(basename "$log"): $(grep -a -c -i traceback "$log")" | tee -a "$CHECK/exit_codes.txt"
  return 0
}

echo "repo: $(pwd)"; echo "OUT=$OUT"; echo "CHECK=$CHECK"; echo "steps: ${STEPS[*]}"
: > "$CHECK/exit_codes.txt"

# ---------------------------------------------------------------- 0
if has 0; then
  say "0 git"
  { git log --oneline -8; git status -sb; git log -1 --format='%an <%ae>'; } | tee "$CHECK/00_git.txt"
fi

# ---------------------------------------------------------------- 1
if has 1; then
  say "1 pytest"
  if [ -f tests/data/scenarios_golden.json ]; then
    python -m pytest tests -q 2>&1 | tee "$CHECK/01_pytest.txt" | tail -5
  else
    echo "tests/data/scenarios_golden.json missing: golden and sweeps tests excluded"
    python -m pytest tests -q --ignore=tests/test_scenarios_golden.py --ignore=tests/test_sweeps.py \
      2>&1 | tee "$CHECK/01_pytest.txt" | tail -5
  fi
fi

# ---------------------------------------------------------------- 2
if has 2; then
  say "2 scenarios load"
  python - <<'PY' 2>&1 | tee "$CHECK/02_scenarios.txt"
import scenario_loader as sl
for n in ("1d_surface_free2_tv_probe", "1d_full_free2_tv_probe",
          "1d_surface_free2_pin0", "1d_surface_hetero_free2_pin0"):
    c = sl.load_scenario(n)
    print(n, "| split:", c.train_val_split, "| pos:", c.num_swc_positions,
          "| pin:", c.pin_layers, c.pin_source)
PY
fi

# ---------------------------------------------------------------- 3
if has 3; then
  runp "3 baseline 1d_surface_tdr" "$CHECK/03_baseline.log" \
       python run_pipeline.py --scenario 1d_surface_tdr --mode 1d --num-swc-positions $NPOS
  RUN="$(latest 1d_surface_tdr)"; echo "run: $RUN"
  grep -a "Final RMSE (Step 5)" "$RUN"/pipeline_report_*.md 2>/dev/null | tee "$CHECK/03_rmse.txt"
  echo "expected final RMSE: 0.031212 (Ben's machine)"
  ls "$RUN/output_model" 2>/dev/null | tee "$CHECK/03_files.txt" | grep -E "ws_params|tdr_train" || echo "WARNING: new files missing"
fi

# ---------------------------------------------------------------- 4
if has 4; then
  runp "4 train/val tv_probe" "$CHECK/04_tv.log" \
       python run_pipeline.py --scenario 1d_surface_free2_tv_probe --mode 1d
  tr '\r' '\n' < "$CHECK/04_tv.log" | grep -a "\[SPLIT\]\|\[STEP 5b\]" | cut -c1-200 | tee "$CHECK/04_split.txt"
fi

# ---------------------------------------------------------------- 5
if has 5; then
  runp "5a uniform pass1+pass2 (run_pinned_tdr)" "$CHECK/05_pinned.log" \
       python run_pinned_tdr.py --scenario 1d_surface_free2 --mode 1d --num-swc-positions $NPOS
  tr '\r' '\n' < "$CHECK/05_pinned.log" | grep -a "\[pinned-tdr\]\|\[PIN\]\|Layer pinning" | cut -c1-220 | tee "$CHECK/05_pin.txt"
  runp "5b uniform oracle (pin0)" "$CHECK/05_oracle.log" \
       python run_pipeline.py --scenario 1d_surface_free2_pin0 --mode 1d --num-swc-positions $NPOS

  runp "5c hetero pass1+pass2 (run_pinned_tdr)" "$CHECK/05h_pinned.log" \
       python run_pinned_tdr.py --scenario 1d_surface_hetero_free2 --mode 1d --num-swc-positions $NPOS
  runp "5d hetero oracle (pin0)" "$CHECK/05h_oracle.log" \
       python run_pipeline.py --scenario 1d_surface_hetero_free2_pin0 --mode 1d --num-swc-positions $NPOS

  if [ "$WITH_FULL" = 1 ]; then
    runp "5e uniform full-depth" "$CHECK/05_full.log" \
         python run_pipeline.py --scenario 1d_full_free2 --mode 1d --num-swc-positions $NPOS
    runp "5f hetero full-depth" "$CHECK/05h_full.log" \
         python run_pipeline.py --scenario 1d_full_hetero_free2 --mode 1d --num-swc-positions $NPOS
  fi
fi

# ---------------------------------------------------------------- 6
compare() { # compare <tag> <free scenario> <full scenario>
  local tag="$1" base="$2" full="$3"
  local free pin ora fullrun args
  free="$(latest "$base")"; pin="$(latest "${base}_pintdr")"; ora="$(latest "${base}_pin0")"
  if [ -z "$free" ] || [ -z "$pin" ] || [ -z "$ora" ]; then
    echo "SKIP $tag: missing run (free='$free' pinTDR='$pin' oracle='$ora'); run step 5 first"; return
  fi
  [ -f "$free/pass2_scenarios.json" ] || echo "NOTE $tag: latest free run has no pass2_scenarios.json; it may not be the pass 1 of the latest pinTDR run:
      $(ls -t "$OUT"/field_survey_"$base"/*/pass2_scenarios.json 2>/dev/null | head -1)"
  args=(free="$free" pinTDR="$pin" oracle="$ora")
  fullrun="$(latest "$full")"; [ -n "$fullrun" ] && args+=(full="$fullrun")
  echo "runs for $tag:"; printf '  %s\n' "${args[@]}"
  python compare_pinned.py "${args[@]}" --csv "$CHECK/05_$tag.csv" --plot "$CHECK/05_$tag.png" 2>&1 \
    | tee "$CHECK/05_$tag.md"
}

if has 6; then
  say "6 compare"
  compare uniform 1d_surface_free2        1d_full_free2
  compare hetero  1d_surface_hetero_free2 1d_full_hetero_free2

  say "bundle"
  ( cd "$CHECK" && tar czf mfc2_check.tgz *.txt *.md *.csv *.png 2>/dev/null; ls -la mfc2_check.tgz )
  echo; echo "Send back: $CHECK/mfc2_check.tgz (logs *.log only if something failed)"
fi

say "done"; cat "$CHECK/exit_codes.txt"
