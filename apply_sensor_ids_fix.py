#!/usr/bin/env python3
"""Run from the pipeline directory: python3 apply_sensor_ids_fix.py
Patches 2_run_petro.py and plot_water_content.py so run_pipeline.py's
sensor_ids / probe_ids keyword arguments are accepted. Writes .bak backups."""
import shutil

def patch(path, edits):
    src = open(path, encoding="utf-8").read()
    shutil.copy(path, path + ".bak")
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"[{path}] expected 1 match, found {n} for:\n{old}")
        src = src.replace(old, new)
    open(path, "w", encoding="utf-8").write(src)
    print(f"patched {path} ({len(edits)} edits, backup: {path}.bak)")

patch("2_run_petro.py", [
    # run_petro signature
    ("    lab_p0: np.ndarray | list | None = None,\n"
     "    theta_obs_std: float | Sequence[float] | None = None,\n"
     ") -> list[EMmethod.PetroFit]:",
     "    lab_p0: np.ndarray | list | None = None,\n"
     "    theta_obs_std: float | Sequence[float] | None = None,\n"
     "    sensor_ids: Sequence[str] | None = None,\n"
     ") -> list[EMmethod.PetroFit]:"),
    # route sensor ids with the same kept_indices
    ("        routed_sample_positions = np.asarray(sample_positions)[kept_indices]\n",
     "        routed_sample_positions = np.asarray(sample_positions)[kept_indices]\n"
     "        routed_sensor_ids = (\n"
     "            np.asarray(sensor_ids, dtype=object)[kept_indices]\n"
     "            if sensor_ids is not None else None\n"
     "        )\n"),
    ("            sample_positions=routed_sample_positions,\n"
     "            output_path=os.path.join(output_dir, \"parity_validation_by_sensor.png\"),",
     "            sample_positions=routed_sample_positions,\n"
     "            sensor_ids=routed_sensor_ids,\n"
     "            output_path=os.path.join(output_dir, \"parity_validation_by_sensor.png\"),"),
    # run_petro_localized signature
    ("    theta_obs_std: float | Sequence[float] | None = None,\n"
     "    global_fit: \"EMmethod.PetroFit | None\" = None,\n"
     ") -> list[\"EMmethod.LocalizedPetroFit\"]:",
     "    theta_obs_std: float | Sequence[float] | None = None,\n"
     "    global_fit: \"EMmethod.PetroFit | None\" = None,\n"
     "    sensor_ids: Sequence[str] | None = None,\n"
     ") -> list[\"EMmethod.LocalizedPetroFit\"]:"),
])

patch("plot_water_content.py", [
    ("    xy_imshow_extra: bool = True,\n) -> None:\n    \"\"\"\n    Render all water-content figures",
     "    xy_imshow_extra: bool = True,\n    probe_ids: Sequence[str] | None = None,\n) -> None:\n    \"\"\"\n    Render all water-content figures"),
    ("    xy_imshow_extra: bool = True,\n) -> None:\n    \"\"\"\n    Render the ground-truth counterpart",
     "    xy_imshow_extra: bool = True,\n    probe_ids: Sequence[str] | None = None,\n) -> None:\n    \"\"\"\n    Render the ground-truth counterpart"),
])
