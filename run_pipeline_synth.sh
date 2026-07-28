python3 generate_synthetic_real_mirror.py --coord-csv ./data/SOILNET_MFC2_coordinates_sensors.csv --outdir ./data/synthetic_real_mirror/ --scale 0.1
python3 run_and_score.py --project-dir /path/to/pipeline --em-dir ./data/synthetic_real_mirror/EMI_MFC2_synthetic \
  --tdr-dir ./data/synthetic_real_mirror/daily_MFC_synthetic --coord-file ./data/SOILNET_MFC2_coordinates_sensors.csv \
  --scenario mfc2_real_synthetic_mirror --config mfc2_real_synthetic_mirror.json \
  --out ./capture.npz
python3 score_against_truth.py --capture ./capture.npz --generator-dir . --out ./scoring.csv
