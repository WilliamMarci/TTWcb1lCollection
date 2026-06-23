DBCCUT=$1
CVCUT=$2

# conda activate mlenv
# python build_bdt_cache.py \
# --mc_path "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/MC/scored_samples_1merged_/*.root" \
# --data_path "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/Data/scored_samples_1merged_/*.root" \
# --cache_dir ./bdt_cache \
# --model_path ./bdt_dbc_model.pkl
# conda deactivate

#if not exist, make the shape
if [ ! -f "shapes_asimov_vcb_dbc${DBCCUT}_cv${CVCUT}_bdt_sc.root" ]; then
  ./make_shape_asimov.py   --dbc ${DBCCUT}   --cv ${CVCUT}   --lumi 1   --sr-asimov sb   --sr-asimov-mu 1.0 --bdt-dbc --cache_dir ./bdt_cache
fi
# python3 rebin_shape_asimov.py   -i shapes_asimov_vcb_dbc${DBCCUT}_cv${CVCUT}_bdt.root -o rebin_scan_outputs_dbc${DBCCUT}_cv${CVCUT}_bdt --cv-cut ${CVCUT} --schemes equal_6 equal_8 tail_7a signal_eq_7 origin
python3 rebin_shape_asimov.py   -i shapes_asimov_vcb_dbc${DBCCUT}_cv${CVCUT}_bdt_sc.root -o rebin_scan_outputs_dbc${DBCCUT}_cv${CVCUT}_bdt_sc --cv-cut ${CVCUT} --schemes origin
python3 run_rebin_benchmark.py --scan-dir rebin_scan_outputs_dbc${DBCCUT}_cv${CVCUT}_bdt_sc --run-impacts

