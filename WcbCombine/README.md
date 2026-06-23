# WcbAnlysisCombine

## Initial Setup

At first, you need to set up your CMSSW environment like following:

```bash
cmsrel CMSSW_14_1_0_pre4
cd CMSSW_14_1_0_pre4/src
cmsenv
git -c advice.detachedHead=false clone --depth 1 --branch v10.5.0 https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit
scramv1 b clean; scramv1 b -j$(nproc --ignore=2) # always make a clean build, with n - 2 cores on the system
cd ../../
git clone https://github.com/WilliamMarci/WcbCombine.git WcbAnlysisCombine
cd WcbAnlysisCombine
```

each time you want to use the combine framework, you need to set up the environment by:

```bash
cd CMSSW_14_1_0_pre4/src
cmsenv
```

## Blind with Calibration

In fact, you can directly run the combine use

```shell
auto_run_combine.sh 0.985 0.7
```
first argument is the dbc cut, and the second one is the event classifier cut. You can also run the combine step by step, and the steps are listed as following:


```shell
./make_shape_asimov.py   --dbc 0.85   --cv 0.7   --lumi 1   --sr-asimov sb   --sr-asimov-mu 1.0

./write_shape_card.py shapes_asimov_vcb_dbc0.85_cv0.7_2.root \
  -r calib sr \
  --include-syst 'FT_Stat_flav*' PU ElEff MuEff L1PreFiring TrigEff Theory_muR Theory_muF Theory_isr Theory_fsr Theory_topPt Theory_hdamp Theory_pdfSum \
  --auto-groups \
  -o card_calib_sr_hybrid.txt

python3 dump_shapes.py -i shapes_asimov_vcb_dbc0.85_cv0.7_2.root

export PYTHONPATH=$PWD:$PYTHONPATH

text2workspace.py card_calib_sr_hybrid.txt   -o ws_mu_lambda_hybrid.root   -P VcbModel:vcbModel   --PO poi=mu,lambda_cal
text2workspace.py card_calib_sr_hybrid.txt   -o ws_lambda_cal_hybrid.root   -P VcbModel:vcbModel   --PO poi=lambda_cal
text2workspace.py card_calib_sr_hybrid.txt   -o ws_mu_hybrid.root   -P VcbModel:vcbModel   --PO poi=mu

combine -M MultiDimFit ws_mu_lambda_hybrid.root   -t -1   --setParameters mu=1,lambda_cal=1   --redefineSignalPOIs mu,lambda_cal   --algo singles   --robustFit 1   --cminDefaultMinimizerStrategy 0   -n _closure_2d

python3 significance_from_scan.py higgsCombine_mu_scan.MultiDimFit.mH120.root --poi mu --plot
# auto analysis

./make_shape_asimov.py   --dbc 0.85   --cv 0.7   --lumi 1   --sr-asimov sb   --sr-asimov-mu 1.0

python3 rebin_shape_asimov.py   -i shapes_asimov_vcb_dbc0.85_cv0.7_2.root -o rebin_scan_outputs_dbc0.85_cv0.7 --cv-cut 0.7 --schemes equal_6 equal_8 tail_7a signal_eq_7 origin

python3 run_rebin_benchmark.py --scan-dir rebin_scan_outputs_dbc0.85_cv0.7

```

an auto shell to run it by dbc and cv:
```shell
DBCCUT=$1
CVCUT=$2

#if not exist, make the shape
if [ ! -f "shapes_asimov_vcb_dbc${DBCCUT}_cv${CVCUT}_2.root" ]; then
  echo ./make_shape_asimov.py   --dbc ${DBCCUT}   --cv ${CVCUT}   --lumi 1   --sr-asimov sb   --sr-asimov-mu 1.0
fi
python3 rebin_shape_asimov.py   -i shapes_asimov_vcb_dbc${DBCCUT}_cv${CVCUT}_2.root -o rebin_scan_outputs_dbc${DBCCUT}_cv${CVCUT} --cv-cut ${CVCUT} --schemes equal_6 equal_8 tail_7a signal_eq_7 origin
python3 run_rebin_benchmark.py --scan-dir rebin_scan_outputs_dbc${DBCCUT}_cv${CVCUT}

```



python3 dump_shapes.py -i shapes_asimov_vcb_dbc0.85_cv0.5_2.root
./write_shape_card.py shapes_asimov_vcb_dbc0.85_cv0.5_2.root \
  -r calib sr \
  --include-syst 'FT_*' PU ElEff MuEff L1PreFiring TrigEff Theory_muR Theory_muF Theory_isr Theory_fsr Theory_topPt Theory_hdamp Theory_pdfSum \
  --auto-groups \
  -o card_calib_sr_hybrid_dbc0.85_cv0.5.txt