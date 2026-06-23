# NanoTuples

Custom NanoAOD ntuple producers with additional boosted jet taggers and their PF candidates.

Original repo in https://github.com/hqucms/NanoTuples/tree/dev/NanoAODv8.

------

## Version

The current version is based on [NanoAODv9](https://gitlab.cern.ch/cms-nanoAOD/nanoaod-doc/-/wikis/Releases/NanoAODv9).

Customizations:

- AK15 jets w/ ParticleNet-MD (V02d, EOY training)
- [*not enabled by default*] PFCands of AK15 jets

------

## Setup

### Set up CMSSW

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc700
[ -r CMSSW_10_6_31 ] || cmsrel CMSSW_10_6_31
cd CMSSW_10_6_31/src
cmsenv
```

### Get customized NanoAOD producers

```bash
git clone https://github.com/lyazj/hss-nano PhysicsTools/NanoTuples -b dev-ak15tagger-UL
```

### Use an updated onnxruntime package (before compiling the code)

```bash
PhysicsTools/NanoTuples/scripts/install_onnxruntime.sh
```

### Get the ParT model

```bash
wget https://coli.web.cern.ch/coli/tmp/.231117-195737_ak15_stage2/model.onnx -O $CMSSW_BASE/src/PhysicsTools/NanoTuples/data/InclParticleTransformer-MD/ak15/V02/model.onnx
```

### Compile

```bash
scram b -j16
```

### Test

MC (UL18, MiniAODv2):

```bash
cmsDriver.py --python_filename test_nanoTuples_mc2018.py --eventcontent NANOAODSIM --customise PhysicsTools/NanoTuples/nanoTuples_cff.nanoTuples_customizeMC --datatier NANOAODSIM --fileout file:nano_mc2018.root --conditions 106X_upgrade2018_realistic_v16_L1v1 --step NANO --filein /store/mc/RunIISummer20UL18MiniAODv2/WplusH_HToCC_WToLNu_M-125_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/2520000/A4FFC5A7-79DF-FE4B-A515-1F3EA0513509.root --era Run2_2018,run2_nanoAOD_106Xv2 --mc -n 50
```

Data (UL18, MiniAODv2):

```bash
cmsDriver.py --python_filename test_nanoTuples_data2018.py --eventcontent NANOAOD --customise PhysicsTools/NanoTuples/nanoTuples_cff.nanoTuples_customizeData --datatier NANOAOD --fileout file:nano_data2018.root --conditions 106X_dataRun2_v37 --step NANO --filein /store/data/Run2018A/SingleMuon/MINIAOD/UL2018_MiniAODv2_GT36-v1/2820000/000EE25A-A8E8-1444-8A0B-0DBEBE5634FB.root --era Run2_2018,run2_nanoAOD_106Xv2 --data -n 50
```
