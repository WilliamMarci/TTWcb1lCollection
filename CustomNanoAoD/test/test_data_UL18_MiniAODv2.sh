#!/bin/bash

stdbuf -oL cmsDriver.py \
    --data \
    -n 50 \
    --nThreads 1 \
    --python_filename UL2018_MiniAODv2_GT36-v1_2820000.py \
    --eventcontent NANOAOD \
    --datatier NANOAOD \
    --conditions 106X_dataRun2_v37 \
    --step NANO \
    --era Run2_2018,run2_nanoAOD_106Xv2 \
    --customise PhysicsTools/NanoTuples/nanoTuples_cff.nanoTuples_customizeData \
    --filein /store/data/Run2018A/SingleMuon/MINIAOD/UL2018_MiniAODv2_GT36-v1/2820000/000EE25A-A8E8-1444-8A0B-0DBEBE5634FB.root \
    --fileout file:UL2018_MiniAODv2_GT36-v1_2820000-000EE25A-A8E8-1444-8A0B-0DBEBE5634FB.root \
    2>&1 | tee test_data_UL18_MiniAODv2.log
