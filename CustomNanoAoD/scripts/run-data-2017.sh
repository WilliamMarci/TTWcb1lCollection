#!/bin/bash

# Avoid "input: $HOME/.root.mimes, output: $HOME/.root.mimes" error.
# REF: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideCrabFaq
if [ -z "${HOME}" ]; then
    export HOME="$(pwd)"
fi

if [ $# -lt 3 ]; then
    >&2 echo "usage: $(basename "$0") <nevent> <nthread> <file-in> <file-out>"
    exit 1
fi
NEVENT="$1"
NTHREAD="$2"
FILEIN="$3"
FILEOUT="$4"
if [ -z "${FILEOUT}" ]; then
    FILEOUT="${FILEIN/MiniAODv2/CustomizedNanoAODv9}"
fi
if [ "${FILEIN:0:7}" != "root://" ]; then FILEIN="file:${FILEIN}"; fi
if [ "${FILEOUT:0:7}" != "root://" ]; then FILEOUT="file:${FILEOUT}"; fi

set -ev
voms-proxy-info  # early stop on proxy error
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc700
[ -r CMSSW_10_6_31 ] || cmsrel CMSSW_10_6_31
cd CMSSW_10_6_31/src
cmsenv

rm -rf PhysicsTools/NanoTuples
git clone https://github.com/lyazj/hss-nano PhysicsTools/NanoTuples -b dev-ak15tagger-UL
PhysicsTools/NanoTuples/scripts/install_onnxruntime.sh
wget https://coli.web.cern.ch/coli/tmp/.231117-195737_ak15_stage2/model.onnx -O $CMSSW_BASE/src/PhysicsTools/NanoTuples/data/InclParticleTransformer-MD/ak15/V02/model.onnx
scram b -j$(cat /proc/cpuinfo | grep MHz | wc -l)

cmsDriver.py \
    --data \
    -n "${NEVENT}" \
    --nThreads "${NTHREAD}" \
    --python_filename run-data-2017.py \
    --eventcontent NANOAOD \
    --datatier NANOAOD \
    --conditions 106X_dataRun2_v35 \
    --step NANO \
    --era Run2_2017,run2_nanoAOD_106Xv2 \
    --customise PhysicsTools/NanoTuples/nanoTuples_cff.nanoTuples_customizeData \
    --filein "${FILEIN}" \
    --fileout "${FILEOUT}" \
