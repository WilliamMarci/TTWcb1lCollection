#!/bin/bash

# Avoid "input: $HOME/.root.mimes, output: $HOME/.root.mimes" error.
# REF: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideCrabFaq
if [ -z "${HOME}" ]; then
    export HOME="$(pwd)"
fi

if [ $# -lt 3 ]; then
    >&2 echo "usage: $(basename "$0") <nevent> <nthread> <file-in> <file-out> <x509up>"
    exit 1
fi
NEVENT="$1"
NTHREAD="$2"
FILEIN="$3"
FILEOUT="$4"
X509UP="$5"
if [ -z "${FILEOUT}" ]; then
    FILEOUT="${FILEIN/MiniAODv2/CustomizedNanoAODv9}"
fi
if [ ! -z "${X509UP}" ]; then
    export X509_USER_PROXY="${X509UP}"
    if ! voms-proxy-info -file /tmp/x509up_u${UID}; then
        cp "${X509_USER_PROXY}" /tmp/x509up_u${UID}
    fi  
fi
if [ "${FILEIN:0:7}" != "root://" ]; then FILEIN="file:${FILEIN}" fi
if [ "${FILEOUT:0:7}" != "root://" ]; then FILEOUT="file:${FILEOUT}" fi

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
wget https://coli.web.cern.ch/coli/tmp/.231024-123346_partv2_ak15_testonly/model.onnx -O PhysicsTools/NanoTuples/data/InclParticleTransformer-MD/ak15/V02/model.onnx
scram b -j$(cat /proc/cpuinfo | grep MHz | wc -l)

# 6_HIG-RunIISummer20UL18CustomizedNanoAODv9-02334.sh
cmsDriver.py \
    --mc \
    -n "${NEVENT}" \
    --nThreads "${NTHREAD}" \
    --python_filename HIG-RunIISummer20UL18CustomizedNanoAOD.py \
    --eventcontent NANOAODSIM \
    --datatier NANOAODSIM \
    --conditions 106X_upgrade2018_realistic_v16_L1v1 \
    --step NANO \
    --era Run2_2018,run2_nanoAOD_106Xv2 \
    --customise PhysicsTools/NanoTuples/nanoTuples_cff.nanoTuples_customizeMC \
    --filein "${FILEIN}" \
    --fileout "${FILEOUT}" \
