#!/bin/bash

inputdir=/eos/cms/store/cmst3/group/hh/trees/hqu/20240224_dev_ak4puppi_2017_0L/mc

workdir=/afs/cern.ch/user/h/hqu/work/nn/ttH/train
modeldir=/eos/cms/store/cmst3/user/hqu/training/ttH
condadir=/afs/cern.ch/work/h/hqu/miniconda3

model=ParT
suffix="_20240224_dev_ak4puppi_pre+ttHinc+ttZEFT+ttZBB+TTbb+QCDPythia"

. "${condadir}/etc/profile.d/conda.sh"
conda activate weaver

nvcc --version
nvidia-smi
export TMPDIR=$(pwd)
set -x

# copy data
DATADIR="${TMPDIR}/_tmp_data"
mkdir ${DATADIR}
rsync -avL ${inputdir}/*.root ${DATADIR}/

if [[ "$model" == "ParT" ]]; then
    modelopts="${workdir}/networks/ParT.py"
    batchopts="--batch-size 2048 --start-lr 4e-3 --use-amp"
elif [[ "$model" == "PNXT" ]]; then
    modelopts="${workdir}/networks/PNXT.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PN" ]]; then
    modelopts="${workdir}/networks/PN.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "MLP" ]]; then
    modelopts="${workdir}/networks/MLP.py"
    batchopts="--batch-size 4096 --start-lr 1e-1"
else
    echo "Invalid model $model!"
    exit 1
fi

weaver --data-train \
    "TTHTo2C_TTToHadronic:${DATADIR}/TTHTo2C_TTToHadronic_tree.root" \
    "TTHTo2C_TTToSemiLep:${DATADIR}/TTHTo2C_TTToSemiLep_tree.root" \
    "TTHTo2C_TTTo2L2Nu:${DATADIR}/TTHTo2C_TTTo2L2Nu_tree.root" \
    "TTHTo2B_TTToHadronic:${DATADIR}/TTHTo2B_TTToHadronic_tree.root" \
    "TTHTo2B_TTToSemiLep:${DATADIR}/TTHTo2B_TTToSemiLep_tree.root" \
    "TTHTo2B_TTTo2L2Nu:${DATADIR}/TTHTo2B_TTTo2L2Nu_tree.root" \
    "TTZToQQ:${DATADIR}/TTZToQQ_tree.root" \
    "TTToHadronic:${DATADIR}/TTToHadronic_tree.root" \
    "TTToSemiLeptonic:${DATADIR}/TTToSemiLeptonic_tree.root" \
    "TTTo2L2Nu:${DATADIR}/TTTo2L2Nu_tree.root" \
    "QCD_HT300to500:${DATADIR}/QCD_HT300to500_tree.root" \
    "QCD_HT500to700:${DATADIR}/QCD_HT500to700_tree.root" \
    "QCD_HT700to1000:${DATADIR}/QCD_HT700to1000_tree.root" \
    "QCD_HT1000to1500:${DATADIR}/QCD_HT1000to1500_tree.root" \
    "QCD_HT1500to2000:${DATADIR}/QCD_HT1500to2000_tree.root" \
    "QCD_HT2000toInf:${DATADIR}/QCD_HT2000toInf_tree.root" \
    "WJetsToQQ_HT-200to400:${DATADIR}/WJetsToQQ_HT-200to400_tree.root" \
    "WJetsToQQ_HT-400to600:${DATADIR}/WJetsToQQ_HT-400to600_tree.root" \
    "WJetsToQQ_HT-600to800:${DATADIR}/WJetsToQQ_HT-600to800_tree.root" \
    "WJetsToQQ_HT-800toInf:${DATADIR}/WJetsToQQ_HT-800toInf_tree.root" \
    "ZJetsToQQ_HT-200to400:${DATADIR}/ZJetsToQQ_HT-200to400_tree.root" \
    "ZJetsToQQ_HT-400to600:${DATADIR}/ZJetsToQQ_HT-400to600_tree.root" \
    "ZJetsToQQ_HT-600to800:${DATADIR}/ZJetsToQQ_HT-600to800_tree.root" \
    "ZJetsToQQ_HT-800toInf:${DATADIR}/ZJetsToQQ_HT-800toInf_tree.root" \
    "ttHTocc:${DATADIR}/ttHTocc_tree.root" \
    "ttHTobb:${DATADIR}/ttHTobb_tree.root" \
    "TTZToQQ-EFT:${DATADIR}/TTZToQQ-EFT_tree.root" \
    "TTZToBB:${DATADIR}/TTZToBB_tree.root" \
    "TTbb_4f_TTToHadronic:${DATADIR}/TTbb_4f_TTToHadronic_tree.root" \
    "TTbb_4f_TTToSemiLeptonic:${DATADIR}/TTbb_4f_TTToSemiLeptonic_tree.root" \
    "TTbb_4f_TTTo2L2Nu:${DATADIR}/TTbb_4f_TTTo2L2Nu_tree.root" \
    "QCD_Pt_120to170:${DATADIR}/QCD_Pt_120to170_tree.root" \
    "QCD_Pt_170to300:${DATADIR}/QCD_Pt_170to300_tree.root" \
    "QCD_Pt_300to470:${DATADIR}/QCD_Pt_300to470_tree.root" \
    "QCD_Pt_470to600:${DATADIR}/QCD_Pt_470to600_tree.root" \
    "QCD_Pt_600to800:${DATADIR}/QCD_Pt_600to800_tree.root" \
    "QCD_Pt_800to1000:${DATADIR}/QCD_Pt_800to1000_tree.root" \
    "QCD_Pt_1000to1400:${DATADIR}/QCD_Pt_1000to1400_tree.root" \
    "QCD_Pt_1800to2400:${DATADIR}/QCD_Pt_1800to2400_tree.root" \
    "QCD_Pt_2400to3200:${DATADIR}/QCD_Pt_2400to3200_tree.root" \
    "QCD_Pt_3200toInf:${DATADIR}/QCD_Pt_3200toInf_tree.root" \
    --samples-per-epoch 2560000 --train-val-split 0.8 \
    --data-config ${workdir}/data/ttH_0L_${model}_pretrain.yaml --network-config $modelopts \
    --model-prefix ${modeldir}/ttH_0L/${model}/{auto}${suffix}/net \
    --num-workers 1 --fetch-step 0.2 $batchopts \
    --num-epochs 50 --gpus 0 \
    --optimizer ranger --log ${workdir}/logs/ttH_0L_${model}_{auto}${suffix}.log \
    --cross-validation 'event%5'
