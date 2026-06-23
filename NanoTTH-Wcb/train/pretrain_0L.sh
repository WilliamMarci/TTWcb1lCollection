#!/bin/bash

set -x

echo "args: $@"

DATADIR=$1

model=$2
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/ParT.py"
    batchopts="--batch-size 2048 --start-lr 4e-3"
elif [[ "$model" == "PNXT" ]]; then
    modelopts="networks/PNXT.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/PN.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "MLP" ]]; then
    modelopts="networks/MLP.py"
    batchopts="--batch-size 4096 --start-lr 1e-1"
else
    echo "Invalid model $model!"
    exit 1
fi

# set a comment via `COMMENT`
suffix=${COMMENT}

weaver --data-train \
    "ttHTocc:${DATADIR}/mc/ttHTocc_tree.root" \
    "ttHTobb:${DATADIR}/mc/ttHTobb_tree.root" \
    "TTZToQQ:${DATADIR}/mc/TTZToQQ_tree.root" \
    "TTToHadronic:${DATADIR}/mc/TTToHadronic_tree.root" \
    "TTToSemiLeptonic:${DATADIR}/mc/TTToSemiLeptonic_tree.root" \
    "TTZToQQ-EFT:${DATADIR}/mc/TTZToQQ-EFT_tree.root" \
    "QCD_HT300to500:${DATADIR}/mc/QCD_HT300to500_tree.root" \
    "QCD_HT500to700:${DATADIR}/mc/QCD_HT500to700_tree.root" \
    "QCD_HT700to1000:${DATADIR}/mc/QCD_HT700to1000_tree.root" \
    "QCD_HT1000to1500:${DATADIR}/mc/QCD_HT1000to1500_tree.root" \
    "QCD_HT1500to2000:${DATADIR}/mc/QCD_HT1500to2000_tree.root" \
    "QCD_HT2000toInf:${DATADIR}/mc/QCD_HT2000toInf_tree.root" \
    "QCD_Pt_120to170:${DATADIR}/mc/QCD_Pt_120to170_tree.root" \
    "QCD_Pt_170to300:${DATADIR}/mc/QCD_Pt_170to300_tree.root" \
    "QCD_Pt_300to470:${DATADIR}/mc/QCD_Pt_300to470_tree.root" \
    "QCD_Pt_470to600:${DATADIR}/mc/QCD_Pt_470to600_tree.root" \
    "QCD_Pt_600to800:${DATADIR}/mc/QCD_Pt_600to800_tree.root" \
    "QCD_Pt_800to1000:${DATADIR}/mc/QCD_Pt_800to1000_tree.root" \
    "QCD_Pt_1000to1400:${DATADIR}/mc/QCD_Pt_1000to1400_tree.root" \
    "QCD_Pt_1800to2400:${DATADIR}/mc/QCD_Pt_1800to2400_tree.root" \
    "QCD_Pt_2400to3200:${DATADIR}/mc/QCD_Pt_2400to3200_tree.root" \
    "QCD_Pt_3200toInf:${DATADIR}/mc/QCD_Pt_3200toInf_tree.root" \
    "WJetsToQQ_HT-200to400:${DATADIR}/mc/WJetsToQQ_HT-200to400_tree.root" \
    "WJetsToQQ_HT-400to600:${DATADIR}/mc/WJetsToQQ_HT-400to600_tree.root" \
    "WJetsToQQ_HT-600to800:${DATADIR}/mc/WJetsToQQ_HT-600to800_tree.root" \
    "WJetsToQQ_HT-800toInf:${DATADIR}/mc/WJetsToQQ_HT-800toInf_tree.root" \
    "ZJetsToQQ_HT-200to400:${DATADIR}/mc/ZJetsToQQ_HT-200to400_tree.root" \
    "ZJetsToQQ_HT-400to600:${DATADIR}/mc/ZJetsToQQ_HT-400to600_tree.root" \
    "ZJetsToQQ_HT-600to800:${DATADIR}/mc/ZJetsToQQ_HT-600to800_tree.root" \
    "ZJetsToQQ_HT-800toInf:${DATADIR}/mc/ZJetsToQQ_HT-800toInf_tree.root" \
    --samples-per-epoch 2560000 --train-val-split 0.8 \
    --data-config data/ttH_0L_${model}_pretrain.yaml --network-config $modelopts \
    --model-prefix training/ttH_0L/${model}/{auto}${suffix}/net \
    --num-workers 1 --fetch-step 0.2 $batchopts \
    --num-epochs 50 --gpus 0 \
    --optimizer ranger --log logs/ttH_0L_${model}_{auto}${suffix}.log \
    --cross-validation 'event%5' \
    "${@:3}"
