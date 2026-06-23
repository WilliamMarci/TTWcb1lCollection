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
    "TTHTo2C_TTToHadronic:${DATADIR}/mc/TTHTo2C_TTToHadronic_tree.root" \
    "TTHTo2C_TTToSemiLep:${DATADIR}/mc/TTHTo2C_TTToSemiLep_tree.root" \
    "TTHTo2C_TTTo2L2Nu:${DATADIR}/mc/TTHTo2C_TTTo2L2Nu_tree.root" \
    "TTHTo2B_TTToHadronic:${DATADIR}/mc/TTHTo2B_TTToHadronic_tree.root" \
    "TTHTo2B_TTToSemiLep:${DATADIR}/mc/TTHTo2B_TTToSemiLep_tree.root" \
    "TTHTo2B_TTTo2L2Nu:${DATADIR}/mc/TTHTo2B_TTTo2L2Nu_tree.root" \
    "TTZToQQ:${DATADIR}/mc/TTZToQQ_tree.root" \
    "TTToHadronic:${DATADIR}/mc/TTToHadronic_tree.root" \
    "TTToSemiLeptonic:${DATADIR}/mc/TTToSemiLeptonic_tree.root" \
    "TTTo2L2Nu:${DATADIR}/mc/TTTo2L2Nu_tree.root" \
    --samples-per-epoch 2560000 --train-val-split 0.8 \
    --data-config data/ttH_1L_${model}.yaml --network-config $modelopts \
    --model-prefix training/ttH_1L/${model}/{auto}${suffix}/net \
    --num-workers 1 --fetch-step 0.2 $batchopts \
    --num-epochs 50 --gpus 0 \
    --optimizer ranger --log logs/ttH_1L_${model}_{auto}${suffix}.log \
    --cross-validation 'event%5' \
    "${@:3}"
