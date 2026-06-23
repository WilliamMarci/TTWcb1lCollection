#!/bin/bash

inputdir=/eos/cms/store/cmst3/group/hh/trees/hqu/20240224_dev_ak4puppi_2017_2L/mc
loadmodel=/eos/cms/store/cmst3/user/hqu/training/ttH/ttH_1L/ParT/20240225-051423_ParT_ranger_lr0.0001_batch2048_20240224_dev_ak4puppi_FineTune

workdir=/afs/cern.ch/user/h/hqu/work/nn/ttH/train
modeldir=/eos/cms/store/cmst3/user/hqu/training/ttH
condadir=/afs/cern.ch/work/h/hqu/miniconda3

model=ParT
suffix="_20240224_dev_ak4puppi_FineTune_1LFinal"

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
    --samples-per-epoch 512000 --train-val-split 0.8 \
    --data-config ${workdir}/data/ttH_2L_${model}.yaml --network-config $modelopts \
    --model-prefix ${modeldir}/ttH_2L/${model}/{auto}${suffix}/net \
    --num-workers 1 --fetch-step 0.2 $batchopts \
    --num-epochs 20 --gpus 0 \
    --optimizer ranger --log ${workdir}/logs/ttH_2L_${model}_{auto}${suffix}.log \
    --load-model-weights ${loadmodel}_{fold}/net_best_epoch_state.pt \
    --start-lr 1e-4 --exclude-model-weights 'part.fc..+' --optimizer-option lr_mult '("part.fc.*", 10)' --num-epochs 20 \
    --cross-validation 'event%5'
