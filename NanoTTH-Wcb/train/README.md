# ttH training setup

## Preliminary

Setup `weaver-core`:

- instructions at: https://github.com/hqucms/weaver-core

## Training procedure

### FH channel

- pretraining w/ additional samples (QCD_Pythia, ttZ-EFT samples) and lower ht threshold (ht>300)
  - [batch/run_0Lpre.sh](train/batch/run_0Lpre.sh)
- fine-tuning w/ the nominal set of samples and ht>500
  - [train_0L.sh](train_0L.sh)
  - reference accuracy: 0.3328+/-0.0005

```bash
./train_0L.sh /data/hqu/trees/20230924_dev_ak4puppi_2017_0L ParT --gpus 0 --use-amp --load-model-weights training/ttH_0L/ParT/20230924-193552_ParT_ranger_lr0.004_batch2048_20230918_dev_ak4puppi_pre_+ttZEFT+QCDPythia+UL18QCD_{fold}/net_best_epoch_state.pt --start-lr 1e-4 --exclude-model-weights 'fc..+' --optimizer-option lr_mult '("fc.*", 10)' --num-epochs 20
```

### SL channel

- pretraining w/ additional samples (TTZToQQ-EFT, ttHTobb 1L/2L)
  - [batch/run_1Lpre.sh](train/batch/run_1Lpre.sh)
- fine-tuning w/ the nominal set of samples
  - [train_1L.sh](train_1L.sh)
  - reference accuracy: 0.3694+/-0.0003

```bash
./train_1L.sh /data/hqu/trees/20230918_dev_ak4puppi_2017_1L ParT --gpus 0 --use-amp --load-model-weights training/ttH_1L/ParT/20230919-001215_ParT_ranger_lr0.004_batch2048_20230918_dev_ak4puppi_pre_+ttHbbExtra+ttZEFT_{fold}/net_best_epoch_state.pt --start-lr 1e-4 --exclude-model-weights 'part.fc..+' --optimizer-option lr_mult '("part.fc.*", 10)' --num-epochs 20
```

### DL channel

- fine-tuning starting from the final (pre-trained+fine-tuned) SL model
  - [train_2L.sh](train_2L.sh)
  - reference accuracy: 0.3980+/-0.0005

```bash
./train_2L.sh /data/hqu/trees/20230918_dev_ak4puppi_2017_2L ParT --gpus 0 --use-amp --load-model-weights training/ttH_1L/ParT/20230919-210432_ParT_ranger_lr0.0001_batch2048_20230918_dev_ak4puppi_FineTune_reset_fc_{fold}/net_best_epoch_state.pt --start-lr 1e-4 --exclude-model-weights 'part.fc..+' --optimizer-option lr_mult '("part.fc.*", 10)' --num-epochs 20
```

### Conversion of trained models to ONNX

```bash
export chn=0L; export modeldir=""; for i in {0..4}; do weaver -c data/ttH_${chn}_ParT.yaml -n networks/ParT.py -m ${modeldir}_fold${i}/net_best_epoch_state.pt --export-onnx ${modeldir}_fold${i}/net.${i}.onnx; done
```
