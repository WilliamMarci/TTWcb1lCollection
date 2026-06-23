# NanoAOD production

Produce custom NanoAODs with ParticleNetAK4.

<!-- TOC -->

- [NanoAOD production](#nanoaod-production)
    - [Version](#version)
    - [Setup](#setup)
    - [Production](#production)

<!-- /TOC -->

---

## Version

The current version is based on [NanoAODv9](https://gitlab.cern.ch/cms-nanoAOD/nanoaod-doc/-/wikis/Releases/NanoAODv9).

Customizations:

- Add ParticleNetAK4 for AK4 jets

---

## Setup

```bash
cmsrel CMSSW_10_6_26
cd CMSSW_10_6_26/src
cmsenv
```

## Production

**Step 0**: switch to the crab production directory and set up grid proxy, CRAB environment, etc.

```bash
# set up grid proxy
voms-proxy-init -rfc -voms cms --valid 168:00
# set up CRAB env (must be done after cmsenv)
source /cvmfs/cms.cern.ch/common/crab-setup.sh
```

**Step 1**: generate the python config file with `generateConfigs.sh`.

**Step 2**: use the `crab.py` script to submit the CRAB jobs:

For MC:

```bash
python crab.py -p mc_2018UL_NANO.py --max-memory 2500 --site T3_US_FNALLPC -o /store/group/lpcjme/NanoAOD/NanoAODv9-ParticleNetAK4/2018/mc -t NanoTuples-24Apr2023_NanoAODv9_ParticleNetAK4 -i mc/mc_2018.conf --num-cores 1 -s FileBased -n 2 --work-area crab_projects_2018UL --dryrun

python crab.py -p mc_2017UL_NANO.py --max-memory 2500 --site T2_CH_CERN -o /store/group/cmst3/group/vhcc/NanoAOD/NanoAODv9-ParticleNetAK4/2017/mc -t NanoTuples-20Apr2023_NanoAODv9_ParticleNetAK4 -i mc/mc_2017.conf --num-cores 1 -s FileBased -n 2 --work-area crab_projects_2017UL --dryrun

python crab.py -p mc_2016ULpostVFP_NANO.py --max-memory 2500 --site T3_US_FNALLPC -o /store/group/lpcjme/NanoAOD/NanoAODv9-ParticleNetAK4/2016/mc -t NanoTuples-24Apr2023_NanoAODv9_ParticleNetAK4 -i mc/mc_2016post.conf --num-cores 1 -s FileBased -n 2 --work-area crab_projects_2016ULpostVFP --dryrun

python crab.py -p mc_2016ULpreVFP_NANO.py --max-memory 2500 --site T3_US_FNALLPC -o /store/group/lpcjme/NanoAOD/NanoAODv9-ParticleNetAK4/2016APV/mc -t NanoTuples-24Apr2023_NanoAODv9_ParticleNetAK4 -i mc/mc_2016pre.conf --num-cores 1 -s FileBased -n 2 --work-area crab_projects_2016ULpreVFP --dryrun
```

For Data:

```bash
python crab.py -p data_2018UL_NANO.py --max-memory 2500 --site T3_US_FNALLPC -o /store/group/lpcjme/NanoAOD/NanoAODv9-ParticleNetAK4/2018/data -t NanoTuples-24Apr2023_NanoAODv9_ParticleNetAK4 -i data/data_2018.conf --num-cores 1 -s EventAwareLumiBased -n 100000 -j 'https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions18/13TeV/Legacy_2018/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt' --work-area crab_projects_data_2018UL --dryrun

python crab.py -p data_2017UL_NANO.py --max-memory 2500 --site T3_US_FNALLPC -o /store/group/lpcjme/NanoAOD/NanoAODv9-ParticleNetAK4/2017/data -t NanoTuples-24Apr2023_NanoAODv9_ParticleNetAK4 -i data/data_2017.conf --num-cores 1 -s EventAwareLumiBased -n 100000 -j 'https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt' --work-area crab_projects_data_2017UL --dryrun

python crab.py -p data_2016ULpostVFP_NANO.py --max-memory 2500 --site T3_US_FNALLPC -o /store/group/lpcjme/NanoAOD/NanoAODv9-ParticleNetAK4/2016/data -t NanoTuples-24Apr2023_NanoAODv9_ParticleNetAK4 -i data/data_2016post.conf --num-cores 1 -s EventAwareLumiBased -n 100000 -j 'https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions16/13TeV/Legacy_2016/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt' --work-area crab_projects_data_2016ULpostVFP --dryrun

python crab.py -p data_2016ULpreVFP_NANO.py --max-memory 2500 --site T3_US_FNALLPC -o /store/group/lpcjme/NanoAOD/NanoAODv9-ParticleNetAK4/2016APV/data -t NanoTuples-24Apr2023_NanoAODv9_ParticleNetAK4 -i data/data_2016pre.conf --num-cores 1 -s EventAwareLumiBased -n 100000 -j 'https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions16/13TeV/Legacy_2016/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt' --work-area crab_projects_data_2016ULpreVFP --dryrun
```

These command will perform a "dryrun" to print out the CRAB configuration files. Please check everything is correct (e.g., the output path, version number, requested number of cores, etc.) before submitting the actual jobs. To actually submit the jobs to CRAB, just remove the `--dryrun` option at the end.

**Step 3**: check job status

The status of the CRAB jobs can be checked with:

```bash
./crab.py --status --work-area crab_projects_*  --options "maxjobruntime=2500 maxmemory=2500" && ./crab.py --summary
```

Note that this will also **resubmit** failed jobs automatically.

The crab dashboard can also be used to get a quick overview of the job status:

- [https://monit-grafana.cern.ch/d/cmsTMGlobal/cms-tasks-monitoring-globalview?orgId=11](https://monit-grafana.cern.ch/d/cmsTMGlobal/cms-tasks-monitoring-globalview?orgId=11)

More options of this `crab.py` script can be found with:

```bash
./crab.py -h
```
