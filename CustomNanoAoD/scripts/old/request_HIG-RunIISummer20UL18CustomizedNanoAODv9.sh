#!/bin/bash

CONTENT='Universe = vanilla
Executable = __TO_BE_REPLACED_EXECUTABLE

+ProjectName="cms.org.cern"

NEVENT = 100
NTHREAD = 4
FILEIN = /eos/user/l/legao/hss/src/aod/HIG-RunIISummer20UL18/miniaod/HIG-RunIISummer20UL18MiniAODv2-02334_$(ID).root
FILEOUT = /eos/user/l/legao/hss/src/aod/HIG-RunIISummer20UL18/nanoaod/HIG-RunIISummer20UL18CustomizedNanoAODv9-02334_$(ID).root
X509UP = __TO_BE_REPLACED_X509UP
Arguments = $(NEVENT) $(NTHREAD) $(FILEIN) $(FILEOUT) $(X509UP)

requirements = (OpSysAndVer =?= "CentOS7")
request_cpus = 4
request_memory = 4096
x509userproxy = $(X509UP)

+JobFlavour = "tomorrow"

Log    = log/HIG-RunIISummer20UL18CustomizedNanoAODv9_$(Cluster)_system.log
Output = log/HIG-RunIISummer20UL18CustomizedNanoAODv9_$(ID)_1.log
Error  = log/HIG-RunIISummer20UL18CustomizedNanoAODv9_$(ID)_2.log

should_transfer_files = NO
Queue ID from request_HIG-RunIISummer20UL18CustomizedNanoAODv9.txt'

CONTENT="${CONTENT/__TO_BE_REPLACED_EXECUTABLE/$(readlink -f HIG-RunIISummer20UL18CustomizedNanoAODv9.sh)}"
CONTENT="${CONTENT/__TO_BE_REPLACED_X509UP/$(readlink -f x509up)}"
echo "${CONTENT}" > request_HIG-RunIISummer20UL18CustomizedNanoAODv9.jdl

for ID in $(ls /eos/user/l/legao/hss/src/aod/HIG-RunIISummer20UL18/miniaod/ \
    | grep '^HIG-RunIISummer20UL18MiniAODv2-02334_[0-9]\+_[0-9]\+_[0-9]\+\.root$' \
    | grep -o '[0-9]\+_[0-9]\+_[0-9]\+\.root$' \
    | grep -o '[0-9]\+_[0-9]\+_[0-9]\+'); do
    if [ $(stat --printf=%s \
        /eos/user/l/legao/hss/src/aod/HIG-RunIISummer20UL18/miniaod/HIG-RunIISummer20UL18MiniAODv2-02334_${ID}.root \
        ) -lt 1000 ]; then
        continue
    fi
    if 2>/dev/null [ $(2>/dev/null stat --printf=%s \
        /eos/user/l/legao/hss/src/aod/HIG-RunIISummer20UL18/nanoaod/HIG-RunIISummer20UL18CustomizedNanoAODv9-02334_${ID}.root \
        ) -ge 1000 ]; then
        echo "Skipping: HIG-RunIISummer20UL18CustomizedNanoAODv9-02334_${ID}.root" >&2
        continue
    fi
    echo "Pending: HIG-RunIISummer20UL18CustomizedNanoAODv9-02334_${ID}.root" >&2
    echo ${ID}
done > request_HIG-RunIISummer20UL18CustomizedNanoAODv9.txt
