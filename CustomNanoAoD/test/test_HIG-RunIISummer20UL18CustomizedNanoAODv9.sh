#!/bin/bash

../scripts/HIG-RunIISummer20UL18CustomizedNanoAODv9.sh 100 \
    $(cat /proc/cpuinfo | grep MHz | wc -l) \
    "$(readlink -f HIG-RunIISummer20UL18MiniAODv2-02334_15946930_1001_1.root)"
