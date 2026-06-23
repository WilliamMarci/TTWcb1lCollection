#!/usr/bin/env bash

set -e

FILE="${1:-shapes_vcb_dbc0.7_cv0.7.root}"
MODE="${2:-combined}"      # combined or single
REGION="${3:-sr}"          # used only in single mode
OUTDIR="${4:-figure}"
CV="${5:-0.7}"

PROCESSES=(
  sig
  bkg_wqq
  bkg_topbc
  bkg_other
)

SYSTEMATICS=(
  PU
  L1PreFiring
  TrigEff
  ElEff
  MuEff
  FT_JER
  FT_JES
  Theory_muR
  Theory_muF
)

mkdir -p "${OUTDIR}"

echo "Checking shapes in file: ${FILE}"
echo "Mode: ${MODE}"
echo "Output dir: ${OUTDIR}"
if [[ "${MODE}" == "single" ]]; then
  echo "Region: ${REGION}"
else
  echo "Combined regions: calib sr"
  echo "CV threshold: ${CV}"
fi
echo

for PROC in "${PROCESSES[@]}"; do
  for SYST in "${SYSTEMATICS[@]}"; do
    echo ">>> Checking process=${PROC}, syst=${SYST}"

    if [[ "${MODE}" == "single" ]]; then
      python3 checkshape.py \
        -f "${FILE}" \
        -r "${REGION}" \
        -p "${PROC}" \
        -s "${SYST}" \
        -o "${OUTDIR}" \
        --logy || true

    elif [[ "${MODE}" == "combined" ]]; then
      python3 checkshape.py \
        -f "${FILE}" \
        --combine-regions calib sr \
        --cv "${CV}" \
        -p "${PROC}" \
        -s "${SYST}" \
        -o "${OUTDIR}" \
        --logy || true

    else
      echo "[ERROR] Unknown mode: ${MODE}"
      echo "        Allowed values: single, combined"
      exit 1
    fi

    echo
  done
done

echo "All checks finished."
