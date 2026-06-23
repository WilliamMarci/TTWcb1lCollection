#! /usr/bin/env python3
import json
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("json_file", help="Input JSON file from step 1")
parser.add_argument("--lumi_scale", type=float, default=41.5, help="Factor to divide MC yields by (to fix double counting)")
args = parser.parse_args()

try:
    with open(args.json_file) as f:
        data = json.load(f)
except Exception as e:
    print(f"Error loading JSON: {e}")
    sys.exit(1)


procs = ["sig", "bkg_wqq", "bkg_topbc", "bkg_other"]
# data
obs_pass = data["pass"].get("data_obs", {}).get("nominal", 0.0)
obs_fail = data["fail"].get("data_obs", {}).get("nominal", 0.0)


def get_rate(region, proc):
    val = data[region].get(proc, {}).get("nominal", 0.0)
    val = val / args.lumi_scale
    return max(val, 0.0001)

rates_pass = [get_rate("pass", p) for p in procs]
rates_fail = [get_rate("fail", p) for p in procs]

print(f"Correction: Dividing MC yields by {args.lumi_scale}...")
print(f"Pass Region: Data={obs_pass}, MC_Total={sum(rates_pass):.2f}")
print(f"Fail Region: Data={obs_fail}, MC_Total={sum(rates_fail):.2f}")


card = f"""
# Vcb Analysis Datacard
# Auto-generated from {args.json_file}
imax 2
jmax 3
kmax *
-------------------------------------------------------------------------
bin          boosted_pass   boosted_fail
observation  {obs_pass:.1f}          {obs_fail:.1f}
-------------------------------------------------------------------------
bin          boosted_pass  boosted_pass  boosted_pass  boosted_pass    boosted_fail  boosted_fail  boosted_fail  boosted_fail
process      sig           bkg_wqq       bkg_topbc     bkg_other       sig           bkg_wqq       bkg_topbc     bkg_other
process      0             1             2             3               0             1             2             3
rate         {rates_pass[0]:.4f}      {rates_pass[1]:.4f}      {rates_pass[2]:.4f}      {rates_pass[3]:.4f}        {rates_fail[0]:.4f}      {rates_fail[1]:.4f}      {rates_fail[2]:.4f}      {rates_fail[3]:.4f}
-------------------------------------------------------------------------
"""

# Systematics 
MIN_YIELD_FOR_SYST = 0.01

sample_syst_keys = data["pass"].get("sig", {}).keys()
syst_list = sorted([k for k in sample_syst_keys if k != "nominal"])

for syst in syst_list:
    line = f"{syst:<25} lnN  "
    for region in ["pass", "fail"]:
        for p in procs:
            raw_nom = data[region].get(p, {}).get("nominal", 0.0)
            corrected_nom = raw_nom / args.lumi_scale
            
            if p in data[region] and syst in data[region][p] and corrected_nom > MIN_YIELD_FOR_SYST:
                dn, up = data[region][p][syst]
                
                # Clip 
                if dn <= 0.01: dn = 0.01
                if up <= 0.01: up = 0.01
                if dn > 10: dn = 10.0
                if up > 10: up = 10.0
                
                line += f"{dn:.4f}/{up:.4f}  "
            else:
                line += "-            "
    card += line + "\n"

card += "* autoMCStats 0\n"

out_file = args.json_file.replace(".json", ".txt")
with open(out_file, "w") as f:
    f.write(card)

print(f"Datacard written to {out_file}")
