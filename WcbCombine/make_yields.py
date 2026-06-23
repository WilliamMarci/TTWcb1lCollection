#! /usr/bin/env python3
import uproot
import awkward as ak
import numpy as np
import glob
import os
import json
import argparse
import fnmatch
import sys


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc=""):
        return iterable

# parameter parsing
parser = argparse.ArgumentParser()
parser.add_argument("--dbc", type=float, default=0.7, help="Cut on Dbc score")
parser.add_argument("--cv", type=float, default=0.7, help="Cut on Event Classifier score")
parser.add_argument("--lumi", type=float, default=41.5, help="Luminosity in fb-1")
parser.add_argument("--chunksize", type=int, default=100000, help="Number of events per chunk")

DEFAULT_MC = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/MC/scored_samples_2final/*.root"
DEFAULT_DATA = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/data/scored_data/merged/*.root"

parser.add_argument("--mc_path", type=str, default=DEFAULT_MC, help="Path pattern for MC files")
parser.add_argument("--data_path", type=str, default=DEFAULT_DATA, help="Path pattern for Data files")

args = parser.parse_args()

MC_PATH = args.mc_path
DATA_PATH = args.data_path

print(f"Config: Dbc>{args.dbc}, CV>{args.cv}")
print(f"MC Path:   {MC_PATH}")
print(f"Data Path: {DATA_PATH}")

# 1. Flavor Tagging Systematics
FLAV_TAG_SOURCES = [
    "JER", "JES", "PUWeight", 
    "LHEScaleWeight_muF_ttbar", "LHEScaleWeight_muF_wjets", "LHEScaleWeight_muF_zjets",
    "LHEScaleWeight_muR_ttbar", "LHEScaleWeight_muR_wjets", "LHEScaleWeight_muR_zjets",
    "PSWeightISR_ttbar", "PSWeightISR_wjets", "PSWeightISR_zjets",
    "PSWeightFSR_ttbar", "PSWeightFSR_wjets", "PSWeightFSR_zjets",
    "XSec_WJets_c", "XSec_WJets_b", "XSec_ZJets_c", "XSec_ZJets_b",
    "Stat_flavB_C0", "Stat_flavB_C1", "Stat_flavB_C2", "Stat_flavB_C3", "Stat_flavB_C4",
    "Stat_flavB_B0", "Stat_flavB_B1", "Stat_flavB_B2", "Stat_flavB_B3", "Stat_flavB_B4",
    "Stat_flavC_C0", "Stat_flavC_C1", "Stat_flavC_C2", "Stat_flavC_C3", "Stat_flavC_C4",
    "Stat_flavC_B0", "Stat_flavC_B1", "Stat_flavC_B2", "Stat_flavC_B3", "Stat_flavC_B4",
    "Stat_flavL_C0", "Stat_flavL_C1", "Stat_flavL_C2", "Stat_flavL_C3", "Stat_flavL_C4",
    "Stat_flavL_B0", "Stat_flavL_B1", "Stat_flavL_B2", "Stat_flavL_B3", "Stat_flavL_B4",
]

# 2. Ratio Systematics
RATIO_SYSTEMATICS = {
    "PU":           ("puWeightUp",          "puWeightDown",          "puWeight"),
    "L1PreFiring":  ("l1PreFiringWeightUp", "l1PreFiringWeightDown", "l1PreFiringWeight"),
    "TrigEff":      ("trigEffWeightUp",     "trigEffWeightDown",     "trigEffWeight"),
    "ElEff":        ("elEffWeight_UP",      "elEffWeight_DOWN",      "elEffWeight"),
    "MuEff":        ("muEffWeight_UP",      "muEffWeight_DOWN",      "muEffWeight"),
}

# 3. Renormalization/Theory Weights
RENORM_SOURCES = [
    "muR", "muF", "isr", "fsr", 
    "pdfSum", "pdfSumWAlphaS", "alphas", 
    "topPt", "hdamp", "hdampML"
]

RENORM_DETAILED = [
    "fsr_G2GG_muR", "fsr_G2QQ_muR", "fsr_Q2QG_muR", "fsr_X2XG_muR",
    "fsr_G2GG_cNS", "fsr_G2QQ_cNS", "fsr_Q2QG_cNS", "fsr_X2XG_cNS",
    "isr_G2GG_muR", "isr_G2QQ_muR", "isr_Q2QG_muR", "isr_X2XG_muR",
    "isr_G2GG_cNS", "isr_G2QQ_cNS", "isr_Q2QG_cNS", "isr_X2XG_cNS"
]

def get_leading(array, default=0):
    return ak.fill_none(ak.firsts(array), default)

def process_chunk(events, is_data):
    # --- 1. Event Selection ---
    g_bc = get_leading(events.ak8_gpt_bc)
    denom_d = (g_bc + get_leading(events.ak8_gpt_qcd) + get_leading(events.ak8_gpt_cc) + 
               get_leading(events.ak8_gpt_bb) + get_leading(events.ak8_gpt_bs) + 
               get_leading(events.ak8_gpt_cs) + get_leading(events.ak8_gpt_qq) + 
               get_leading(events.ak8_gpt_topbw) + 1e-10)
    dbc_score = g_bc / denom_d

    s_w_qq = events.score_cata_w_qq
    denom_c = (s_w_qq + events.score_cata_qcd + events.score_cata_top_bqq + 
               events.score_cata_top_bc + events.score_cata_top_bq + 
               events.score_cata_non + 1e-10)
    class_score = s_w_qq / denom_c

    has_fatjet = ak.num(events.ak8_pt) > 0
    
    mask_stage1 = has_fatjet & (dbc_score > args.dbc)
    mask_pass = mask_stage1 & (class_score > args.cv)
    mask_fail = mask_stage1 & (class_score <= args.cv)

    # --- 2. Weight Calculation ---
    if not is_data:
        prefiring = events.l1PreFiringWeight if "l1PreFiringWeight" in events.fields else 1.0
        
        base_w = (events.genWeight * events.lumiwgt * events.puWeight * 
                  events.trigEffWeight * events.elEffWeight * events.muEffWeight * 
                  events.xsecWeight * prefiring)
        
        w_nom = base_w * events.flavTagWeight
    else:
        w_nom = ak.ones_like(dbc_score)
        base_w = None

    # --- 3. Fill Histograms ---
    chunk_res = {"pass": {}, "fail": {}}

    def fill(region_mask):
        masked_w_nom = w_nom[region_mask]
        nom_val = ak.sum(masked_w_nom)
        if not is_data: nom_val *= args.lumi
        
        res = {"nominal": float(nom_val)}
        
        if not is_data:
            masked_base_w = base_w[region_mask]
            masked_events = events[region_mask]
            
            def calc_syst_yield(weight_array):
                return float(ak.sum(weight_array) * args.lumi)

            # A. FlavTag
            for source in FLAV_TAG_SOURCES:
                name_up = f"flavTagWeight_{source}_UP"
                name_dn = f"flavTagWeight_{source}_DOWN"
                val_up = nom_val
                val_dn = nom_val
                if name_up in masked_events.fields:
                    val_up = calc_syst_yield(masked_base_w * masked_events[name_up])
                if name_dn in masked_events.fields:
                    val_dn = calc_syst_yield(masked_base_w * masked_events[name_dn])
                res[f"FT_{source}"] = [val_dn, val_up]

            # B. Ratio
            for sys_name, (br_up, br_dn, br_nom) in RATIO_SYSTEMATICS.items():
                val_up = nom_val
                val_dn = nom_val
                if br_nom in masked_events.fields:
                    nom_comp = masked_events[br_nom]
                    safe_nom = ak.where(nom_comp == 0, 1.0, nom_comp)
                    if br_up and br_up in masked_events.fields:
                        w_up = masked_w_nom * (masked_events[br_up] / safe_nom)
                        val_up = calc_syst_yield(w_up)
                    if br_dn and br_dn in masked_events.fields:
                        w_dn = masked_w_nom * (masked_events[br_dn] / safe_nom)
                        val_dn = calc_syst_yield(w_dn)
                res[sys_name] = [val_dn, val_up]

            # C. Renorm
            all_renorm = RENORM_SOURCES + RENORM_DETAILED
            for source in all_renorm:
                name_up = f"renormWeight_{source}_up"
                name_dn = f"renormWeight_{source}_down"
                val_up = nom_val
                val_dn = nom_val
                if name_up in masked_events.fields:
                    val_up = calc_syst_yield(masked_w_nom * masked_events[name_up])
                if name_dn in masked_events.fields:
                    val_dn = calc_syst_yield(masked_w_nom * masked_events[name_dn])
                res[f"Theory_{source}"] = [val_dn, val_up]
        return res

    if is_data:
        chunk_res["pass"]["data_obs"] = fill(mask_pass)
        chunk_res["fail"]["data_obs"] = fill(mask_fail)
    else:
        ak8_type_0 = get_leading(events.ak8_type, -1)
        ak8_wbc_0  = get_leading(events.ak8_is_wbc, 0)
        ak8_nc_0   = get_leading(events.ak8_n_c_in_jet, 0)
        is_qcd_val = events.is_qcd

        is_sig = (ak8_type_0 == 1) & (ak8_wbc_0 == 1)           
        is_wqq = (ak8_type_0 == 1) & (is_qcd_val == 0) & (~is_sig)
        is_topbc = (ak8_type_0 == 2) & (ak8_nc_0 == 1) & (is_qcd_val == 0)
        is_other = (~is_sig) & (~is_wqq) & (~is_topbc)

        procs = {"sig": is_sig, "bkg_wqq": is_wqq, "bkg_topbc": is_topbc, "bkg_other": is_other}
        for p_name, p_mask in procs.items():
            if ak.sum(p_mask) > 0:
                chunk_res["pass"][p_name] = fill(mask_pass & p_mask)
                chunk_res["fail"][p_name] = fill(mask_fail & p_mask)
    
    return chunk_res

def analyze_file_chunked(filepath, is_data=False):
    filename = os.path.basename(filepath)
    # ... (patterns definition same as before) ...
    patterns = [
        "ak8_pt", "ak8_gpt_*", "score_cata_*", 
        "ak8_type", "ak8_is_wbc", "ak8_n_c_in_jet", "is_qcd"
    ]
    if not is_data:
        patterns += ["genWeight", "lumiwgt", "puWeight", "trigEffWeight", 
                     "elEffWeight", "muEffWeight", "xsecWeight", "l1PreFiringWeight"]
        patterns += ["flavTagWeight*", "renormWeight*", "puWeight*", "l1PreFiringWeight*", 
                     "trigEffWeight*", "elEffWeight_*", "muEffWeight_*", "bFragWeight*"]

    file_res = {"pass": {}, "fail": {}}

    try:
        with uproot.open(filepath) as f:
            tree = f["Events"]
            num_entries = tree.num_entries
            if num_entries == 0: return None
            
            all_keys = tree.keys()
            branches_to_load = []
            for p in patterns:
                matched = fnmatch.filter(all_keys, p)
                branches_to_load.extend(matched)
            branches_to_load = list(set(branches_to_load))
            
            iterator = tree.iterate(branches_to_load, step_size=args.chunksize, library="ak")
            
            # for small files, NOT show progress bar (to avoid clutter)
            disable_tqdm = num_entries < args.chunksize
            
            for chunk in tqdm(iterator, total=num_entries//args.chunksize + 1, unit="chunk", disable=disable_tqdm, desc=filename[:20]):
                chunk_out = process_chunk(chunk, is_data)
                for reg in ["pass", "fail"]:
                    for proc, val in chunk_out[reg].items():
                        if proc not in file_res[reg]:
                            file_res[reg][proc] = val
                        else:
                            file_res[reg][proc]["nominal"] += val["nominal"]
                            if "data" not in proc:
                                for src in val:
                                    if src == "nominal": continue
                                    if src not in file_res[reg][proc]:
                                        file_res[reg][proc][src] = [0.0, 0.0]
                                    file_res[reg][proc][src][0] += val[src][0]
                                    file_res[reg][proc][src][1] += val[src][1]
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None
    return file_res

final_yields = {"pass": {}, "fail": {}}

def merge_results(source_res):
    if source_res is None: return
    for reg in ["pass", "fail"]:
        for proc, val in source_res[reg].items():
            if proc not in final_yields[reg]:
                final_yields[reg][proc] = val
            else:
                final_yields[reg][proc]["nominal"] += val["nominal"]
                if "data" not in proc:
                    for src in val:
                        if src == "nominal": continue
                        if src not in final_yields[reg][proc]:
                            final_yields[reg][proc][src] = [0.0, 0.0]
                        final_yields[reg][proc][src][0] += val[src][0]
                        final_yields[reg][proc][src][1] += val[src][1]

def main():
    # 1. MC
    mc_files = glob.glob(MC_PATH)
    if not mc_files:
        print(f"WARNING: No MC files found in {MC_PATH}")
    else:
        print(f"Found {len(mc_files)} MC files.")
        for f in mc_files:
            merge_results(analyze_file_chunked(f, is_data=False))

    # 2. Data
    data_files = glob.glob(DATA_PATH)
    if not data_files:
        print(f"WARNING: No Data files found in {DATA_PATH}")
    else:
        print(f"Found {len(data_files)} Data files.")
        for f in data_files:
            merge_results(analyze_file_chunked(f, is_data=True))

    # 3. Kappa Calculation
    print("\nCalculating systematics...")
    for region in final_yields:
        for proc in final_yields[region]:
            if "data" in proc: continue
            nom = final_yields[region][proc]["nominal"]
            if nom == 0: 
                for src in list(final_yields[region][proc].keys()):
                    if src != "nominal": final_yields[region][proc][src] = [1.0, 1.0]
                continue
            for src in list(final_yields[region][proc].keys()):
                if src == "nominal": continue
                abs_dn, abs_up = final_yields[region][proc][src]
                k_dn = abs_dn / nom
                k_up = abs_up / nom
                if k_dn < 0: k_dn = 0.001
                if k_up < 0: k_up = 0.001
                final_yields[region][proc][src] = [k_dn, k_up]

    out_name = f"yields_dbc{args.dbc}_cv{args.cv}_fullsyst.json"
    with open(out_name, "w") as f:
        json.dump(final_yields, f, indent=4)
    print(f"Done! Saved to {out_name}")

if __name__ == "__main__":
    main()