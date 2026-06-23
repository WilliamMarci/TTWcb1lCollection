#!/usr/bin/env python3
import uproot
import awkward as ak
import numpy as np
import ROOT
import argparse
import glob
import fnmatch
from tqdm import tqdm
from collections import defaultdict

ROOT.gROOT.SetBatch(True)

# =========================================================
# Configuration
# =========================================================
BINNING = (50, 0.0, 1.0)  # (Nbins, xmin, xmax)

# Flavor-tag calibration systematics:
# these replace nominal flavTagWeight directly
FLAV_TAG_SOURCES = [
    "JER", "JES", "PUWeight",
    "LHEScaleWeight_muF_ttbar", "LHEScaleWeight_muF_wjets", "LHEScaleWeight_muF_zjets",
    "LHEScaleWeight_muR_ttbar", "LHEScaleWeight_muR_wjets", "LHEScaleWeight_muR_zjets",
    "PSWeightISR_ttbar", "PSWeightISR_wjets", "PSWeightISR_zjets",
    "PSWeightFSR_ttbar", "PSWeightFSR_wjets", "PSWeightFSR_zjets",
    "XSec_WJets_c", "XSec_WJets_b", "XSec_ZJets_c", "XSec_ZJets_b",
    "Stat",
    "Stat_flavB_C0", "Stat_flavB_C1", "Stat_flavB_C2", "Stat_flavB_C3", "Stat_flavB_C4",
    "Stat_flavB_B0", "Stat_flavB_B1", "Stat_flavB_B2", "Stat_flavB_B3", "Stat_flavB_B4",
    "Stat_flavC_C0", "Stat_flavC_C1", "Stat_flavC_C2", "Stat_flavC_C3", "Stat_flavC_C4",
    "Stat_flavC_B0", "Stat_flavC_B1", "Stat_flavC_B2", "Stat_flavC_B3", "Stat_flavC_B4",
    "Stat_flavL_C0", "Stat_flavL_C1", "Stat_flavL_C2", "Stat_flavL_C3", "Stat_flavL_C4",
    "Stat_flavL_B0", "Stat_flavL_B1", "Stat_flavL_B2", "Stat_flavL_B3", "Stat_flavL_B4",
]

# Standard event-level SF systematics: use ratio variation / nominal
RATIO_SYSTEMATICS = {
    "PU":           ("puWeightUp",          "puWeightDown",          "puWeight"),
    "L1PreFiring":  ("l1PreFiringWeightUp", "l1PreFiringWeightDown", "l1PreFiringWeight"),
    "TrigEff":      ("trigEffWeightUp",     "trigEffWeightDown",     "trigEffWeight"),
    "ElEff":        ("elEffWeight_UP",      "elEffWeight_DOWN",      "elEffWeight"),
    "MuEff":        ("muEffWeight_UP",      "muEffWeight_DOWN",      "muEffWeight"),
}

# Write ALL theory sources to shapes by default.
RENORM_SOURCES = [
    "muR", "muF",
    "isr", "fsr",
    "pdfSum", "alphas", "pdfSumWAlphaS",
    "topPt", "hdamp",
    "fsr_G2GG_muR", "fsr_G2QQ_muR", "fsr_Q2QG_muR", "fsr_X2XG_muR",
    "fsr_G2GG_cNS", "fsr_G2QQ_cNS", "fsr_Q2QG_cNS", "fsr_X2XG_cNS",
    "isr_G2GG_muR", "isr_G2QQ_muR", "isr_Q2QG_muR", "isr_X2XG_muR",
    "isr_G2GG_cNS", "isr_G2QQ_cNS", "isr_Q2QG_cNS", "isr_X2XG_cNS",
]

OPTIONAL_MODELING = {
    "bFrag": {
        "nom": "bFragWeightNom",
        "up":  "bFragWeightUp",
        "down": "bFragWeightDown",
    },
    "herwig": {
        "alt": "renormWeight_herwig",
    },
    "fxfx": {
        "alt": "renormWeight_fxfx",
    },
}


# =========================================================
# Utilities
# =========================================================
def get_leading(array, default=0):
    return ak.fill_none(ak.firsts(array), default)


def clip_to_binning(array, xmin, xmax, eps=1e-6):
    array = ak.where(array < xmin, xmin + eps, array)
    array = ak.where(array >= xmax, xmax - eps, array)
    return array


def fill_root_hist(hist, array, weights):
    if len(array) == 0:
        return
    arr_np = ak.to_numpy(array).astype(np.float64)
    w_np = ak.to_numpy(weights).astype(np.float64)

    valid = np.isfinite(arr_np) & np.isfinite(w_np)
    arr_np = arr_np[valid]
    w_np = w_np[valid]

    if len(arr_np) == 0:
        return

    hist.FillN(len(arr_np), arr_np, w_np)


def make_empty_hist(name):
    h = ROOT.TH1F(name, name, BINNING[0], BINNING[1], BINNING[2])
    h.Sumw2()
    h.SetDirectory(0)
    return h


def clone_reset(hist, new_name):
    h = hist.Clone(new_name)
    h.SetDirectory(0)
    h.Reset()
    h.Sumw2()
    return h


def clone_for_write(hist, out_name):
    h = hist.Clone(out_name)
    h.SetDirectory(0)
    return h


def sanitize_hist(hist):
    for i in range(1, hist.GetNbinsX() + 1):
        c = hist.GetBinContent(i)
        e = hist.GetBinError(i)
        if not np.isfinite(c) or c < 0:
            hist.SetBinContent(i, 0.0)
        if not np.isfinite(e) or hist.GetBinContent(i) == 0.0:
            hist.SetBinError(i, 0.0)


def integral(hist):
    return hist.Integral(1, hist.GetNbinsX())


def safe_ratio(num, den, default=1.0):
    return ak.where(den != 0, num / den, default)


def ensure_array_like(x, ref):
    if isinstance(x, (int, float)):
        return ak.ones_like(ref) * x
    return x


def handle_missing(missing_systs, strict, proc, br_name, suffix):
    key = f"{proc}:{suffix}:{br_name}"
    missing_systs[key] += 1
    if strict:
        raise RuntimeError(f"Missing branch {br_name} for process={proc}, syst={suffix}")


def should_apply_tt_modeling(proc):
    return proc in ("sig", "bkg_wqq", "bkg_topbc")


def should_apply_wjets_like(proc):
    return proc == "bkg_other"


def allow_flavtag_source_for_process(src, proc):
    # flavor-tag propagated theory/xsec sources are not universally meaningful
    if ("_ttbar" in src):
        return should_apply_tt_modeling(proc)
    if ("_wjets" in src) or ("WJets" in src):
        return should_apply_wjets_like(proc)
    if ("_zjets" in src) or ("ZJets" in src):
        return should_apply_wjets_like(proc)
    return True


def allow_theory_source_for_process(src, proc):
    # By default write most theory shapes for all MC processes.
    # Restrict top-specific variations to tt-like processes.
    top_specific = {"topPt", "hdamp", "herwig", "fxfx", "bFrag"}
    if src in top_specific:
        return should_apply_tt_modeling(proc)
    if src.startswith("isr_") or src.startswith("fsr_"):
        return should_apply_tt_modeling(proc)
    return True


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dbc", type=float, default=0.7)
    parser.add_argument("--cv", type=float, default=0.7,
                        help="DNN threshold used to split calib/sr regions")
    parser.add_argument("--lumi", type=float, default=41.5)
    parser.add_argument("--chunksize", type=int, default=100000)

    parser.add_argument("--mc_path", type=str,
                        default="/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/MC/scored_samples_2final/*.root")
    parser.add_argument("--data_path", type=str,
                        default="/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/data/scored_data/merged/*.root")

    parser.add_argument("--blind", action="store_true",
                        help="Do not use real data in fit regions")
    parser.add_argument("--asimov", choices=["none", "bkg", "sb"], default="none",
                        help="Build data_obs from nominal templates")
    parser.add_argument("--asimov_mu", type=float, default=1.0,
                        help="Signal strength for sb Asimov")

    parser.add_argument("--strict_missing_syst", action="store_true",
                        help="Raise error if a systematic branch is missing")

    parser.add_argument("--apply_toppt_nominal", action="store_true", default=True,
                        help="Multiply nominal event weight by topptWeight")

    parser.add_argument("--symmetrize_bfrag", action="store_true",
                        help="If only bFrag up exists, build down by symmetrization around nominal")

    args = parser.parse_args()

    out_filename = f"shapes_vcb_dbc{args.dbc}_cv{args.cv}.root"

    print(f"Creating shapes: {out_filename}")
    print(f"Base region                : dbc_score > {args.dbc}")
    print(f"Calibration region (calib) : dbc_score > {args.dbc} AND class_score <= {args.cv}")
    print(f"Signal region (sr)         : dbc_score > {args.dbc} AND class_score > {args.cv}")
    print(f"Observable in both regions : class_score")
    print("")

    print("Systematic configuration:")
    print(f"  FlavorTag replacement sources : {len(FLAV_TAG_SOURCES)}")
    print(f"  Ratio systematics             : {list(RATIO_SYSTEMATICS.keys())}")
    print(f"  Theory sources                : {RENORM_SOURCES}")
    print(f"  Optional modeling             : {list(OPTIONAL_MODELING.keys())}")
    print(f"  topPt in nominal              : {args.apply_toppt_nominal}")
    print("")

    f_out = ROOT.TFile(out_filename, "RECREATE")

    regions = ["calib", "sr"]
    mc_processes = ["sig", "bkg_wqq", "bkg_topbc", "bkg_other"]
    processes = mc_processes + ["data_obs"]

    # -----------------------------------------------------
    # define suffixes
    # -----------------------------------------------------
    syst_suffixes = [""]

    for src in FLAV_TAG_SOURCES:
        syst_suffixes.append(f"_FT_{src}Up")
        syst_suffixes.append(f"_FT_{src}Down")

    for src in RATIO_SYSTEMATICS.keys():
        syst_suffixes.append(f"_{src}Up")
        syst_suffixes.append(f"_{src}Down")

    for src in RENORM_SOURCES:
        syst_suffixes.append(f"_Theory_{src}Up")
        syst_suffixes.append(f"_Theory_{src}Down")

    syst_suffixes.append("_Model_bFragUp")
    syst_suffixes.append("_Model_bFragDown")
    syst_suffixes.append("_Model_herwigUp")
    syst_suffixes.append("_Model_fxfxUp")

    # -----------------------------------------------------
    # initialize hists
    # -----------------------------------------------------
    hists = {}
    for reg in regions:
        hists[reg] = {}
        for proc in processes:
            hists[reg][proc] = {}
            suffixes_to_create = [""] if proc == "data_obs" else syst_suffixes
            for suffix in suffixes_to_create:
                mem_name = f"{reg}__{proc}{suffix}" if suffix else f"{reg}__{proc}"
                hists[reg][proc][suffix] = make_empty_hist(mem_name)

    missing_systs = defaultdict(int)

    # -----------------------------------------------------
    # branches
    # -----------------------------------------------------
    base_branches = [
        "ak8_pt",
        "ak8_gpt_*",
        "score_cata_*",
        "ak8_type", "ak8_is_wbc", "ak8_n_c_in_jet",
        "is_qcd",
    ]

    mc_branches = base_branches + [
        "genWeight",
        "lumiwgt",
        "xsecWeight",

        "puWeight", "puWeightUp", "puWeightDown",
        "trigEffWeight", "trigEffWeightUp", "trigEffWeightDown",
        "elEffWeight", "elEffWeight_UP", "elEffWeight_DOWN",
        "muEffWeight", "muEffWeight_UP", "muEffWeight_DOWN",
        "l1PreFiringWeight", "l1PreFiringWeightUp", "l1PreFiringWeightDown",

        "flavTagWeight",
        "flavTagWeight*",

        "topptWeight",

        "renormWeight*",

        "bFragWeightNom",
        "bFragWeightUp",
        "bFragWeightDown",

        "pdfSumWgt",
        "pdfSumWgtWAlphaS",
        "jetVetoMapEventVeto",
    ]

    # -----------------------------------------------------
    # processor
    # -----------------------------------------------------
    def process_files(file_pattern, is_data):
        files = glob.glob(file_pattern)
        print(f"Processing {len(files)} files for {'Data' if is_data else 'MC'}...")

        branches = base_branches if is_data else mc_branches

        for filepath in tqdm(files):
            try:
                with uproot.open(filepath) as f:
                    tree = f["Events"]
                    if tree.num_entries == 0:
                        continue

                    keys = tree.keys()
                    load_branches = []
                    for p in branches:
                        load_branches.extend(fnmatch.filter(keys, p))
                    load_branches = sorted(list(set(load_branches)))

                    for events in tree.iterate(load_branches, step_size=args.chunksize, library="ak"):
                        g_bc = get_leading(events.ak8_gpt_bc)
                        denom_d = (
                            g_bc + get_leading(events.ak8_gpt_qcd) + get_leading(events.ak8_gpt_cc) +
                            get_leading(events.ak8_gpt_bb) + get_leading(events.ak8_gpt_bs) +
                            get_leading(events.ak8_gpt_cs) + get_leading(events.ak8_gpt_qq) +
                            get_leading(events.ak8_gpt_topbw) + 1e-10
                        )
                        dbc_score = g_bc / denom_d

                        s_w_qq = events.score_cata_w_qq
                        denom_c = (
                            s_w_qq + events.score_cata_qcd + events.score_cata_top_bqq +
                            events.score_cata_top_bc + events.score_cata_top_bq +
                            events.score_cata_non + 1e-10
                        )
                        class_score = s_w_qq / denom_c

                        has_fatjet = ak.num(events.ak8_pt) > 0
                        mask_base  = has_fatjet & (dbc_score > args.dbc)
                        mask_sr    = mask_base & (class_score > args.cv)
                        mask_calib = mask_base & (class_score <= args.cv)

                        region_masks = {
                            "calib": mask_calib,
                            "sr": mask_sr,
                        }

                        if ak.sum(mask_calib) == 0 and ak.sum(mask_sr) == 0:
                            continue

                        for reg, reg_mask in region_masks.items():
                            if ak.sum(reg_mask) == 0:
                                continue

                            events_reg = events[reg_mask]
                            score_reg = class_score[reg_mask]
                            score_reg = clip_to_binning(score_reg, BINNING[1], BINNING[2])

                            if not is_data:
                                genWeight = ensure_array_like(events_reg.genWeight, score_reg)
                                lumiwgt = ensure_array_like(events_reg.lumiwgt, score_reg)
                                xsecWeight = ensure_array_like(events_reg.xsecWeight, score_reg)

                                puWeight = ensure_array_like(events_reg.puWeight, score_reg)
                                trigEffWeight = ensure_array_like(events_reg.trigEffWeight, score_reg)
                                elEffWeight = ensure_array_like(events_reg.elEffWeight, score_reg)
                                muEffWeight = ensure_array_like(events_reg.muEffWeight, score_reg)

                                if "l1PreFiringWeight" in events_reg.fields:
                                    l1pref = ensure_array_like(events_reg.l1PreFiringWeight, score_reg)
                                else:
                                    l1pref = ak.ones_like(score_reg)

                                base_w = (
                                    genWeight *
                                    lumiwgt *
                                    xsecWeight *
                                    puWeight *
                                    trigEffWeight *
                                    elEffWeight *
                                    muEffWeight *
                                    l1pref
                                )

                                if args.apply_toppt_nominal:
                                    if "topptWeight" in events_reg.fields:
                                        topptWeight = ensure_array_like(events_reg.topptWeight, score_reg)
                                        base_w = base_w * topptWeight
                                    else:
                                        handle_missing(
                                            missing_systs, args.strict_missing_syst,
                                            "ALL_MC", "topptWeight", f"{reg}:nominal_toppt"
                                        )

                                flavTagWeight = ensure_array_like(events_reg.flavTagWeight, score_reg)
                                w_nom = base_w * flavTagWeight * args.lumi
                            else:
                                base_w = None
                                w_nom = ak.ones_like(score_reg)

                            if is_data:
                                masks = {"data_obs": ak.ones_like(score_reg, dtype=bool)}
                            else:
                                ak8_type_0 = get_leading(events_reg.ak8_type, -1)
                                ak8_wbc_0  = get_leading(events_reg.ak8_is_wbc, 0)
                                ak8_nc_0   = get_leading(events_reg.ak8_n_c_in_jet, 0)
                                is_qcd_val = events_reg.is_qcd

                                is_sig    = (ak8_type_0 == 1) & (ak8_wbc_0 == 1)
                                is_wqq    = (ak8_type_0 == 1) & (is_qcd_val == 0) & (~is_sig)
                                is_topbc  = (ak8_type_0 == 2) & (ak8_nc_0 == 1) & (is_qcd_val == 0)
                                is_other  = (~is_sig) & (~is_wqq) & (~is_topbc)

                                masks = {
                                    "sig": is_sig,
                                    "bkg_wqq": is_wqq,
                                    "bkg_topbc": is_topbc,
                                    "bkg_other": is_other,
                                }

                            for proc, p_mask in masks.items():
                                if ak.sum(p_mask) == 0:
                                    continue

                                p_score = score_reg[p_mask]
                                p_w_nom = w_nom[p_mask]

                                fill_root_hist(hists[reg][proc][""], p_score, p_w_nom)

                                if is_data:
                                    continue

                                p_events = events_reg[p_mask]
                                p_base_w = base_w[p_mask] * args.lumi

                                # ---------------------------------
                                # Type A: Flavor-tag replacement systematics
                                # ---------------------------------
                                for src in FLAV_TAG_SOURCES:
                                    if not allow_flavtag_source_for_process(src, proc):
                                        continue

                                    for direction_in, direction_out in [("UP", "Up"), ("DOWN", "Down")]:
                                        br_name = f"flavTagWeight_{src}_{direction_in}"
                                        suffix = f"_FT_{src}{direction_out}"

                                        if br_name in p_events.fields:
                                            w_syst = p_base_w * ensure_array_like(p_events[br_name], p_score)
                                            fill_root_hist(hists[reg][proc][suffix], p_score, w_syst)
                                        else:
                                            handle_missing(missing_systs, args.strict_missing_syst,
                                                           proc, br_name, f"{reg}:{suffix}")

                                # ---------------------------------
                                # Type B: Standard ratio systematics
                                # ---------------------------------
                                for sys_name, (br_up, br_dn, br_nom) in RATIO_SYSTEMATICS.items():
                                    if br_nom not in p_events.fields:
                                        handle_missing(missing_systs, args.strict_missing_syst,
                                                       proc, br_nom, f"{reg}:_{sys_name}")
                                        continue

                                    nom_val = ensure_array_like(p_events[br_nom], p_score)

                                    if br_up in p_events.fields:
                                        up_val = ensure_array_like(p_events[br_up], p_score)
                                        ratio_up = safe_ratio(up_val, nom_val, default=1.0)
                                        w_up = p_w_nom * ratio_up
                                        fill_root_hist(hists[reg][proc][f"_{sys_name}Up"], p_score, w_up)
                                    else:
                                        handle_missing(missing_systs, args.strict_missing_syst,
                                                       proc, br_up, f"{reg}:_{sys_name}Up")

                                    if br_dn in p_events.fields:
                                        dn_val = ensure_array_like(p_events[br_dn], p_score)
                                        ratio_dn = safe_ratio(dn_val, nom_val, default=1.0)
                                        w_dn = p_w_nom * ratio_dn
                                        fill_root_hist(hists[reg][proc][f"_{sys_name}Down"], p_score, w_dn)
                                    else:
                                        handle_missing(missing_systs, args.strict_missing_syst,
                                                       proc, br_dn, f"{reg}:_{sys_name}Down")

                                # ---------------------------------
                                # Type C: Theory/generator relative-weight systematics
                                # convention: w_syst = w_nom * renormWeight_xxx
                                # ---------------------------------
                                for src in RENORM_SOURCES:
                                    if not allow_theory_source_for_process(src, proc):
                                        continue

                                    for direction_in, direction_out in [("up", "Up"), ("down", "Down")]:
                                        br_name = f"renormWeight_{src}_{direction_in}"
                                        suffix = f"_Theory_{src}{direction_out}"

                                        if br_name in p_events.fields:
                                            rel_w = ensure_array_like(p_events[br_name], p_score)
                                            w_syst = p_w_nom * rel_w
                                            fill_root_hist(hists[reg][proc][suffix], p_score, w_syst)
                                        else:
                                            handle_missing(missing_systs, args.strict_missing_syst,
                                                           proc, br_name, f"{reg}:{suffix}")

                                # ---------------------------------
                                # Type D1: b fragmentation
                                # ---------------------------------
                                nom_br = OPTIONAL_MODELING["bFrag"]["nom"]
                                up_br  = OPTIONAL_MODELING["bFrag"]["up"]
                                dn_br  = OPTIONAL_MODELING["bFrag"]["down"]

                                if allow_theory_source_for_process("bFrag", proc):
                                    if nom_br in p_events.fields and up_br in p_events.fields:
                                        nom_bfrag = ensure_array_like(p_events[nom_br], p_score)
                                        up_bfrag  = ensure_array_like(p_events[up_br], p_score)

                                        ratio_up = safe_ratio(up_bfrag, nom_bfrag, default=1.0)
                                        w_up = p_w_nom * ratio_up
                                        fill_root_hist(hists[reg][proc]["_Model_bFragUp"], p_score, w_up)

                                        if dn_br is not None and dn_br in p_events.fields:
                                            dn_bfrag = ensure_array_like(p_events[dn_br], p_score)
                                            ratio_dn = safe_ratio(dn_bfrag, nom_bfrag, default=1.0)
                                            w_dn = p_w_nom * ratio_dn
                                            fill_root_hist(hists[reg][proc]["_Model_bFragDown"], p_score, w_dn)
                                        elif args.symmetrize_bfrag:
                                            ratio_dn = safe_ratio(1.0, ratio_up, default=1.0)
                                            w_dn = p_w_nom * ratio_dn
                                            fill_root_hist(hists[reg][proc]["_Model_bFragDown"], p_score, w_dn)
                                        else:
                                            handle_missing(missing_systs, args.strict_missing_syst,
                                                           proc, "bFragWeightDown", f"{reg}:_Model_bFragDown")
                                    else:
                                        if nom_br not in p_events.fields:
                                            handle_missing(missing_systs, args.strict_missing_syst,
                                                           proc, nom_br, f"{reg}:_Model_bFrag")
                                        if up_br not in p_events.fields:
                                            handle_missing(missing_systs, args.strict_missing_syst,
                                                           proc, up_br, f"{reg}:_Model_bFragUp")

                                # ---------------------------------
                                # Type D2: alternative modeling (one-sided)
                                # ---------------------------------
                                if allow_theory_source_for_process("herwig", proc):
                                    br_alt = OPTIONAL_MODELING["herwig"]["alt"]
                                    if br_alt in p_events.fields:
                                        alt_w = ensure_array_like(p_events[br_alt], p_score)
                                        w_alt = p_w_nom * alt_w
                                        fill_root_hist(hists[reg][proc]["_Model_herwigUp"], p_score, w_alt)
                                    else:
                                        handle_missing(missing_systs, args.strict_missing_syst,
                                                       proc, br_alt, f"{reg}:_Model_herwigUp")

                                if allow_theory_source_for_process("fxfx", proc):
                                    br_alt = OPTIONAL_MODELING["fxfx"]["alt"]
                                    if br_alt in p_events.fields:
                                        alt_w = ensure_array_like(p_events[br_alt], p_score)
                                        w_alt = p_w_nom * alt_w
                                        fill_root_hist(hists[reg][proc]["_Model_fxfxUp"], p_score, w_alt)
                                    else:
                                        handle_missing(missing_systs, args.strict_missing_syst,
                                                       proc, br_alt, f"{reg}:_Model_fxfxUp")

            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    # -----------------------------------------------------
    # Process MC first
    # -----------------------------------------------------
    process_files(args.mc_path, is_data=False)

    # -----------------------------------------------------
    # Build data_obs
    # -----------------------------------------------------
    use_real_data = (not args.blind) and (args.asimov == "none")

    if use_real_data:
        process_files(args.data_path, is_data=True)
    else:
        print("Not using real data in calib/sr for data_obs (blind and/or Asimov mode enabled).")
        for reg in regions:
            h_data = clone_reset(hists[reg]["sig"][""], f"{reg}__data_obs")

            h_data.Add(hists[reg]["bkg_wqq"][""])
            h_data.Add(hists[reg]["bkg_topbc"][""])
            h_data.Add(hists[reg]["bkg_other"][""])

            if args.asimov == "sb":
                h_sig_tmp = hists[reg]["sig"][""].Clone(f"{reg}__sig_tmp")
                h_sig_tmp.SetDirectory(0)
                h_sig_tmp.Scale(args.asimov_mu)
                h_data.Add(h_sig_tmp)

            hists[reg]["data_obs"][""] = h_data

    # -----------------------------------------------------
    # Write output
    # -----------------------------------------------------
    f_out.cd()
    for reg in regions:
        if not f_out.GetDirectory(reg):
            f_out.mkdir(reg)
        d = f_out.GetDirectory(reg)
        d.cd()

        for proc in processes:
            for suffix, h in hists[reg][proc].items():
                sanitize_hist(h)
                out_name = f"{proc}{suffix}"
                h_write = clone_for_write(h, out_name)
                sanitize_hist(h_write)
                h_write.Write(out_name)

        f_out.cd()

    f_out.Close()

    print("\n=== Yield summary (nominal) ===")
    print(f"Split threshold at class_score = {args.cv}")
    for reg in regions:
        print(f"[Region: {reg}]")
        for proc in processes:
            h = hists[reg][proc][""]
            y_all = integral(h)
            print(f"  {proc:12s} : total = {y_all:.6f}")

    if len(missing_systs) > 0:
        print("\n=== Missing systematic branches detected ===")
        for k, v in sorted(missing_systs.items()):
            print(f"  {k}  (seen {v} times)")
    else:
        print("\nNo missing systematic branches detected.")

    print("\nDone.")


if __name__ == "__main__":
    main()
