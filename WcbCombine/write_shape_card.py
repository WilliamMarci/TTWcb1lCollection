#! /usr/bin/env python3
import ROOT
import argparse
import os
import sys
import fnmatch

ROOT.gROOT.SetBatch(True)

DEFAULT_PROCS = ["sig", "bkg_wqq", "bkg_topbc", "bkg_other"]


def list_keys(tdir):
    return [k.GetName() for k in tdir.GetListOfKeys()]


def extract_systematics(obj_names, proc):
    """
    From object names like:
      sig
      sig_PUUp
      sig_PUDown
      sig_FT_JERUp
      sig_FT_JERDown
      sig_Model_herwigUp
    extract syst base names:
      PU, FT_JER, Model_herwig, ...
    """
    systs = set()
    prefix = proc + "_"
    for name in obj_names:
        if not name.startswith(prefix):
            continue
        tail = name[len(prefix):]
        if tail.endswith("Up"):
            systs.add(tail[:-2])
        elif tail.endswith("Down"):
            systs.add(tail[:-4])
    return systs


def hist_exists(obj_names, name):
    return name in obj_names


def format_columns(items, width=12):
    return "  ".join(f"{x:<{width}s}" for x in items)


def parse_kv_list(entries):
    out = {}
    for item in entries:
        if "=" not in item:
            raise ValueError(f"Invalid format '{item}', expected name=value")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise ValueError(f"Invalid format '{item}', empty name or value")
        out[k] = v
    return out


def matches_any_pattern(name, patterns):
    if not patterns:
        return True
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def matches_no_pattern(name, patterns):
    if not patterns:
        return True
    return not any(fnmatch.fnmatch(name, p) for p in patterns)


def classify_syst(syst):
    if syst.startswith("FT_"):
        return "flavtag"
    if syst in ["PU", "L1PreFiring", "TrigEff", "ElEff", "MuEff"]:
        return "exp"
    if syst.startswith("Theory_"):
        return "theory"
    if syst.startswith("Model_"):
        return "model"
    return "other"


def get_hist(tdir, name):
    if not tdir:
        return None
    obj = tdir.Get(name)
    if not obj:
        return None
    if not obj.InheritsFrom("TH1"):
        return None
    return obj


def get_hist_integral(tdir, name):
    h = get_hist(tdir, name)
    if not h:
        return None
    return h.Integral()


def is_finite_positive(x, eps=0.0):
    if x is None:
        return False
    if not ROOT.TMath.Finite(float(x)):
        return False
    return float(x) > eps


def valid_shape_pair(tdir, proc, syst, eps=0.0):
    """
    Accept a shape nuisance for a given (region, process) only if:
      - nominal exists
      - Up exists
      - Down exists
      - all integrals are finite
      - nominal, Up, Down integrals are all > eps
    This protects text2workspace/combine from bogus zero-norm shapes.
    """
    i_nom = get_hist_integral(tdir, proc)
    i_up = get_hist_integral(tdir, f"{proc}_{syst}Up")
    i_dn = get_hist_integral(tdir, f"{proc}_{syst}Down")

    return (
        is_finite_positive(i_nom, eps=eps)
        and is_finite_positive(i_up, eps=eps)
        and is_finite_positive(i_dn, eps=eps)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Write Combine shape datacard from ROOT templates with filtering and Vcb-model helpers"
    )
    parser.add_argument("root_file", help="Input ROOT file with shapes")
    parser.add_argument("-r", "--regions", nargs="+", default=["calib", "sr"],
                        help="Region/directory names inside ROOT file")
    parser.add_argument("-o", "--output", default=None, help="Output datacard name")
    parser.add_argument("--procs", nargs="+", default=DEFAULT_PROCS,
                        help="Ordered list of processes")
    parser.add_argument("--signal-procs", nargs="+", default=["sig"],
                        help="Processes to assign non-positive process IDs")

    # General options
    parser.add_argument("--autoMCStats", default="0", help="Value for '* autoMCStats'")
    parser.add_argument("--observation", default="-1",
                        help="Observation entry. Use -1 to read from data_obs histogram")
    parser.add_argument("--rate", default="-1",
                        help="Rate entry for all processes. Use -1 to read from nominal histograms")

    # Syst filtering
    parser.add_argument("--include-syst", nargs="*", default=[],
                        help="Only include systematics matching these glob patterns, e.g. FT_* Theory_muR Theory_muF")
    parser.add_argument("--exclude-syst", nargs="*", default=[],
                        help="Exclude systematics matching these glob patterns")
    parser.add_argument("--drop-one-sided", action="store_true",
                        help="Drop systematics that do not have at least one process with both Up and Down")
    parser.add_argument("--allow-one-sided", nargs="*", default=["Model_herwig", "Model_fxfx"],
                        help="Patterns allowed to remain ignored without error if they are one-sided only")

    # Optional extra card lines
    parser.add_argument("--flatParam", action="append", default=[],
                        help="Append a flatParam line, e.g. --flatParam lambda_cal")
    parser.add_argument("--param", action="append", default=[],
                        help="Append a param line in form name=value, e.g. --param r=0.15,0.03")
    parser.add_argument("--rateParam", action="append", default=[],
                        help="Append a raw rateParam line body in form name=bin,process,formula,args")
    parser.add_argument("--extArg", action="append", default=[],
                        help="Append an extArg line in form name=value")
    parser.add_argument("--group", action="append", default=[],
                        help="Append a nuisance group line in form groupName=syst1,syst2,syst3")
    parser.add_argument("--add-line", action="append", default=[],
                        help="Append any raw line at end of card")

    # Vcb helper options
    parser.add_argument("--add-vcb-model-lines", action="store_true",
                        help="Add common Vcb model lines: mu flatParam, lambda_cal flatParam, r param ...")
    parser.add_argument("--r-value", default="0.20 0.05",
                        help="RHS of 'r param', default '0.20 0.05'")
    parser.add_argument("--use-rateparam-wqq", action="store_true",
                        help="Write example rateParam lines for bkg_wqq scaling instead of leaving it fully to PhysicsModel")
    parser.add_argument("--wqq-r-value", default="0.20",
                        help="Numeric r value for simple card-side scale_wqq example if --use-rateparam-wqq is used")

    # Auto groups
    parser.add_argument("--auto-groups", action="store_true",
                        help="Automatically add groups: flavtag, exp, theory, model")

    args = parser.parse_args()

    root_file = args.root_file
    regions = args.regions
    procs = args.procs
    signal_procs = set(args.signal_procs)

    if not os.path.exists(root_file):
        print(f"ERROR: file not found: {root_file}")
        sys.exit(1)

    f = ROOT.TFile.Open(root_file)
    if not f or f.IsZombie():
        print(f"ERROR: cannot open ROOT file: {root_file}")
        sys.exit(1)

    # -----------------------------------------------------
    # Validate regions and collect object names
    # -----------------------------------------------------
    region_obj_names = {}
    required = procs + ["data_obs"]

    for region in regions:
        d = f.Get(region)
        if not d:
            print(f"ERROR: directory '{region}' not found in {root_file}")
            sys.exit(1)

        obj_names = list_keys(d)
        region_obj_names[region] = obj_names

        missing = [x for x in required if x not in obj_names]
        if missing:
            print(f"ERROR: missing required histograms in region '{region}':")
            for m in missing:
                print(f"  - {region}/{m}")
            sys.exit(1)

    # -----------------------------------------------------
    # Build process IDs
    # -----------------------------------------------------
    proc_ids = {}
    next_sig = 0
    next_bkg = 1
    for p in procs:
        if p in signal_procs:
            proc_ids[p] = next_sig
            next_sig -= 1
        else:
            proc_ids[p] = next_bkg
            next_bkg += 1

    # -----------------------------------------------------
    # Collect candidate systematics from all regions/processes
    # -----------------------------------------------------
    syst_candidates = set()
    for region in regions:
        obj_names = region_obj_names[region]
        for proc in procs:
            syst_candidates |= extract_systematics(obj_names, proc)

    # Apply include/exclude filters
    syst_filtered = []
    for syst in sorted(syst_candidates):
        if not matches_any_pattern(syst, args.include_syst):
            continue
        if not matches_no_pattern(syst, args.exclude_syst):
            continue
        syst_filtered.append(syst)

    # Keep only nuisances that are usable somewhere
    systs_final = []
    syst_proc_map = {}   # syst -> {(region,proc): "1"/"-"}

    for syst in syst_filtered:
        usable_somewhere = False
        one_sided_allowed = matches_any_pattern(syst, args.allow_one_sided)

        per_cell = {}
        for region in regions:
            tdir = f.Get(region)
            for proc in procs:
                if valid_shape_pair(tdir, proc, syst):
                    per_cell[(region, proc)] = "1"
                    usable_somewhere = True
                else:
                    per_cell[(region, proc)] = "-"

        if usable_somewhere:
            systs_final.append(syst)
            syst_proc_map[syst] = per_cell
        else:
            # If a nuisance is only one-sided (e.g. herwig/fxfx), we do not write a
            # standard Combine "shape" row because that requires usable Up/Down shapes.
            # We silently ignore allowed one-sided nuisances unless the user wants strict dropping.
            if args.drop_one_sided:
                pass
            elif one_sided_allowed:
                pass
            else:
                pass

    # -----------------------------------------------------
    # Default output name
    # -----------------------------------------------------
    if args.output is None:
        base = os.path.splitext(os.path.basename(root_file))[0]
        regtag = "_".join(regions)
        out_file = f"datacard_{regtag}_{base}.txt"
    else:
        out_file = args.output

    # -----------------------------------------------------
    # Build card
    # -----------------------------------------------------
    lines = []
    lines.append("# Shape datacard")
    lines.append(f"# Auto-generated from {root_file}")
    lines.append(f"# Regions: {', '.join(regions)}")
    if args.include_syst:
        lines.append(f"# Included syst patterns: {', '.join(args.include_syst)}")
    if args.exclude_syst:
        lines.append(f"# Excluded syst patterns: {', '.join(args.exclude_syst)}")

    lines.append(f"imax {len(regions)}")
    lines.append(f"jmax {len(procs) - 1}")
    lines.append("kmax *")
    lines.append("-------------------------------------------------------------------------------")

    for region in regions:
        lines.append(f"shapes *        {region}  {root_file} {region}/$PROCESS {region}/$PROCESS_$SYSTEMATIC")
        lines.append(f"shapes data_obs {region}  {root_file} {region}/data_obs")

    lines.append("-------------------------------------------------------------------------------")
    lines.append("bin             " + format_columns(regions, width=12))
    lines.append("observation     " + format_columns([args.observation] * len(regions), width=12))
    lines.append("-------------------------------------------------------------------------------")

    bin_row = []
    proc_row = []
    procid_row = []
    rate_row = []

    for region in regions:
        for p in procs:
            bin_row.append(region)
            proc_row.append(p)
            procid_row.append(str(proc_ids[p]))
            rate_row.append(args.rate)

    lines.append("bin             " + format_columns(bin_row, width=12))
    lines.append("process         " + format_columns(proc_row, width=12))
    lines.append("process         " + format_columns(procid_row, width=12))
    lines.append("rate            " + format_columns(rate_row, width=12))
    lines.append("-------------------------------------------------------------------------------")

    # -----------------------------------------------------
    # Shape nuisances
    # -----------------------------------------------------
    for syst in systs_final:
        vals = []
        for region in regions:
            for proc in procs:
                vals.append(syst_proc_map[syst][(region, proc)])
        lines.append(f"{syst:24s} shape  " + format_columns(vals, width=12))

    # -----------------------------------------------------
    # Extra lines
    # -----------------------------------------------------
    extra_lines = []

    if args.add_vcb_model_lines:
        extra_lines.extend([
            "mu flatParam",
            "lambda_cal flatParam",
            f"r param {args.r_value}",
        ])

        if args.use_rateparam_wqq:
            rnum = args.wqq_r_value
            for reg in regions:
                extra_lines.append(
                    f"scale_wqq_{reg} rateParam {reg} bkg_wqq (@0*( -{rnum}) + 1)/(1-{rnum}) mu"
                )

    for name in args.flatParam:
        extra_lines.append(f"{name} flatParam")

    try:
        param_dict = parse_kv_list(args.param)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    for name, rhs in param_dict.items():
        rhs = rhs.replace(",", " ")
        extra_lines.append(f"{name} param {rhs}")

    try:
        extarg_dict = parse_kv_list(args.extArg)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    for name, rhs in extarg_dict.items():
        extra_lines.append(f"{name} extArg {rhs}")

    try:
        rp_dict = parse_kv_list(args.rateParam)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    for name, rhs in rp_dict.items():
        rhs = rhs.replace(",", " ")
        extra_lines.append(f"{name} rateParam {rhs}")

    extra_lines.extend(args.add_line)

    if extra_lines:
        lines.append("-------------------------------------------------------------------------------")
        lines.extend(extra_lines)

    # -----------------------------------------------------
    # Groups
    # -----------------------------------------------------
    group_lines = []

    if args.auto_groups:
        grouped = {"flavtag": [], "exp": [], "theory": [], "model": [], "other": []}
        for s in systs_final:
            grouped[classify_syst(s)].append(s)

        for gname in ["flavtag", "exp", "theory", "model", "other"]:
            if grouped[gname]:
                group_lines.append(f"{gname} group = " + " ".join(grouped[gname]))

    if args.group:
        try:
            group_dict = parse_kv_list(args.group)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        for gname, members in group_dict.items():
            members = members.replace(",", " ")
            group_lines.append(f"{gname} group = {members}")

    if group_lines:
        lines.append("-------------------------------------------------------------------------------")
        lines.extend(group_lines)

    lines.append("-------------------------------------------------------------------------------")
    lines.append(f"* autoMCStats {args.autoMCStats}")

    with open(out_file, "w") as fout:
        fout.write("\n".join(lines) + "\n")

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------
    print(f"Datacard written to: {out_file}")
    print(f"Regions: {', '.join(regions)}")
    print("Processes and IDs:")
    for p in procs:
        print(f"  - {p:12s} : {proc_ids[p]}")

    print(f"Found {len(systs_final)} shape systematics after filtering")
    if systs_final:
        print("Systematics:")
        for s in systs_final:
            print(f"  - {s}")

    if extra_lines:
        print("Extra card lines:")
        for x in extra_lines:
            print(f"  + {x}")

    if group_lines:
        print("Group lines:")
        for x in group_lines:
            print(f"  + {x}")

    f.Close()


if __name__ == "__main__":
    main()
