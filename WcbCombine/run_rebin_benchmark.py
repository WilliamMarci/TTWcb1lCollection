#!/usr/bin/env python3
import os
import re
import json
import glob
import shutil
import argparse
import subprocess


DEFAULT_INCLUDE_SYST = [
    "FT_Stat_flav*",
    "PU",
    "ElEff",
    "MuEff",
    "L1PreFiring",
    "TrigEff",
    "Theory_muR",
    "Theory_muF",
    "Theory_isr",
    "Theory_fsr",
    "Theory_topPt",
    "Theory_hdamp",
    "Theory_pdfSum",
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_cmd(cmd, log_path, cwd=None, env=None):
    ensure_dir(os.path.dirname(log_path))
    print(f"[RUN] {' '.join(cmd)}")
    # if cwd:
        # print(f"      cwd -> {cwd}")
    # print(f"      log -> {log_path}")

    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")

    return proc.returncode



def find_plot1dscan():
    path = shutil.which("plot1DScan.py")
    return path


def extract_significance_from_log(log_path):
    significance = None
    pat = re.compile(r"Significance:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
    with open(log_path, "r") as f:
        for line in f:
            m = pat.search(line)
            if m:
                significance = float(m.group(1))
    return significance


def extract_bestfit_from_multidimfit_root_name(combine_dir, name_tag):
    pattern = os.path.join(combine_dir, f"higgsCombine{name_tag}.MultiDimFit.mH*.root")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def extract_mu_lambda_from_log(log_path):
    result = {
        "mu_bestfit": None,
        "mu_err_lo": None,
        "mu_err_hi": None,
        "lambda_cal_bestfit": None,
        "lambda_cal_err_lo": None,
        "lambda_cal_err_hi": None,
    }

    # Combine singles output often has lines like:
    # mu : 1.00000  -0.12345/+0.13579
    # lambda_cal : 1.00000  -0.11111/+0.12222
    pat = re.compile(
        r"^\s*(mu|lambda_cal)\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*"
        r"-\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*/\+\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    )

    with open(log_path, "r") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            poi = m.group(1)
            best = float(m.group(2))
            err_lo = float(m.group(3))
            err_hi = float(m.group(4))

            if poi == "mu":
                result["mu_bestfit"] = best
                result["mu_err_lo"] = err_lo
                result["mu_err_hi"] = err_hi
            elif poi == "lambda_cal":
                result["lambda_cal_bestfit"] = best
                result["lambda_cal_err_lo"] = err_lo
                result["lambda_cal_err_hi"] = err_hi

    return result


def make_plot1d(plot1dscan, scan_root, poi, out_stem, log_path, y_cut="5.5"):
    out_dir = os.path.dirname(os.path.abspath(out_stem))
    out_name = os.path.basename(out_stem)

    ensure_dir(out_dir)

    cmd = [
        plot1dscan,
        os.path.abspath(scan_root),
        "--POI", poi,
        "--y-cut", y_cut,
        "--y-max", y_cut,
        "-o", out_name,
    ]
    run_cmd(cmd, log_path, cwd=out_dir)

def extract_top_impacts(impacts_json_path, top_n=10):
    if not os.path.exists(impacts_json_path):
        return []

    with open(impacts_json_path, "r") as f:
        data = json.load(f)

    params = data.get("params", [])
    items = []

    for p in params:
        name = p.get("name")
        impact_hi = p.get("impact_mu_hi")
        impact_lo = p.get("impact_mu_lo")

        # fallback: some versions may use generic keys
        if impact_hi is None:
            impact_hi = p.get("impact_hi")
        if impact_lo is None:
            impact_lo = p.get("impact_lo")

        abs_impact = 0.0
        vals = [v for v in [impact_hi, impact_lo] if isinstance(v, (int, float))]
        if vals:
            abs_impact = max(abs(v) for v in vals)

        items.append({
            "name": name,
            "impact_hi": impact_hi,
            "impact_lo": impact_lo,
            "abs_impact": abs_impact,
        })

    items.sort(key=lambda x: x["abs_impact"], reverse=True)
    return items[:top_n]

def run_impacts_mu(combine_dir, logs_dir, plots_dir, ws_mu):
    impacts_json = os.path.join(combine_dir, "impacts_mu.json")

    cmd_init = [
        "combineTool.py", "-M", "Impacts",
        "-d", ws_mu,
        "-m", "125",
        "--doInitialFit",
        "-t", "-1",
        "--expectSignal", "1",
        "--setParameters", "mu=1,lambda_cal=1",
        "--freezeParameters", "lambda_cal",
        "--redefineSignalPOIs", "mu",
        "--robustFit", "1",
    ]
    run_cmd(cmd_init, os.path.join(logs_dir, "impacts_mu_initial.log"), cwd=combine_dir)

    cmd_fits = [
        "combineTool.py", "-M", "Impacts",
        "-d", ws_mu,
        "-m", "125",
        "--doFits",
        "-t", "-1",
        "--expectSignal", "1",
        "--setParameters", "mu=1,lambda_cal=1",
        "--freezeParameters", "lambda_cal",
        "--redefineSignalPOIs", "mu",
        "--robustFit", "1",
    ]
    run_cmd(cmd_fits, os.path.join(logs_dir, "impacts_mu_fits.log"), cwd=combine_dir)

    cmd_json = [
        "combineTool.py", "-M", "Impacts",
        "-d", ws_mu,
        "-m", "125",
        "-o", impacts_json,
        "-t", "-1",
        "--expectSignal", "1",
        "--setParameters", "mu=1,lambda_cal=1",
        "--freezeParameters", "lambda_cal",
        "--redefineSignalPOIs", "mu",
    ]
    run_cmd(cmd_json, os.path.join(logs_dir, "impacts_mu_json.log"), cwd=combine_dir)

    plot_impacts = shutil.which("plotImpacts.py")
    if plot_impacts:
        out_name = "impacts_mu"
        cmd_plot = [
            plot_impacts,
            "-i", impacts_json,
            "-o", out_name,
        ]
        run_cmd(cmd_plot, os.path.join(logs_dir, "impacts_mu_plot.log"), cwd=plots_dir)

        # plotImpacts.py usually writes into cwd with out_name
        for ext in ["pdf", "png"]:
            src = os.path.join(plots_dir, f"{out_name}.{ext}")
            if os.path.exists(src):
                pass

    return impacts_json


def run_one_scheme(
    scheme_dir,
    scheme_name,
    card_script,
    regions,
    include_systs,
    plot1dscan,
    skip_plot=False,
    run_impacts=False,
):
    shapes_root = os.path.join(scheme_dir, f"shapes_{scheme_name}.root")
    if not os.path.exists(shapes_root):
        raise RuntimeError(f"Missing shapes root: {shapes_root}")

    combine_dir = os.path.join(scheme_dir, "combine")
    logs_dir = os.path.join(scheme_dir, "logs")
    plots_dir = os.path.join(scheme_dir, "plots_combine")
    ensure_dir(combine_dir)
    ensure_dir(logs_dir)
    ensure_dir(plots_dir)

    card_path = os.path.join(combine_dir, "card_calib_sr_hybrid.txt")
    ws_mu_lambda = os.path.join(combine_dir, "ws_mu_lambda_hybrid.root")
    ws_lambda = os.path.join(combine_dir, "ws_lambda_cal_hybrid.root")
    ws_mu = os.path.join(combine_dir, "ws_mu_hybrid.root")

    # -------------------------------------------------
    # 1) write card
    # -------------------------------------------------
    cmd_card = [
        "python3", card_script,
        os.path.abspath(shapes_root),
        "-r", *regions,
        "--include-syst", *include_systs,
        "--auto-groups",
        "-o", card_path,
    ]
    run_cmd(cmd_card, os.path.join(logs_dir, "write_shape_card.log"), cwd=combine_dir)

    # -------------------------------------------------
    # 2) text2workspace
    # -------------------------------------------------
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")

    cmd_t2w_2d = [
        "text2workspace.py", card_path,
        "-o", ws_mu_lambda,
        "-P", "VcbModel:vcbModel",
        "--PO", "poi=mu,lambda_cal",
    ]
    run_cmd(cmd_t2w_2d, os.path.join(logs_dir, "text2workspace_mu_lambda.log"), cwd=combine_dir, env=env)

    cmd_t2w_lambda = [
        "text2workspace.py", card_path,
        "-o", ws_lambda,
        "-P", "VcbModel:vcbModel",
        "--PO", "poi=lambda_cal",
    ]
    run_cmd(cmd_t2w_lambda, os.path.join(logs_dir, "text2workspace_lambda.log"), cwd=combine_dir, env=env)

    cmd_t2w_mu = [
        "text2workspace.py", card_path,
        "-o", ws_mu,
        "-P", "VcbModel:vcbModel",
        "--PO", "poi=mu",
    ]
    run_cmd(cmd_t2w_mu, os.path.join(logs_dir, "text2workspace_mu.log"), cwd=combine_dir, env=env)

    # -------------------------------------------------
    # 3) closure 2D
    # -------------------------------------------------
    cmd_fit_2d = [
        "combine", "-M", "MultiDimFit", ws_mu_lambda,
        "-t", "-1",
        "--setParameters", "mu=1,lambda_cal=1",
        "--setParameterRanges", "mu=0,5:lambda_cal=0.5,1.5",
        "--redefineSignalPOIs", "mu,lambda_cal",
        "--algo", "singles",
        "--robustFit", "1",
        "--cminDefaultMinimizerStrategy", "0",
        "-n", "_closure_2d",
    ]
    run_cmd(cmd_fit_2d, os.path.join(logs_dir, "combine_closure_2d.log"), cwd=combine_dir)

    # -------------------------------------------------
    # 4) 1D scan mu
    # -------------------------------------------------
    cmd_scan_mu = [
        "combine", "-M", "MultiDimFit", ws_mu,
        "-t", "-1",
        "--setParameters", "mu=1,lambda_cal=1",
        "--setParameterRanges", "mu=0,5:lambda_cal=0.5,1.5",
        "--redefineSignalPOIs", "mu",
        "--algo", "grid",
        "--points", "80",
        "--robustFit", "1",
        "--cminDefaultMinimizerStrategy", "0",
        "-n", "_scan_mu",
    ]
    run_cmd(cmd_scan_mu, os.path.join(logs_dir, "combine_scan_mu.log"), cwd=combine_dir)

    # -------------------------------------------------
    # 5) 1D scan lambda_cal
    # -------------------------------------------------
    cmd_scan_lambda = [
        "combine", "-M", "MultiDimFit", ws_lambda,
        "-t", "-1",
        "--setParameters", "mu=1,lambda_cal=1",
        "--redefineSignalPOIs", "lambda_cal",
        "--algo", "grid",
        "--points", "80",
        "--robustFit", "1",
        "--cminDefaultMinimizerStrategy", "0",
        "-n", "_scan_lambda_cal",
    ]
    run_cmd(cmd_scan_lambda, os.path.join(logs_dir, "combine_scan_lambda_cal.log"), cwd=combine_dir)

    # -------------------------------------------------
    # 6) significance
    # -------------------------------------------------
    cmd_signif = [
        "combine", "-M", "Significance", ws_mu,
        "-t", "-1",
        "--expectSignal", "1",
        "--setParameters", "mu=1,lambda_cal=1",
        "--setParameterRanges", "mu=0,5:lambda_cal=0.5,1.5",
        "--redefineSignalPOIs", "mu",
        "-n", "_exp_signif",
    ]
    run_cmd(cmd_signif, os.path.join(logs_dir, "combine_significance.log"), cwd=combine_dir)

    # -------------------------------------------------
    # 7) plot 1D scans
    # -------------------------------------------------
    mu_scan_root = extract_bestfit_from_multidimfit_root_name(combine_dir, "_scan_mu")
    lambda_scan_root = extract_bestfit_from_multidimfit_root_name(combine_dir, "_scan_lambda_cal")

    if (not skip_plot) and plot1dscan:
        if mu_scan_root:
            out_stem = os.path.join(plots_dir, "scan_mu")
            make_plot1d(
                plot1dscan,
                mu_scan_root,
                "mu",
                out_stem,
                os.path.join(logs_dir, "plot1d_mu.log"),
            )
        if lambda_scan_root:
            out_stem = os.path.join(plots_dir, "scan_lambda_cal")
            make_plot1d(
                plot1dscan,
                lambda_scan_root,
                "lambda_cal",
                out_stem,
                os.path.join(logs_dir, "plot1d_lambda_cal.log"),
            )

    impacts_info = None
    if run_impacts:
        print(f"\n[INFO] Running impacts for mu...")
        impacts_json = run_impacts_mu(combine_dir, logs_dir, plots_dir, ws_mu)
        impacts_info = {
            "mu_json": os.path.abspath(impacts_json),
            "top_nuisances": extract_top_impacts(impacts_json, top_n=10),
        }

    # -------------------------------------------------
    # 8) collect summary
    # -------------------------------------------------
    summary = {
        "scheme": scheme_name,
        "scheme_dir": os.path.abspath(scheme_dir),
        "shapes_root": os.path.abspath(shapes_root),
        "card": os.path.abspath(card_path),
        "workspaces": {
            "mu_lambda": os.path.abspath(ws_mu_lambda),
            "lambda_cal": os.path.abspath(ws_lambda),
            "mu": os.path.abspath(ws_mu),
        },
        "outputs": {
            "scan_mu_root": os.path.abspath(mu_scan_root) if mu_scan_root else None,
            "scan_lambda_cal_root": os.path.abspath(lambda_scan_root) if lambda_scan_root else None,
        },
        "significance": extract_significance_from_log(os.path.join(logs_dir, "combine_significance.log")),
        "fit": extract_mu_lambda_from_log(os.path.join(logs_dir, "combine_closure_2d.log")),
        "status": "ok",
        "impacts": impacts_info,

    }

    with open(os.path.join(scheme_dir, "combine_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Combine benchmark for all rebin schemes")
    parser.add_argument("--scan-dir", required=True, help="Directory containing scheme subdirectories")
    parser.add_argument("--card-script", default="./write_shape_card.py", help="Path to write_shape_card.py")
    parser.add_argument("--regions", nargs="+", default=["calib", "sr"], help="Regions to pass to write_shape_card.py")
    parser.add_argument("--include-syst", nargs="+", default=DEFAULT_INCLUDE_SYST, help="Systematics to include in card")
    parser.add_argument("--plot1d-script", default=None, help="Path to plot1DScan.py (default: auto find in PATH)")
    parser.add_argument("--schemes", nargs="*", default=None, help="Only run selected schemes")
    parser.add_argument("--skip-plot", action="store_true", help="Skip plot1DScan.py")
    parser.add_argument("--run-impacts", action="store_true", help="Run combine impacts for mu")
    args = parser.parse_args()

    scan_dir = os.path.abspath(args.scan_dir)
    if not os.path.isdir(scan_dir):
        raise RuntimeError(f"scan-dir does not exist: {scan_dir}")

    card_script = os.path.abspath(args.card_script)
    if not os.path.exists(card_script):
        raise RuntimeError(f"card script not found: {card_script}")

    plot1dscan = args.plot1d_script
    if plot1dscan is None and (not args.skip_plot):
        plot1dscan = find_plot1dscan()
        if not plot1dscan:
            print("[WARN] plot1DScan.py not found in PATH, plotting will be skipped")
            args.skip_plot = True

    all_scheme_dirs = []
    for item in sorted(os.listdir(scan_dir)):
        path = os.path.join(scan_dir, item)
        if not os.path.isdir(path):
            continue
        shape_root = os.path.join(path, f"shapes_{item}.root")
        if os.path.exists(shape_root):
            all_scheme_dirs.append((item, path))

    if args.schemes:
        wanted = set(args.schemes)
        all_scheme_dirs = [(name, path) for name, path in all_scheme_dirs if name in wanted]

    if not all_scheme_dirs:
        raise RuntimeError("No valid scheme directories found")

    benchmark = {}

    for scheme_name, scheme_dir in all_scheme_dirs:
        print(f"\n=== Running scheme: {scheme_name} ===")
        try:
            summary = run_one_scheme(
                scheme_dir=scheme_dir,
                scheme_name=scheme_name,
                card_script=card_script,
                regions=args.regions,
                include_systs=args.include_syst,
                plot1dscan=plot1dscan,
                skip_plot=args.skip_plot,
                run_impacts=args.run_impacts,
            )
            benchmark[scheme_name] = summary
            print(f"[OK] {scheme_name}")
        except Exception as e:
            print(f"[FAIL] {scheme_name}: {e}")
            benchmark[scheme_name] = {
                "scheme": scheme_name,
                "scheme_dir": os.path.abspath(scheme_dir),
                "status": "failed",
                "error": str(e),
            }

    out_json = os.path.join(scan_dir, "benchmark_summary.json")
    with open(out_json, "w") as f:
        json.dump(benchmark, f, indent=2)

    print(f"\nSaved benchmark summary to: {out_json}")


if __name__ == "__main__":
    main()
