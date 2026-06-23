#! /usr/bin/env python3
import os
import subprocess
import itertools
import shutil
import glob
import sys
import getpass
import numpy as np

# Intro
# -----------------------
# this script performs a grid scan over DBC and CV cuts in blind mode. And can get NLL scans and impacts for each point about `lambda_cal` (the calibration factor).

# CONFIGURATION
# -----------------------
# `DBC_CUTS` and `CV_CUTS` define the grid of working points to scan.
# `ORIG_MC_PATTERN` and `ORIG_DATA_PATTERN` specify the original locations of the ROOT files.
# `RANGES` sets the allowed range for `lambda_cal` during fits.
DBC_CUTS = [round(x, 2) for x in np.arange(0.6, 0.9001, 0.05)]
CV_CUTS  = [round(x, 2) for x in np.arange(0.6, 0.9001, 0.05)]
ORIG_MC_PATTERN   = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/MC/scored_samples_2final/*.root"
ORIG_DATA_PATTERN = "/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/data/scored_data/merged/*.root"
RANGES = "lambda_cal=0.01,2.0" 
# [Blind Mode]
PARAMS_VAL = "mask_boosted_pass=1,mu=1,r=0.00084"
FREEZE_OPTS = "--freezeParameters mu"
POI_OPTS = "--redefineSignalPOIs lambda_cal"
IMPACT_EXCLUDE_OPTS = '--exclude "mu,mask_boosted_pass"'
DIAG_BASE_DIR = "scan_lambda_blind"
# Default Setting
USER = getpass.getuser()
CACHE_DIR = f"/tmp/{USER}/vcb_analysis_cache"
CACHE_MC  = os.path.join(CACHE_DIR, "MC")
CACHE_DATA = os.path.join(CACHE_DIR, "Data")

# DEBUG
def print_log_tail(log_file, lines=20):
    if not os.path.exists(log_file):
        print(f"    [Log Not Found] {log_file}")
        return
    
    print(f"\n    {'='*20} Error Log Tail ({os.path.basename(log_file)}) {'='*20}")
    try:
        with open(log_file, 'r') as f:
            content = f.readlines()
            tail = content[-lines:] if len(content) > lines else content
            print("".join(tail))
    except Exception as e:
        print(f"    [Read Failed] {e}")
    print(f"    {'='*60}\n")

def run_command(cmd, log_file=None, live_output=False):
    env = os.environ.copy()
    cwd = os.getcwd()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{cwd}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = cwd

    if live_output:
        ret = subprocess.run(cmd, shell=True, stdout=None, stderr=None, env=env)
    elif log_file:
        with open(log_file, "w") as f:
            ret = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, text=True, env=env)
    else:
        ret = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    
    if ret.returncode != 0:
        print(f"  [Error] Command failed: {cmd}")
        if not live_output and not log_file: 
            print("  >>> Error Output:")
            print(ret.stderr)
            print("  <<< End Error")
        elif log_file:
            print_log_tail(log_file)
        return False
    return True

def prepare_data_cache():
    print(f"\n[Cache] Checking data cache in {CACHE_DIR}...")
    os.makedirs(CACHE_MC, exist_ok=True)
    os.makedirs(CACHE_DATA, exist_ok=True)
    def sync_files(src_pattern, dst_dir, name):
        src_files = glob.glob(src_pattern)
        if not src_files:
            print(f"  [Error] No files found for {name} in {src_pattern}")
            return False
        dst_files = glob.glob(os.path.join(dst_dir, "*.root"))
        if len(dst_files) == len(src_files):
            print(f"  [Skip] {name} files already cached ({len(dst_files)} files).")
            return True
        print(f"  [Copy] Copying {len(src_files)} {name} files to local cache...")
        for f in dst_files: os.remove(f)
        for f in src_files:
            shutil.copy2(f, dst_dir)
            print(f"    Copied {os.path.basename(f):<80}", end="\r")
        print(f"\n  [Done] {name} copy finished.")
        return True
    ok_mc = sync_files(ORIG_MC_PATTERN, CACHE_MC, "MC")
    ok_data = sync_files(ORIG_DATA_PATTERN, CACHE_DATA, "Data")
    return ok_mc and ok_data

def main():
    if not prepare_data_cache(): return

    local_mc_path = os.path.join(CACHE_MC, "*.root")
    local_data_path = os.path.join(CACHE_DATA, "*.root")
    has_combine_tool = shutil.which("combineTool.py") is not None

    os.makedirs(DIAG_BASE_DIR, exist_ok=True)

    for dbc, cv in itertools.product(DBC_CUTS, CV_CUTS):
        tag = f"dbc{dbc}_cv{cv}"
        print(f"\n>>> Processing Point: {tag} (BLIND MODE)")
        
        sub_dir = os.path.join(DIAG_BASE_DIR, tag)
        os.makedirs(sub_dir, exist_ok=True)

        json_file = f"yields_{tag}_fullsyst.json"
        card_file = f"yields_{tag}_fullsyst.txt"
        ws_file   = f"workspace_{tag}.root"
        
        # --- Step 1: Generate Yields ---
        if not os.path.exists(json_file):
            print("  Generating yields...")
            cmd_step1 = (f"{sys.executable} -u make_yields.py "
                         f"--dbc {dbc} --cv {cv} "
                         f"--mc_path '{local_mc_path}' "
                         f"--data_path '{local_data_path}'")
            if not run_command(cmd_step1, live_output=True): continue
        
        # --- Step 2: Write Card ---
        if not run_command(f"{sys.executable} write_card.py {json_file}"): continue
        
        # --- Step 3: Workspace ---
        print("  Creating workspace...")
        cmd_t2w = (f"text2workspace.py {card_file} -o {ws_file} "
                   f"-P VcbModel:vcbModel --channel-masks")
        if not run_command(cmd_t2w): continue
            
        # --- Step 4: FitDiagnostics (Blind) ---
        fit_log = os.path.join(sub_dir, "fit.log")
        cmd_fit = (
            f"combine -M FitDiagnostics {ws_file} -n .{tag} "
            f"--saveShapes --saveWithUncertainties "
            f"--setParameters {PARAMS_VAL} "
            f"{FREEZE_OPTS} " 
            f"--setParameterRanges {RANGES} "
            f"--cminDefaultMinimizerStrategy 0 "
        )
        run_command(cmd_fit, log_file=fit_log)
        if os.path.exists(f"fitDiagnostics.{tag}.root"):
            shutil.move(f"fitDiagnostics.{tag}.root", os.path.join(sub_dir, "fitDiagnostics.root"))
        
        # --- Step 5: NLL Scans (Standard plot1DScan.py) ---
        print("  Running NLL Scans (Lambda only)...")
        
        scan_file = f"higgsCombine.{tag}.lambda.MultiDimFit.mH120.root"
        
        # 5.1 Run MultiDimFit
        cmd_scan_lam = (
            f"combine -M MultiDimFit {ws_file} -n .{tag}.lambda "
            f"--algo grid --points 30 "
            f"--setParameters {PARAMS_VAL} "
            f"{FREEZE_OPTS} "
            f"{POI_OPTS} "
            f"--setParameterRanges {RANGES} "
            f"-P lambda_cal --cminDefaultMinimizerStrategy 0 "
        )
        run_command(cmd_scan_lam, log_file=os.path.join(sub_dir, "scan_lambda_run.log"))

        # 5.2 Run plot1DScan.py
        if os.path.exists(scan_file):
            print("    Plotting NLL scan...")
            out_scan_name = "scan_lambda"
            cmd_plot_scan = (
                f"plot1DScan.py {scan_file} "
                f"-o {out_scan_name} "
                f"--POI lambda_cal "
                f"--main-label 'Blind Scan' "
            )
            run_command(cmd_plot_scan)
            
            for ext in [".png", ".pdf"]:
                if os.path.exists(out_scan_name + ext):
                    shutil.move(out_scan_name + ext, os.path.join(sub_dir, out_scan_name + ext))
            
            os.remove(scan_file)

        # --- Step 6: Impacts (Standard plotImpacts.py) ---
        if has_combine_tool:
            print("  Running Impacts (on lambda_cal)...")
            
            # Impacts Step 1: Initial Fit
            cmd_imp_init = (
                f"combineTool.py -M Impacts -d {ws_file} -m 120 "
                f"--doInitialFit --robustFit 1 "
                f"--setParameters {PARAMS_VAL} "
                f"{FREEZE_OPTS} "
                f"{POI_OPTS} "
                f"{IMPACT_EXCLUDE_OPTS} " # [关键] 排除 mu 和 mask
                f"--setParameterRanges {RANGES} "
            )
            if not run_command(cmd_imp_init, log_file=os.path.join(sub_dir, "impacts_init.log")): continue
            
            # Impacts Step 2: Do Fits
            cmd_imp_fits = (
                f"combineTool.py -M Impacts -d {ws_file} -m 120 "
                f"--robustFit 1 --doFits "
                f"--setParameters {PARAMS_VAL} "
                f"{FREEZE_OPTS} "
                f"{POI_OPTS} "
                f"{IMPACT_EXCLUDE_OPTS} " # [关键]
                f"--setParameterRanges {RANGES} "
                f"--parallel 4 "
            )
            if not run_command(cmd_imp_fits, log_file=os.path.join(sub_dir, "impacts_fits.log")): continue
            
            # Impacts Step 3: JSON
            json_out = os.path.join(sub_dir, "impacts.json")
            cmd_imp_json = (
                f"combineTool.py -M Impacts -d {ws_file} -m 120 "
                f"-o {json_out} "
                f"--setParameters {PARAMS_VAL} "
                f"{FREEZE_OPTS} "
                f"{POI_OPTS} "
                f"{IMPACT_EXCLUDE_OPTS} "
                f"--setParameterRanges {RANGES} "
            )
            run_command(cmd_imp_json)
            
            # Impacts Step 4: Plot
            if os.path.exists(json_out):
                print("    Plotting Impacts...")
                cmd_plot_imp = f"plotImpacts.py -i {json_out} -o impacts"
                run_command(cmd_plot_imp)
                
                if os.path.exists("impacts.pdf"):
                    shutil.move("impacts.pdf", os.path.join(sub_dir, "impacts.pdf"))

        print(f"  [Done] Results saved in {sub_dir}")

    print("\nAll scans finished.")

if __name__ == "__main__":
    main()
