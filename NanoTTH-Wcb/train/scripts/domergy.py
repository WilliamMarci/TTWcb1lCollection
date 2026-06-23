#!/usr/bin/env python3
import os
import sys
import yaml
import subprocess
import shutil
from array import array
import ROOT
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# --- Configuration (自动转换为绝对路径) ---
# ==========================================

# 获取当前脚本所在目录，确保相对路径正确解析
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 基础路径
BASE_PATH = '/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/MC/'
# BASE_PATH = '/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/Data/'

# 模式选择
types = 'all' # 'train' or 'all'

if types == 'train':
    INPUT_DIR = os.path.join(BASE_PATH, 'filetagged_samples') 
    MERGED_DIR = os.path.join(BASE_PATH, 'filetagged_samples_1merged_')
    FINAL_DIR = os.path.join(BASE_PATH, 'filetagged_samples_2final')
else:
    INPUT_DIR = os.path.join(BASE_PATH, 'scored_samples') 
    MERGED_DIR = os.path.join(BASE_PATH, 'scored_samples_1merged_')
    FINAL_DIR = os.path.join(BASE_PATH, 'scored_samples_2final')
    # INPUT_DIR = os.path.join(BASE_PATH, 'scored_samples_v1') 
    # MERGED_DIR = os.path.join(BASE_PATH, 'scored_samples_1merged_v1')
    # FINAL_DIR = os.path.join(BASE_PATH, 'scored_samples_2final_v1')

PART_DIR = os.path.join(MERGED_DIR, 'part')

# 使用绝对路径，防止子进程找不到文件
SAMPLE_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'sample/1L_MC.yaml')
XSEC_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'sample/xsec.conf')
FILETAG_CONFIG_PATH = os.path.join(SCRIPT_DIR, 'patch/filetag_v1.yaml')
HADDNANO_PATH = 'haddnano.py' # 假设在 PATH 中，如果不在，请用 os.path.join(SCRIPT_DIR, 'haddnano.py')
ADD_BRANCH_SCRIPT = os.path.join(SCRIPT_DIR, 'add_branch.py')

LUMI = 1000.0 

# ==========================================
# --- Helper Functions ---
# ==========================================

def get_xsec_dict(filename):
    xsec_map = {}
    if not filename or not os.path.exists(filename):
        print(f"[WARNING] Xsec file not found: {filename}")
        return xsec_map
        
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                xsec_val = float(eval(parts[0]))
                dataset_path = parts[1]
                clean_path = dataset_path.lstrip('/')
                dataset_name = clean_path.split('/')[0]
                xsec_map[dataset_name] = xsec_val
            except Exception as e:
                print(f"[WARNING] Could not parse line: '{line}'. Error: {e}")
                continue
    return xsec_map

def add_weight_branch(file, xsec, lumi=1000.0, treename='Events', wgtbranch='xsecWeight'):
    """
    智能极速加权方案 (UPDATE 模式) - 修复版
    """
    print(f"   -> Adding weight branch to {os.path.basename(file)} (xsec={xsec:.4g}, lumi={lumi})")
    ROOT.PyConfig.IgnoreCommandLineOptions = True

    # --- 内部函数：核心加权逻辑 ---
    def _append_weight_logic(target_file):
        f = ROOT.TFile.Open(target_file, 'UPDATE')
        if not f or f.IsZombie(): raise RuntimeError(f"Cannot open {target_file}")
        
        run_tree = f.Get('Runs')
        event_tree = f.Get(treename)
        
        # 1. 创建直方图对象 (不要过早删除!)
        htmp = ROOT.TH1D('htmp', 'htmp', 1, 0, 10)
        ROOT.gROOT.SetBatch(True)
        
        # 计算 sumW
        run_tree.Project('htmp', '1.0', 'genEventSumw')
        sumwgts = float(htmp.Integral())
        
        if sumwgts == 0:
            print(f"      [WARNING] genEventSumw is 0, skipping.")
            f.Close()
            return

        xsecwgt = xsec * lumi / sumwgts
        
        # 辅助：填充函数
        def _fill_branch(name, data_array, len_var=None):
            if event_tree.GetBranch(name): return 
            
            if len_var:
                b = event_tree.Branch(name, data_array, f'{name}[{len_var}]/F')
            else:
                b = event_tree.Branch(name, data_array, f'{name}/F')
            
            n_entries = event_tree.GetEntries()
            b.SetBasketSize(max(n_entries * 8, 32000))
            
            for i in range(n_entries):
                b.Fill()
            b.ResetAddress()

        # 1. 主权重
        _fill_branch(wgtbranch, array('f', [xsecwgt]))

        # 2. LHE Norm
        if event_tree.GetBranch('LHEScaleWeight'):
            run_tree.GetEntry(0)
            n = getattr(run_tree, 'nLHEScaleSumw', 0)
            if n > 0:
                vals = []
                for i in range(n):
                    # 复用 htmp 对象
                    run_tree.Project('htmp', '1.0', f'LHEScaleSumw[{i}]*genEventSumw')
                    denom = htmp.Integral()
                    vals.append(sumwgts / denom if denom != 0 else 0)
                _fill_branch('LHEScaleWeightNorm', array('f', vals), 'nLHEScaleWeight')

        # 3. PDF Norm
        if event_tree.GetBranch('LHEPdfWeight'):
            run_tree.GetEntry(0)
            n = getattr(run_tree, 'nLHEPdfSumw', 0)
            if n > 0:
                vals = []
                for i in range(n):
                    run_tree.Project('htmp', '1.0', f'LHEPdfSumw[{i}]*genEventSumw')
                    denom = htmp.Integral()
                    vals.append(sumwgts / denom if denom != 0 else 0)
                _fill_branch('LHEPdfWeightNorm', array('f', vals), 'nLHEPdfWeight')

        # 4. PS Norm
        if event_tree.GetBranch('PSWeight') and run_tree.GetBranch('PSSumw'):
            run_tree.GetEntry(0)
            n = getattr(run_tree, 'nPSSumw', 0)
            if n > 0:
                vals = []
                for i in range(n):
                    run_tree.Project('htmp', '1.0', f'PSSumw[{i}]*genEventSumw')
                    denom = htmp.Integral()
                    vals.append(sumwgts / denom if denom != 0 else 0)
                _fill_branch('PSWeightNorm', array('f', vals), 'nPSWeight')

        event_tree.Write(treename, ROOT.TObject.kOverwrite)
        # 清理
        del htmp
        f.Close()

    # --- 主流程 ---
    try:
        f_check = ROOT.TFile.Open(file)
        if not f_check: 
            print(f"      [ERROR] Cannot open {file}")
            return
        t_check = f_check.Get(treename)
        has_branch = False
        if t_check and t_check.GetBranch(wgtbranch):
            has_branch = True
        f_check.Close()

        target_file = file
        tmp_clean_file = file.replace(".root", "_clean_tmp.root")

        if has_branch:
            print(f"      [INFO] Branch {wgtbranch} exists. Pruning old branch (Fast Clone)...")
            f_old = ROOT.TFile.Open(file)
            t_old = f_old.Get(treename)
            t_old.SetBranchStatus(wgtbranch, 0)
            for b in ['LHEScaleWeightNorm', 'LHEPdfWeightNorm', 'PSWeightNorm']:
                if t_old.GetBranch(b): t_old.SetBranchStatus(b, 0)

            f_new = ROOT.TFile.Open(tmp_clean_file, "RECREATE")
            t_new = t_old.CloneTree(-1, "fast")
            
            r_old = f_old.Get("Runs")
            if r_old:
                r_new = r_old.CloneTree()
            
            f_new.Write()
            f_new.Close()
            f_old.Close()
            
            target_file = tmp_clean_file

        _append_weight_logic(target_file)

        if has_branch:
            shutil.move(tmp_clean_file, file)
            print(f"      [INFO] Overwritten original file with new weights.")

    except Exception as e:
        print(f"      [ERROR] Failed processing {file}: {e}")
        if os.path.exists(tmp_clean_file): os.remove(tmp_clean_file)

def run_haddnano(output_file, input_files):
    if not input_files: return False
    if os.path.exists(output_file): os.remove(output_file)
    cmd = [HADDNANO_PATH, str(output_file)] + [str(f) for f in input_files]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0: return True
        else:
            print(f"  -> Haddnano Failed. Error:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"  -> Execution Error: {e}")
        return False

def load_yaml(path):
    if os.path.exists(path):
        with open(path, 'r') as f: return yaml.safe_load(f)
    print(f"Error: Config file {path} not found.")
    sys.exit(1)

def run_add_filetag_worker(args_tuple):
    """
    Step 3 Worker.
    修正：-o 参数现在接收 temp_dir (目录)，而不是文件路径。
    """
    filename, input_path, final_output_path, temp_dir, script_path, config_path = args_tuple
    
    # 1. 告诉脚本把结果输出到 temp_dir
    # 假设 add_branch.py 会在 temp_dir 下生成同名文件
    cmd = [sys.executable, script_path, '-i', input_path, '-o', temp_dir, '-c', config_path]
    
    # 预期生成的本地文件路径
    expected_local_file = os.path.join(temp_dir, filename)
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stderr = (proc.stderr or '').strip()
        stdout = (proc.stdout or '').strip()
        
        if proc.returncode == 0:
            # 检查文件是否真的生成了
            if os.path.exists(expected_local_file):
                try:
                    # 拷贝回 EOS
                    shutil.copy(expected_local_file, final_output_path)
                    os.remove(expected_local_file) # 清理
                    return {'filename': filename, 'ok': True, 'stderr': stderr, 'stdout': stdout, 'returncode': 0}
                except Exception as copy_e:
                    return {'filename': filename, 'ok': False, 'stderr': f"Copy to EOS failed: {copy_e}", 'stdout': stdout, 'returncode': 1}
            else:
                return {'filename': filename, 'ok': False, 'stderr': f"Script finished but {filename} not found in {temp_dir}", 'stdout': stdout, 'returncode': 1}
        else:
            if os.path.exists(expected_local_file): os.remove(expected_local_file)
            msg = stderr if stderr else stdout
            return {'filename': filename, 'ok': False, 'stderr': msg, 'stdout': stdout, 'returncode': proc.returncode}
            
    except Exception as e:
        if os.path.exists(expected_local_file): os.remove(expected_local_file)
        return {'filename': filename, 'ok': False, 'stderr': str(e), 'stdout': '', 'returncode': None}

# ==========================================
# --- Main Logic ---
# ==========================================

def main():
    # 0. 检查环境
    if not os.path.exists(ADD_BRANCH_SCRIPT):
        print(f"[ERROR] Script not found: {ADD_BRANCH_SCRIPT}")
        print("Please check the path or run from the correct directory.")
        sys.exit(1)

    # 1. 设置临时目录
    user_name = os.environ.get('USER', 'user')
    base_tmp = os.environ.get('TMPDIR', f'/tmp/{user_name}')
    work_tmp_dir = os.path.join(base_tmp, 'wcb_merge_work')
    
    if not os.path.exists(work_tmp_dir):
        try:
            os.makedirs(work_tmp_dir)
        except OSError:
            print(f"[WARNING] Cannot create tmp dir {work_tmp_dir}, using current dir.")
            work_tmp_dir = "."
            
    print(f"[INFO] Using Local Temp Directory: {work_tmp_dir}")

    # 2. 准备输出目录
    for d in [MERGED_DIR, FINAL_DIR, PART_DIR]:
        if not os.path.exists(d): os.makedirs(d)

    print(f"[INFO] Loading configurations...")
    print(f"[INFO] Input Directory: {INPUT_DIR}")
    print(f"[INFO] Merged Directory: {MERGED_DIR}")
    
    sample_config = load_yaml(SAMPLE_CONFIG_PATH)
    xsec_map = get_xsec_dict(XSEC_CONFIG_PATH)
    input_path_obj = Path(INPUT_DIR)

    # ==========================================
    # Step 1: Process Datasets (Pieces -> Local Part -> Weight -> EOS)
    # ==========================================
    print(f"\n[STEP 1] Processing Datasets (Pieces -> Part -> Weight)...")
    category_parts_map = {}

    for category, datasets in sample_config.items():
        print(f"Analyzing Category: {category}")
        category_parts_map[category] = []
        
        for dataset_prefix in datasets:
            raw_files = [str(f) for f in input_path_obj.iterdir() if f.is_file() and f.name.startswith(dataset_prefix) and f.name.endswith('.root')]
            if not raw_files: continue
            
            part_filename = f"{dataset_prefix}_part.root" 
            final_part_file = os.path.join(PART_DIR, part_filename)      
            local_part_file = os.path.join(work_tmp_dir, part_filename)  
            
            if os.path.exists(final_part_file):
                print(f"  -> [SKIP] {part_filename} already exists.")
                category_parts_map[category].append(final_part_file)
                continue

            print(f"  -> Merging {len(raw_files)} pieces into LOCAL: {part_filename}")
            
            if run_haddnano(local_part_file, raw_files):
                if dataset_prefix in xsec_map:
                    add_weight_branch(local_part_file, xsec_map[dataset_prefix], lumi=LUMI)
                else:
                    print(f"  [WARNING] No xsec found for {dataset_prefix}, skipping weighting.")
                
                print(f"  -> Copying to EOS: {final_part_file}")
                try:
                    shutil.copy(local_part_file, final_part_file)
                    category_parts_map[category].append(final_part_file)
                except Exception as e:
                    print(f"  [ERROR] Failed to copy to EOS: {e}")
                
                if os.path.exists(local_part_file): os.remove(local_part_file)
            else:
                print(f"  [ERROR] Failed to merge parts for {dataset_prefix}")

    # ==========================================
    # Step 2: Merge Parts into Category
    # ==========================================
    print(f"\n[STEP 2] Merging Parts into Categories...")
    for category, part_files in category_parts_map.items():
        if not part_files: continue
        
        output_filename = f"{category}_merged.root" 
        final_output_file = os.path.join(MERGED_DIR, output_filename)
        local_output_file = os.path.join(work_tmp_dir, output_filename)
        
        print(f"  -> Merging {len(part_files)} dataset parts into Category: {output_filename}")
        
        if run_haddnano(local_output_file, part_files):
            print(f"  -> Copying to EOS: {final_output_file}")
            try:
                shutil.copy(local_output_file, final_output_file)
            except Exception as e:
                print(f"  [ERROR] Copy failed: {e}")
            
            if os.path.exists(local_output_file): os.remove(local_output_file)
        else:
             print(f"  [ERROR] Failed category merge for {category}")

    # ==========================================
    # Step 3: Add Filetag (Parallel with Tmp)
    # ==========================================
    print(f"\n[STEP 3] Adding Filetag Branches...")

    env_workers = os.environ.get('FILETAG_WORKERS')
    try:
        max_workers = int(env_workers) if env_workers is not None else (os.cpu_count() or 4)
    except ValueError:
        max_workers = os.cpu_count() or 4

    log_path = os.path.join(MERGED_DIR, 'filetag_errors.log')
    logging.basicConfig(filename=log_path, level=logging.ERROR,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    tasks = []
    for filename in os.listdir(MERGED_DIR):
        if not filename.endswith('.root'): continue
        input_path = os.path.join(MERGED_DIR, filename)
        if os.path.isdir(input_path): continue
        
        output_path = os.path.join(FINAL_DIR, filename)
        
        # 将绝对路径传递给 Worker，避免 Worker 环境中路径解析错误
        tasks.append((filename, input_path, output_path, work_tmp_dir, ADD_BRANCH_SCRIPT, FILETAG_CONFIG_PATH))

    total = len(tasks)
    if total == 0:
        print("No files to filetag.")
    else:
        print(f"Dispatching {total} filetag tasks with {max_workers} workers.")
        success = 0
        fail = 0
        done = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_add_filetag_worker, t): t[0] for t in tasks}
            
            for fut in as_completed(futures):
                done += 1
                filename = futures.get(fut, '<unknown>')
                try:
                    res = fut.result()
                except Exception as e:
                    logging.error(f"Executor error for {filename}: {e}")
                    print(f"[{done}/{total}] ERROR: {filename} (executor crash: {e})")
                    fail += 1
                    continue

                if res.get('ok'):
                    success += 1
                    print(f"[{done}/{total}] Success: {filename}")
                else:
                    fail += 1
                    err = res.get('stderr') or f"exit {res.get('returncode')}"
                    logging.error(f"Failed filetag {filename}: {err}")
                    print(f"[{done}/{total}] ERROR: {filename} (Check log)")

        print(f"\nFiletagging complete. Total={total}, Success={success}, Failed={fail}")
        if fail > 0:
            print(f"Errors logged to: {os.path.abspath(log_path)}")

    try:
        if os.path.exists(work_tmp_dir) and not os.listdir(work_tmp_dir):
            os.rmdir(work_tmp_dir)
    except:
        pass

    print(f"\n[DONE] All tasks completed.")

if __name__ == "__main__":
    main()
