#!/usr/bin/env python3
import ROOT
import sys
import os
import glob
from collections import defaultdict

# ==========================================
# --- Configuration (请在此处修改) ---
# ==========================================

# 输入文件路径 (支持通配符 *)
# 示例: '/path/to/merged/*.root' 或 '/path/to/part/ST_*.root'
# INPUT_PATTERN = '/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/MC/filetagged_samples_1merged_/part/*.root'
INPUT_PATTERN = '/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/MC/filetagged_samples_1merged_/*.root'
# INPUT_PATTERN = '/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/MC/merged_samples/part/*.root'

# 要检查的 Branch 名称
# 示例: 'xsecWeight', 'filetag', 'year'
TARGET_BRANCH = 'xsecWeight'

# 浮点数精度 (保留几位小数，用于归类)
# 如果是整数类型的 branch (如 filetag)，这个参数不影响
PRECISION = 6

# 是否只扫描前 N 个事件 (设为 -1 则扫描所有事件)
# 对于非常大的文件，设为 10000 可以快速检查
MAX_EVENTS = 10000

# ==========================================
# --- Main Script ---
# ==========================================

def inspect_file(filepath):
    print(f"\n{'='*80}")
    print(f"File: {os.path.basename(filepath)}")
    # print(f"Path: {filepath}")
    
    f = ROOT.TFile.Open(filepath)
    if not f or f.IsZombie():
        print(f"  [ERROR] Cannot open file.")
        return

    tree = f.Get("Events")
    if not tree:
        print(f"  [ERROR] Tree 'Events' not found.")
        f.Close()
        return

    if not tree.GetBranch(TARGET_BRANCH):
        print(f"  [WARNING] Branch '{TARGET_BRANCH}' not found in this file.")
        f.Close()
        return

    total_events = tree.GetEntries()
    scan_events = total_events if MAX_EVENTS == -1 else min(total_events, MAX_EVENTS)
    
    print(f"  Total Events: {total_events} (Scanning: {scan_events})")
    
    # 统计字典 {value: count}
    val_counts = defaultdict(int)
    
    # 禁用所有 branch，只启用需要的，提高速度
    tree.SetBranchStatus("*", 0)
    tree.SetBranchStatus(TARGET_BRANCH, 1)

    for i in range(scan_events):
        tree.GetEntry(i)
        val = getattr(tree, TARGET_BRANCH)
        
        # 如果是浮点数，进行舍入处理
        if isinstance(val, float):
            val = round(val, PRECISION)
        
        # 针对 vector 或 array 类型的处理 (取第一个值，或者你可以修改逻辑)
        if hasattr(val, '__len__') and not isinstance(val, str):
             if len(val) > 0:
                 val = round(val[0], PRECISION) if isinstance(val[0], float) else val[0]
             else:
                 val = "empty"

        val_counts[val] += 1

    print(f"  Branch: '{TARGET_BRANCH}' Statistics:")
    print(f"  {'-'*50}")
    print(f"  {'Value':<25} | {'Count':<10} | {'Ratio':<10}")
    print(f"  {'-'*50}")
    
    sorted_items = sorted(val_counts.items(), key=lambda x: x[0] if isinstance(x[0], (int, float)) else str(x[0]))
    
    for val, count in sorted_items:
        ratio = count / scan_events * 100
        print(f"  {str(val):<25} | {count:<10} | {ratio:.1f}%")

    if len(val_counts) > 1:
        print(f"  [NOTE] Found {len(val_counts)} different values.")
    else:
        print(f"  [OK] Uniform value.")

    f.Close()

if __name__ == "__main__":
    # 处理通配符
    files = glob.glob(INPUT_PATTERN)
    files.sort()
    
    if not files:
        print(f"[ERROR] No files found matching: {INPUT_PATTERN}")
        sys.exit(1)
        
    print(f"Found {len(files)} files. Inspecting branch '{TARGET_BRANCH}'...")
    
    for f in files:
        inspect_file(f)
