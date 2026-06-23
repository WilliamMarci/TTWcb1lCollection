#!/usr/bin/env python3
import ROOT
import sys
import os
import glob

# ==========================================
# --- Configuration (请在此处修改) ---
# ==========================================

# 输入文件路径
INPUT_PATTERN = '/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/MC/filetagged_samples_1merged_/*.root'
INPUT_PATTERN = '/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/MC/merged_samples/*.root'

# 要检查的 Branch 名称 (通常是 genEventSumw)
BRANCH_NAME = "genEventSumw"

# ==========================================
# --- Main Script ---
# ==========================================

def inspect_runs(filepath):
    print(f"\n{'='*80}")
    filename = os.path.basename(filepath)
    print(f"File: {filename}")
    
    f = ROOT.TFile.Open(filepath)
    if not f or f.IsZombie():
        print(f"  [ERROR] Cannot open file.")
        return

    # 获取 Runs Tree
    tree = f.Get("Runs")
    if not tree:
        print(f"  [ERROR] Tree 'Runs' not found.")
        f.Close()
        return

    # 检查 Branch 是否存在
    if not hasattr(tree, BRANCH_NAME) and not tree.GetBranch(BRANCH_NAME):
        print(f"  [WARNING] Branch '{BRANCH_NAME}' not found in Runs tree.")
        f.Close()
        return

    n_entries = tree.GetEntries()
    print(f"  Runs Tree Entries: {n_entries}")

    # 遍历 Runs Tree 的每一个 Entry
    for i in range(n_entries):
        tree.GetEntry(i)
        
        # 获取值
        try:
            val = getattr(tree, BRANCH_NAME)
        except AttributeError:
            print(f"    Entry {i}: [Error] Could not access attribute.")
            continue

        # 判断类型
        is_scalar = isinstance(val, (int, float))
        
        # 打印详细信息
        if is_scalar:
            print(f"    Entry {i}: [Scalar] Value = {val:.4f}")
        else:
            # 尝试作为数组/列表处理
            try:
                # 转换为列表以便查看
                val_list = list(val)
                size = len(val_list)
                
                # 如果数组太大，只显示前几个
                display_str = str(val_list)
                if size > 10:
                    display_str = f"[{val_list[0]:.2f}, {val_list[1]:.2f}, ... (Total {size} items)]"
                
                print(f"    Entry {i}: [Array]  Size = {size} | Values = {display_str}")
                
                # 额外提示：通常数组的第一个值或者是特定索引的值才是 Nominal Weight
                if size > 0:
                    print(f"              -> Index[0] = {val_list[0]:.4f} (Usually Nominal)")
                    
            except Exception as e:
                print(f"    Entry {i}: [Unknown Type] {type(val)} - {e}")

    f.Close()

if __name__ == "__main__":
    files = glob.glob(INPUT_PATTERN)
    files.sort()
    
    if not files:
        print(f"[ERROR] No files found matching: {INPUT_PATTERN}")
        sys.exit(1)
        
    print(f"Found {len(files)} files. Inspecting '{BRANCH_NAME}' in Runs tree...")
    
    for f in files:
        inspect_runs(f)
