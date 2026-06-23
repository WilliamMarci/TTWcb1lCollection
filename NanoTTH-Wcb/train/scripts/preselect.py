#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
from array import array

# --- Configuration ---
base_path = '/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/MC/'
# base_path = '/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/mc-F_AKM-T_ALL/'
origin_sample_path = f'{base_path}/pieces/'
train_sample_path = f'{base_path}/train_samples/'   # 
merge_sample_path = f'{base_path}/merged_samples/'  # use train samples to merge the pieces with xsec info, and then split them into smaller pieces for training
xsec_split_path = f'{base_path}/xsec_split/'
rename_sample_path_global = f'{base_path}/renamed_samples/'        #rename the branches in merged samples.
filetag_sample_path = f'{base_path}/filetagged_samples/'    #mark the samples with filetag branch, 
history_path = f'{train_sample_path}/history.yaml'
ratio = 0.1

sample_config_path = 'sample/1L_MC.yaml'
rename_map_path = 'patch/rename_v1.yaml'
filetag_config_path = 'patch/filetag_v1.yaml'
xsec_file_path = 'sample/xsec.conf'
ratio_config_path = 'patch/ratio.yaml'


def main():
    start_time = os.popen('date').read().strip()
    # mkdir -p
    rename_sample_path = rename_sample_path_global  # directly use global rename path for now, skipping the rename step for now.
    subprocess.run(['mkdir', '-p', train_sample_path])
    subprocess.run(['mkdir', '-p', origin_sample_path])
    subprocess.run(['mkdir', '-p', os.path.dirname(history_path)])
    subprocess.run(['mkdir', '-p', merge_sample_path])
    subprocess.run(['mkdir', '-p', rename_sample_path])
    subprocess.run(['mkdir', '-p', filetag_sample_path])
    subprocess.run(['mkdir', '-p', xsec_split_path])

    parser = argparse.ArgumentParser(description="Randomly select samples for training and optionally merge them.")
    parser.add_argument("-r", "--reset", action="store_true", help="Reset the history file.")
    args = parser.parse_args()

    # step 1: sample from origin to train
    cmd = ['./randomselect.py',
              '--config', sample_config_path,
              '--history', history_path,
              '--origin', origin_sample_path,
              '--train', train_sample_path,
              '--ratio', str(ratio),
              '--ratio-config', ratio_config_path,
              '--merge-out', merge_sample_path,
              '--piece-xsec-path', xsec_split_path,
              '--xsec-config', xsec_file_path,
              ]

    if args.reset:
        cmd.append('--reset')
        print("[PRE-TRAIN] Resetting history and exiting.")
        subprocess.run(cmd)
        return 
  
    print("[PRE-TRAIN]Executing random selection...")
    subprocess.run(cmd)
    
    # copy history to working dir
    os.system(f'cp {history_path} ./history.yaml')

    # step 2: rename branches in train samples [SKIP]
    print("[PRE-TRAIN]Skipping branch renaming for now...")
    rename_sample_path = merge_sample_path  # directly use merged samples for next step, skipping the rename step for now.
    # print("[PRE-TRAIN]Renaming branches in training samples...")
    # # for input_file in os.listdir(xsec_split_path):
    # for input_file in os.listdir(merge_sample_path):
    #     if not input_file.endswith('.root'):
    #         continue
    #     # input_path = os.path.join(xsec_split_path, input_file)
    #     input_path = os.path.join(merge_sample_path, input_file)
    #     output_path = rename_sample_path
    #     cmd = ['./rename_branch.py',
    #               '-i', input_path,
    #               '-o', output_path,
    #               '-c', rename_map_path,]
    #     subprocess.run(cmd)

    # step 3: add filetag branches
    print("[PRE-TRAIN]Adding filetag branches to renamed samples...")
    for input_file in os.listdir(rename_sample_path):
        if not input_file.endswith('.root'):
            continue
        input_path = os.path.join(rename_sample_path, input_file)
        output_path = filetag_sample_path
        cmd = ['./add_branch.py',
                  '-i', input_path,
                  '-o', output_path,
                  '-c', filetag_config_path,]
        subprocess.run(cmd)

    # step 4: summarize
    # rm renamebranch dir
    subprocess.run(['rm', '-r', rename_sample_path])
    
    print("[PRE-TRAIN]Summary of prepared training samples:")
    end_time = os.popen('date').read().strip()
    run_time = os.popen(f'date -u -d "{end_time}" +%s').read().strip()
    start_time_sec = os.popen(f'date -u -d "{start_time}" +%s').read().strip()
    total_seconds = int(run_time) - int(start_time_sec)
    print(f"Start Time: {start_time}/ End Time: {end_time}/ Total Time: {total_seconds} seconds")
    print(f"result samples are in {filetag_sample_path}")

if __name__ == "__main__":
    main()
#MC/pieces_1merged_/