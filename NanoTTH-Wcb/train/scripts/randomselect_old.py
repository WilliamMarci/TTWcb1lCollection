#!/usr/bin/env python3
import os
import yaml
import shutil
import random
import argparse
import sys
import subprocess
from pathlib import Path
from array import array
import ROOT

# --- Helper Functions ---

def get_xsec_dict(filename):
    """Parses the xsec file and returns a dict {dataset_name: xsec}"""
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
                # Support simple formulas like 10.32*(3*0.108535)
                # We use eval() to calculate the value from the string
                xsec_val = float(eval(parts[0]))
                
                dataset_path = parts[1]
                # Extract the primary dataset name (e.g., /Name/Campaign/Tier -> Name)
                # Remove leading slash if present and take the first component
                clean_path = dataset_path.lstrip('/')
                dataset_name = clean_path.split('/')[0]
                
                xsec_map[dataset_name] = xsec_val
            except Exception as e:
                print(f"[WARNING] Could not parse line: '{line}'. Error: {e}")
                continue
    return xsec_map

def add_weight_branch(file, xsec, lumi=1000., treename='Events', wgtbranch='xsecWeight'):
    """Adds xsec weight and normalization branches to the ROOT file."""
    print(f"   -> Adding weight branch to {os.path.basename(file)} (xsec={xsec:.4g})")
    ROOT.PyConfig.IgnoreCommandLineOptions = True

    def _get_sum(tree, wgtvar):
        htmp = ROOT.TH1D('htmp', 'htmp', 1, 0, 10)
        tree.Project('htmp', '1.0', wgtvar)
        return float(htmp.Integral())

    def _fill_const_branch(tree, branch_name, buff, lenVar=None):
        if lenVar is not None:
            b = tree.Branch(branch_name, buff, '%s[%s]/F' % (branch_name, lenVar))
            b_lenVar = tree.GetBranch(lenVar)
            buff_lenVar = array('I', [0])
            b_lenVar.SetAddress(buff_lenVar)
        else:
            b = tree.Branch(branch_name, buff, branch_name + '/F')

        b.SetBasketSize(tree.GetEntries() * 2)  # be sure we do not trigger flushing
        for i in range(tree.GetEntries()):
            if lenVar is not None:
                b_lenVar.GetEntry(i)
            b.Fill()

        b.ResetAddress()
        if lenVar is not None:
            b_lenVar.ResetAddress()

    f = ROOT.TFile(file, 'UPDATE')
    run_tree = f.Get('Runs')
    tree = f.Get(treename)
    
    if not tree or not run_tree:
        print(f"[ERROR] Could not find tree {treename} or Runs in {file}")
        f.Close()
        return

    # fill cross section weights to the 'Events' tree
    sumwgts = _get_sum(run_tree, 'genEventSumw')
    if sumwgts == 0:
        print(f"[WARNING] genEventSumw is 0 in {file}, skipping weight calculation.")
        f.Close()
        return
        
    xsecwgt = xsec * lumi / sumwgts
    xsec_buff = array('f', [xsecwgt])
    _fill_const_branch(tree, wgtbranch, xsec_buff)

    # fill LHE weight re-normalization factors
    if tree.GetBranch('LHEScaleWeight'):
        run_tree.GetEntry(0)
        if hasattr(run_tree, 'nLHEScaleSumw'):
            nScaleWeights = run_tree.nLHEScaleSumw
            scale_weight_norm_buff = array('f',
                                           [sumwgts / _get_sum(run_tree, 'LHEScaleSumw[%d]*genEventSumw' % i)
                                            for i in range(nScaleWeights)])
            _fill_const_branch(tree, 'LHEScaleWeightNorm', scale_weight_norm_buff, lenVar='nLHEScaleWeight')

    if tree.GetBranch('LHEPdfWeight'):
        run_tree.GetEntry(0)
        if hasattr(run_tree, 'nLHEPdfSumw'):
            nPdfWeights = run_tree.nLHEPdfSumw
            pdf_weight_norm_buff = array('f',
                                         [sumwgts / _get_sum(run_tree, 'LHEPdfSumw[%d]*genEventSumw' % i)
                                          for i in range(nPdfWeights)])
            _fill_const_branch(tree, 'LHEPdfWeightNorm', pdf_weight_norm_buff, lenVar='nLHEPdfWeight')

    # fill PS weight re-normalization factors
    if tree.GetBranch('PSWeight') and run_tree.GetBranch('PSSumw'):
        run_tree.GetEntry(0)
        if hasattr(run_tree, 'nPSSumw'):
            nPSWeights = run_tree.nPSSumw
            ps_weight_norm_buff = array('f',
                                        [sumwgts / _get_sum(run_tree, 'PSSumw[%d]*genEventSumw' % i)
                                         for i in range(nPSWeights)])
            _fill_const_branch(tree, 'PSWeightNorm', ps_weight_norm_buff, lenVar='nPSWeight')

    tree.Write(treename, ROOT.TObject.kOverwrite)
    f.Close()


class DatasetSampler:
    def __init__(self, config_path, history_path, origin_path, train_path, ratio):
        self.config_path = config_path
        self.history_path = history_path
        self.origin_path = Path(origin_path)
        self.train_path = Path(train_path)
        self.ratio = ratio
        self.config = self._load_yaml(config_path)
        
        # Ensure directories exist
        if not self.origin_path.exists():
            print(f"Error: Origin path {self.origin_path} does not exist.")
            sys.exit(1)
        if not self.train_path.exists():
            os.makedirs(self.train_path)

    def _load_yaml(self, path):
        if os.path.exists(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        return None

    def _save_yaml(self, data, path):
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def _interactive_check(self, target_path, auto_yes=False):
        """Checks if file exists and asks user to overwrite."""
        if target_path.exists():
            if auto_yes:
                try:
                    if target_path.is_dir():
                        shutil.rmtree(target_path)
                    else:
                        os.remove(target_path)
                    return True
                except Exception as e:
                    print(f"Error removing {target_path}: {e}")
                    return False

            while True:
                choice = input(f"\n[WARNING] Target file exists: {target_path}\nOverwrite? (y/n): ").lower()
                if choice == 'y':
                    try:
                        if target_path.is_dir():
                            shutil.rmtree(target_path)
                        else:
                            os.remove(target_path)
                    except Exception as e:
                        print(f"Error removing {target_path}: {e}")
                        return False
                    return True
                elif choice == 'n':
                    print(f"Skipping...")
                    return False
                else:
                    print("Please enter 'y' or 'n'.")
        return True

    def _interactive_move(self, src, dst):
        """Moves file with interactive collision check."""
        if not self._interactive_check(dst):
            return False
        
        try:
            shutil.move(str(src), str(dst))
            print(f"Moved: {src.name} -> {dst.parent}")
            return True
        except Exception as e:
            print(f"Error moving {src.name}: {e}")
            return False

    def get_all_files_by_type(self):
        files_map = {}
        for category, datasets in self.config.items():
            files_map[category] = {}
            for dataset_prefix in datasets:
                found_files = [
                    f for f in self.origin_path.iterdir() 
                    if f.is_file() and f.name.startswith(dataset_prefix)
                ]
                files_map[category][dataset_prefix] = found_files
        return files_map

    def select_samples(self, files_map):
        to_train = []
        to_origin = []
        print(f"Selecting samples with ratio {self.ratio}...")

        for category, datasets in files_map.items():
            for dataset_name, files in datasets.items():
                total_count = len(files)
                if total_count == 0:
                    continue

                select_count = int(total_count * self.ratio)
                if select_count < 1 and total_count > 0:
                    select_count = 1
                
                selected = random.sample(files, select_count)
                remaining = [f for f in files if f not in selected]

                to_train.extend([f.name for f in selected])
                to_origin.extend([f.name for f in remaining])
                print(f"  [{category}] {dataset_name}: Selected {select_count}/{total_count}")

        return to_train, to_origin

    def run(self):
        history = self._load_yaml(self.history_path)

        if history:
            print(f"History found at {self.history_path}. Replaying move operation...")
            train_files = history.get('train', [])
            for fname in train_files:
                src = self.origin_path / fname
                dst = self.train_path / fname
                if src.exists():
                    self._interactive_move(src, dst)
                elif dst.exists():
                    print(f"File already in target: {fname}")
                else:
                    print(f"Warning: File {fname} missing.")
        else:
            print("No history found. Performing new random selection...")
            files_map = self.get_all_files_by_type()
            train_files, origin_files = self.select_samples(files_map)

            moved_actual = []
            for fname in train_files:
                src = self.origin_path / fname
                dst = self.train_path / fname
                if self._interactive_move(src, dst):
                    moved_actual.append(fname)

            new_history = {
                'info': {
                    'ratio': self.ratio,
                    'origin_path': str(self.origin_path),
                    'train_path': str(self.train_path)
                },
                'origin': origin_files, 
                'train': moved_actual
            }
            self._save_yaml(new_history, self.history_path)
            print(f"History saved to {self.history_path}")

    def reset(self):
        print("Reset option selected. Moving files back to origin...")
        history = self._load_yaml(self.history_path)
        
        if not history:
            print("No history file found. Cannot perform automated reset.")
            return

        files_to_move = history.get('train', [])
        for fname in files_to_move:
            src = self.train_path / fname
            dst = self.origin_path / fname
            if src.exists():
                self._interactive_move(src, dst)
            else:
                print(f"Warning: File {fname} not found in train path.")

        if history:
            os.remove(self.history_path)
            print(f"History file {self.history_path} deleted.")
        print("Reset complete.")

    def _run_haddnano(self, haddnano_path, output_file, input_files):
        """Helper to run haddnano command"""
        cmd = [haddnano_path, str(output_file)] + [str(f) for f in input_files]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  -> Success.")
                return True
            else:
                print(f"  -> Failed. Error:\n{result.stderr}")
                return False
        except Exception as e:
            print(f"  -> Execution Error: {e}")
            return False

    def merge_samples(self, merge_out_path, haddnano_path, xsec_config=None):
        """
        Merges files in train_path. 
        If xsec_config is provided:
          1. Merge individual dataset files into 'part' directory.
          2. Add xsecWeight to these part files.
          3. Merge part files into final category file.
        """
        print(f"\nStarting Merge Process...")
        print(f"Haddnano script: {haddnano_path}")
        print(f"Output directory: {merge_out_path}")

        merge_out = Path(merge_out_path)
        if not merge_out.exists():
            os.makedirs(merge_out)

        # Load xsec map if config provided
        xsec_map = {}
        if xsec_config:
            print(f"Using Xsec Config: {xsec_config}")
            xsec_map = get_xsec_dict(xsec_config)
            part_dir = merge_out / "part"
            if not part_dir.exists():
                os.makedirs(part_dir)
            print(f"Intermediate files will be stored in: {part_dir}")

        for category, datasets in self.config.items():
            files_to_final_merge = []
            
            print(f"[{category}] Processing...")

            if xsec_config:
                # --- XSEC FLOW: Merge by Dataset -> Weight -> Merge by Category ---
                for dataset_prefix in datasets:
                    # 1. Find raw files for this specific dataset
                    raw_files = [
                        f for f in self.train_path.iterdir() 
                        if f.is_file() and f.name.startswith(dataset_prefix) and f.name.endswith('.root')
                    ]

                    if not raw_files:
                        continue

                    # 2. Merge to part file
                    part_filename = f"{dataset_prefix}_part.root"
                    part_file = part_dir / part_filename
                    
                    # Always overwrite part files to ensure clean state
                    if part_file.exists():
                        os.remove(part_file)

                    print(f"  Merging {len(raw_files)} files into part: {part_filename}...")
                    if self._run_haddnano(haddnano_path, part_file, raw_files):
                        # 3. Add Weight
                        # We look for the dataset_prefix in the xsec_map
                        if dataset_prefix in xsec_map:
                            add_weight_branch(str(part_file), xsec_map[dataset_prefix])
                            files_to_final_merge.append(part_file)
                        else:
                            print(f"  [WARNING] No xsec found for {dataset_prefix}, skipping this dataset in final merge.")
            
            else:
                # --- NORMAL FLOW: Collect all files -> Merge by Category ---
                for dataset_prefix in datasets:
                    found = [
                        f for f in self.train_path.iterdir() 
                        if f.is_file() and f.name.startswith(dataset_prefix) and f.name.endswith('.root')
                    ]
                    files_to_final_merge.extend(found)

            # --- Final Merge Step ---
            if not files_to_final_merge:
                print(f"  No files to merge for category {category}.")
                continue

            output_filename = f"{category}_train.root"
            output_file = merge_out / output_filename

            if not self._interactive_check(output_file):
                continue

            print(f"  Final Merge: {len(files_to_final_merge)} files into {output_filename}...")
            self._run_haddnano(haddnano_path, output_file, files_to_final_merge)

    def process_individual_files(self, output_path_str, xsec_config):
        """
        Copies individual files to output_path, renames them by prepending the category,
        and adds xsec weights.
        """
        print(f"\nStarting Individual File Processing (Rename + Weight)...")
        output_path = Path(output_path_str)
        if not output_path.exists():
            os.makedirs(output_path)

        # Load xsec map
        xsec_map = get_xsec_dict(xsec_config)

        for category, datasets in self.config.items():
            print(f"[{category}] Processing individual files...")
            for dataset_prefix in datasets:
                # Find files in train_path (the selected files)
                current_files = [
                    f for f in self.train_path.iterdir()
                    if f.is_file() and f.name.startswith(dataset_prefix) and f.name.endswith('.root')
                ]

                if not current_files:
                    continue

                # Check xsec availability
                if dataset_prefix not in xsec_map:
                    print(f"  [WARNING] No xsec found for {dataset_prefix}, skipping weights for these files.")
                    xsec_val = None
                else:
                    xsec_val = xsec_map[dataset_prefix]

                for src_file in current_files:
                    # Rename format: Category_OriginalName
                    new_filename = f"{category}_{src_file.name}"
                    dst_file = output_path / new_filename

                    # Copy file first (overwrite if exists to ensure clean state)
                    try:
                        shutil.copy2(src_file, dst_file)
                        print(f"  -> Copied: {src_file.name} -> {new_filename}")
                        
                        # Add weight if xsec is available
                        if xsec_val is not None:
                            add_weight_branch(str(dst_file), xsec_val)
                    except Exception as e:
                        print(f"  [ERROR] Failed to process {src_file.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Randomly select files for training and optionally merge them.")
    parser.add_argument('--config', required=True, help='Path to dataset config yaml')
    parser.add_argument('--history', default='history.yaml', help='Path to history yaml')
    parser.add_argument('--origin', required=True, help='Source directory path')
    parser.add_argument('--train', required=True, help='Target training sample directory path')
    parser.add_argument('--ratio', type=float, default=0.1, help='Selection ratio (0.0 - 1.0)')
    parser.add_argument('--reset', action='store_true', help='Move files back to origin based on history')
    
    # Merge Arguments
    parser.add_argument('--merge-out', help='If provided, merge train samples to this directory')
    parser.add_argument('--haddnano', default='haddnano.py', help='Path to haddnano.py script')
    parser.add_argument('--xsec-config', help='Path to xsec file. If provided, adds xsecWeight.')

    # New Argument for renamed individual files
    parser.add_argument('--piece-xsec-path', help='If provided, copy, rename (Category_Name), and weight individual files to this directory')

    args = parser.parse_args()

    sampler = DatasetSampler(
        config_path=args.config,
        history_path=args.history,
        origin_path=args.origin,
        train_path=args.train,
        ratio=args.ratio
    )

    if args.reset:
        sampler.reset()
    else:
        # 1. Perform Selection/Move
        sampler.run()
    
    # 2. Perform Merge if requested
    if args.merge_out:
        sampler.merge_samples(args.merge_out, args.haddnano, args.xsec_config)

    # 3. Perform Individual File Processing if requested
    if args.piece_xsec_path:
        if not args.xsec_config:
            print("[ERROR] --xsec-config is required when using --piece-xsec-path")
        else:
            sampler.process_individual_files(args.piece_xsec_path, args.xsec_config)
        

if __name__ == "__main__":
    main()
