#!/usr/bin/env python3
import os
import yaml
import shutil
import random
import argparse
import sys
import subprocess
from pathlib import Path

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

    def _interactive_check(self, target_path):
        """Checks if file exists and asks user to overwrite."""
        if target_path.exists():
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

    def merge_samples(self, merge_out_path, haddnano_path):
        """
        Merges files in train_path based on config categories.
        """
        print(f"\nStarting Merge Process...")
        print(f"Haddnano script: {haddnano_path}")
        print(f"Output directory: {merge_out_path}")

        merge_out = Path(merge_out_path)
        if not merge_out.exists():
            os.makedirs(merge_out)


        for category, datasets in self.config.items():
            # 1. Collect all files in train_path that belong to this category
            files_to_merge = []
            for dataset_prefix in datasets:
                # Find files in train_path starting with this prefix
                found = [
                    f for f in self.train_path.iterdir() 
                    if f.is_file() and f.name.startswith(dataset_prefix) and f.name.endswith('.root')
                ]
                files_to_merge.extend(found)

            if not files_to_merge:
                print(f"[{category}] No files found in train sample to merge. Skipping.")
                continue

            # 2. Define output filename
            output_filename = f"{category}_train.root"
            output_file = merge_out / output_filename

            # 3. Check for overwrite
            if not self._interactive_check(output_file):
                continue

            # 4. Execute haddnano command
            # Command format: python haddnano.py output.root input1.root input2.root ...
            cmd = [haddnano_path, str(output_file)] + [str(f) for f in files_to_merge]
            
            print(f"[{category}] Merging {len(files_to_merge)} files into {output_filename}...")
            try:
                # Using subprocess to call the external script
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"  -> Success.")
                else:
                    print(f"  -> Failed. Error:\n{result.stderr}")
            except Exception as e:
                print(f"  -> Execution Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Randomly select files for training and optionally merge them.")
    parser.add_argument('--config', required=True, help='Path to dataset config yaml')
    parser.add_argument('--history', default='history.yaml', help='Path to history yaml')
    parser.add_argument('--origin', required=True, help='Source directory path')
    parser.add_argument('--train', required=True, help='Target training sample directory path')
    parser.add_argument('--ratio', type=float, default=0.1, help='Selection ratio (0.0 - 1.0)')
    parser.add_argument('--reset', action='store_true', help='Move files back to origin based on history')
    
    # New Merge Arguments
    parser.add_argument('--merge-out', help='If provided, merge train samples to this directory')
    parser.add_argument('--haddnano', default='haddnano.py', help='Path to haddnano.py script')

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
        sampler.merge_samples(args.merge_out, args.haddnano)
        

if __name__ == "__main__":
    main()
