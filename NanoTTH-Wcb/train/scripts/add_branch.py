#!/usr/bin/env python3
import os
import yaml
import glob
import argparse
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor

class FileTagger(Module):
    def __init__(self, config):
        self.groups = config.get('file_groups', {})
        self.branch_defs = config.get('branches', {})
        
        # This will store the calculated values for the CURRENT file being processed
        # Format: {'branch_name': value_to_fill}
        self.current_file_values = {}

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree
        
        # 1. Identify the current file name (remove path)
        # inputFile is a TFile, GetName() returns the path
        fname = os.path.basename(inputFile.GetName())
        print(f"\n--- Processing File: {fname} ---")

        # 2. Determine which group(s) this file belongs to
        matched_groups = []
        for group_name, substrings in self.groups.items():
            for s in substrings:
                if s in fname:
                    matched_groups.append(group_name)
                    break # Found a match for this group, move to next group
        
        print(f"Matched Groups: {matched_groups}")

        # 3. Book Branches and Calculate Values
        self.current_file_values = {}
        
        for br_name, br_config in self.branch_defs.items():
            b_type = br_config.get('type', 'I')
            b_title = br_config.get('title', '')
            b_default = br_config.get('default', 0)
            b_mappings = br_config.get('mappings', {})

            # Determine value for this file
            final_val = b_default
            
            # Check if any matched group has a specific value mapping
            # Priority: Last matching group in list overrides previous ones (or you can add logic)
            for group in matched_groups:
                if group in b_mappings:
                    final_val = b_mappings[group]

            # Store the value to use in analyze()
            self.current_file_values[br_name] = final_val
            
            # Create the branch
            print(f"Booking branch '{br_name}' (Type: {b_type}) with constant value: {final_val}")
            self.out.branch(br_name, b_type, title=b_title)

    def analyze(self, event):
        # 4. Fill the branches with the pre-calculated constant values
        for br_name, val in self.current_file_values.items():
            self.out.fillBranch(br_name, val)
        return True

def get_files(input_path):
    """Returns a list of root files from a directory or a single file."""
    if input_path.endswith('.root'):
        return [input_path]
    elif os.path.isdir(input_path):
        # Look for all .root files in the directory
        return glob.glob(os.path.join(input_path, "*.root"))
    else:
        raise ValueError(f"Input {input_path} is neither a ROOT file nor a directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tag NanoAOD files with new branches based on filenames.")
    parser.add_argument("-i", "--input", required=True, help="Input directory or .root file")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-c", "--config", required=True, help="YAML configuration file")
    
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Get list of files
    files = get_files(args.input)
    if not files:
        print(f"No root files found in {args.input}")
        exit(1)

    print(f"Found {len(files)} files to process.")

    # Run PostProcessor
    # Note: We run PostProcessor on the list of files. 
    # The 'FileTagger' module's beginFile() will be called for EACH file individually,
    # allowing us to set the correct value for that specific file.
    
    p = PostProcessor(
        args.output,
        files,
        cut=None,
        branchsel=None,
        modules=[FileTagger(config)],
        provenance=True,
        fwkJobReport=False
    )
    p.run()
