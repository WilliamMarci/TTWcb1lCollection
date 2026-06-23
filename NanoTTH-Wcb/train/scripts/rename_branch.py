#!/usr/bin/env python3
import os
import yaml
import argparse
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor

class SmartRenamer(Module):
    def __init__(self, config):
        self.new_branches_config = config.get('creates', {})
        self.mappings = [] 

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree
        existing_branches = [b.GetName() for b in inputTree.GetListOfBranches()]

        print("--- Booking New Branches ---")
        for new_name, params in self.new_branches_config.items():
            src = params['src']
            
            if src not in existing_branches:
                print(f"[WARNING] Source branch '{src}' not found. Skipping '{new_name}'.")
                continue

            # If lenVar is missing, it defaults to None (Scalar branch)
            b_type = params.get('type', 'F') 
            b_lenVar = params.get('lenVar') 
            b_title = params.get('title')

            print(f"Booking branch: {new_name} from {src}, type={b_type}, lenVar={b_lenVar}, title={b_title}")
            self.out.branch(new_name, b_type, lenVar=b_lenVar, title=b_title)
            
            self.mappings.append((new_name, src))
            
            # kind = f"Vector (len={b_lenVar})" if b_lenVar else "Scalar"
            # print(f"Created {new_name} [{kind}] from {src}")

    def analyze(self, event):
        # Simple read -> write loop
        for new_name, src_name in self.mappings:
            # print(f"Processing event {event.event} for branch '{new_name}'")
            # getattr reads both Scalars (int/float) and Vectors (arrays) correctly
            # if event.event % 10000 == 0:
                # print(f"Filling branch '{new_name}' from '{src_name}' for event {event.event}")
            val = getattr(event, src_name)
            # convert to list if it's not a scalar
            if type(val) not in [int, float]:
                val = list(val)
            # print(f"Value to fill for '{new_name}': {val}")
            self.out.fillBranch(new_name, val)
        return True

def generate_drop_file(config, filename="auto_drop.txt"):
    """Creates a file to drop the specific old branches defined in YAML."""
    drops = config.get('drops', [])
    with open(filename, "w") as f:
        f.write("keep *\n") # Keep everything else
        for d in drops:
            f.write(f"drop {d}\n")
    return filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Auto-generate the drop rules
    drop_file = generate_drop_file(config)

    # 2. Run the PostProcessor
    p = PostProcessor(
        args.output,
        [args.input],
        branchsel=None,            # Read ALL input branches
        outputbranchsel=drop_file, # Drop the old ones in output
        modules=[SmartRenamer(config)],
        provenance=True
    )
    p.run()
