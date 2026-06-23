#
# source /cvmfs/sft.cern.ch/lcg/views/LCG_102rc1/x86_64-centos7-gcc11-opt/setup.sh
#
import ROOT
import numpy as np
from correctionlib.schemav2 import (
    Category,
    Correction,
    VERSION,
    CorrectionSet
)

import sys
import os
import glob
import numpy as np

import optparse
parser = optparse.OptionParser()
parser.add_option("-o", dest="outfile")
parser.add_option("-d", dest="indir")
(opts, args) = parser.parse_args()

in_files = glob.glob(os.path.join(opts.indir, "*.root"))

content_files = []
content_files2 = []
for f in in_files:
    file_name = os.path.basename(f).replace("_tree.root","")
    print (file_name)
    rf = ROOT.TFile.Open(f)
    keys = [k.GetName() for k in list(rf.GetListOfKeys())]
    keys = [k for k in keys if k.startswith("sumWgts")]

    content_keys = []
    content_keys2 = []
    for key in keys:
        print(key)
        weight_name = key.replace("sumWgts_","").replace("sumWgts","all")
        h = rf.Get(key)
        entries = np.array([h.GetBinContent(i+1) for i in range(h.GetNbinsX())])
        rel_rate = entries/entries[0]
        weights = [h.GetXaxis().GetBinLabel(i+1) for i in range(h.GetNbinsX())]
        # print (entries)
        # print (weights)

        content_weights = []
        content_weights2 = []
        
        for i, label in enumerate(weights):
            if label=="": break
            # fill the actual weights
            content_weights.append({"key": label, "value": np.nan_to_num(rel_rate[i], nan=1.0, posinf=1.0, neginf=1.0)})
            content_weights2.append({"key": label, "value": np.nan_to_num(entries[i], nan=1.0, posinf=1.0, neginf=1.0)})
            print(label, rel_rate[i])

        # make weight dimension
        cat_weight = Category.parse_obj(
            {
                "nodetype": "category", "input": "weight", "content": content_weights
            }
        )
        cat_weight2 = Category.parse_obj(
            {
                "nodetype": "category", "input": "weight", "content": content_weights2
            }
        )
        content_keys.append({"key": weight_name, "value": cat_weight})
        content_keys2.append({"key": weight_name, "value": cat_weight2})

    # make key dimension
    cat_key = Category.parse_obj(
        {
            "nodetype": "category", "input": "process", "content": content_keys
        }
    )
    cat_key2 = Category.parse_obj(
        {
            "nodetype": "category", "input": "process", "content": content_keys2
        }
    )
    content_files.append({"key": file_name, "value": cat_key})
    content_files2.append({"key": file_name, "value": cat_key2})

# make file dimension
cat_file = Category.parse_obj(
    {
        "nodetype": "category", "input": "file", "content": content_files
    }
)
cat_file2 = Category.parse_obj(
    {
        "nodetype": "category", "input": "file", "content": content_files2
    }
)

# make cset
correction = {
    "version": 1,
    "name": "normalization",
    "description": "...",        
    "inputs": [
        {"name": "file",    "type": "string"},
        {"name": "process", "type": "string"},
        {"name": "weight",  "type": "string"},
    ],
    "output": {"name": "factor", "type": "real"},
    "data": cat_file
}
correction2 = {
    "version": 1,
    "name": "entries",
    "description": "...",        
    "inputs": [
        {"name": "file",    "type": "string"},
        {"name": "process", "type": "string"},
        {"name": "weight",  "type": "string"},
    ],
    "output": {"name": "factor", "type": "real"},
    "data": cat_file2
}
 
json = Correction.parse_obj(correction)
json2 = Correction.parse_obj(correction2)
cset = CorrectionSet.parse_obj(
    {
        "schema_version": VERSION,
        "corrections": [json, json2],
    }
)
with open(opts.outfile, "w") as f:
    f.write(            
        cset.json(exclude_unset=True, indent=2)
    )

