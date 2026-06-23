#!/usr/bin/env python

from __future__ import print_function
import os
import sys
if len(sys.argv) not in (2, 3):
    print('usage: %s <prepid> [ <maxevent> ]' % os.path.basename(sys.argv[0]), file=sys.stderr)
    sys.exit(1)
import requests
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

prepid = sys.argv[1]
maxevent = int(sys.argv[2]) if len(sys.argv) > 2 else None
metadata = requests.get('https://cms-pdmv-prod.web.cern.ch/mcm/public/restapi/requests/get/%s' % prepid).json()['results']
mcm_dataset = metadata['dataset_name']
das_dataset = metadata['output_dataset']
if len(das_dataset) != 1:
    raise RuntimeError('unexpected DAS dataset length: %d' % len(das_dataset))
das_dataset = das_dataset[0]
print('-', mcm_dataset)
print('  -', prepid)
print('   ', 'dataset:', das_dataset)
if maxevent: print('   ', 'maxevent:', maxevent)
directory = os.path.join(basedir, 'samples', mcm_dataset, prepid)
if not os.path.isdir(directory): os.makedirs(directory)
open(os.path.join(directory, 'dataset'), 'w').write(das_dataset + '\n')
if maxevent: open(os.path.join(directory, 'maxevent'), 'w').write('%d\n' % maxevent)
