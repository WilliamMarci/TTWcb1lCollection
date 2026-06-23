#!/usr/bin/env python2

from __future__ import print_function
import os
import sys
import re

basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(basedir, 'python'))

PREFETCH_PREFIX = '/eos/user/l/legao/hss/prefetch'

import sample

samples = sample.list_samples()

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print('Usage: %s <url> | %s remove <path>' % ((os.path.basename(sys.argv[0]),) * 2))
    print('\nPrefetched files:')
    for dataset_samples in samples.values():
        for sample in dataset_samples.values():
            if not sample.prefetch: continue
            print('\n  - %s/%s:\n' % (sample.mcm_dataset, sample.mcm_prepid))
            for name in sample.prefetch:
                print('      - %s' % name)
    sys.exit(0)
if len(sys.argv) == 3 and sys.argv[1] == 'remove':
    dataset_in, prepid_in, name_in = re.search('^([^/]+)(?:/([^/]+))?(?:/([^/]+))?$', sys.argv[2]).groups()
    for dataset, dataset_samples in samples.items():
        if dataset_in != '*' and dataset != dataset_in: continue
        for prepid, sample in dataset_samples.items():
            if prepid_in is not None and prepid_in != '*' and prepid != prepid_in: continue
            for name, dest in sample.prefetch.items():
                if name_in is not None and name_in != '*' and name != name_in: continue
                print('removing prefetch: %s/%s/%s' % (dataset, prepid, name))
                os.remove(os.path.join(sample.directory, 'prefetch-' + name))
                os.remove(dest[5:])
    sys.exit(0)
url = sys.argv[1]

name, dataset = re.search(r'^root://[^/]*/(/store/mc/[^/]*/([^/]*)/MINIAODSIM/.*\.root)$', url).groups()
for sample in samples[dataset].values():
    for file in sample.filelist:
        file = file['file'][0]
        if file['name'] == name:
            if os.system("xrdcp '%s' '%s'" % (url, PREFETCH_PREFIX + name)) == 0:
                open(os.path.join(sample.directory, 'prefetch-%s' % os.path.basename(name)), 'w').write('file:' + PREFETCH_PREFIX + name)
