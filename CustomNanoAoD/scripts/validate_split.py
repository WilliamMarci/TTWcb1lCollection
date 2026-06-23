#!/usr/bin/env python2

from __future__ import print_function
import os
import sys

basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(basedir, 'python'))

import sample

samples = sample.list_samples()

import re
split_name_pattern = re.compile(r'^(.*)-(\d)$')

def split_name(name):
    r = split_name_pattern.search(name)
    if not r: return None
    return r.groups()

split_samples = { }
for dataset, dataset_samples in samples.items():
    print('-', dataset)
    for prepid, samples in dataset_samples.items():
        print(' ', '+', prepid)
        split_result = split_name(prepid)
        if not split_result: continue
        origin, sid = split_result
        split_samples[origin] = split_samples.get(origin, [(-1, dataset_samples[origin])]) + [(int(sid), samples)]

for origin, list_of_samples in split_samples.items():
    print('\nChecking: %s' % origin)
    sids = set()
    samples_grouped_by_name = { }
    for sid, samples in list_of_samples:
        sids.add(sid)
        for nevent, filename in samples.select(2**128 - 1):
            basename = os.path.basename(filename)
            group = samples_grouped_by_name.get(basename, { })
            group[sid] = nevent
            samples_grouped_by_name[basename] = group
    sids = sorted(sids)
    first = True
    for name, group in sorted(samples_grouped_by_name.items()):
        if first:
            first = False
            print(' ' * len(name), *(sids + ['diff']), sep='\t')
        #if len(group) < len(sids): continue
        got = sum(group.get(sid, 0) for sid in sids[1:])
        if got == 0: continue
        expected = group.get(sids[0], 0)
        diff = got - expected
        #if diff == 0: continue
        print(name, *([group.get(sid, '-') for sid in sids] + [diff]), sep='\t')
