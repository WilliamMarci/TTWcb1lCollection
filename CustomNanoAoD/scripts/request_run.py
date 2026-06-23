#!/usr/bin/env python3

from __future__ import print_function
import os
import sys

basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(basedir, 'python'))

import sample
import traceback

samples = sample.list_samples()

if len(sys.argv) < 2 or len(sys.argv) > 4:
    print('Usage: %s [ <prepid> [ <nevent> [ <dryrun> ] ] ]' % os.path.basename(sys.argv[0]))
    print('\nAvailable prepids:')
    for dataset, dataset_samples in sorted(samples.items()):
        print('\n-', dataset)
        for prepid, sample in sorted(dataset_samples.items()):
            print(' ', '+', prepid, '(%d events%s)' % (sample.count(), ', %d maximum' % sample.maxevent if sample.maxevent is not None else ''))
    sys.exit(0)

def generate_x509up(x509up=None):
    x509up = x509up or os.path.abspath(os.path.join(basedir, 'scripts', 'x509up'))
    if os.popen("2>/dev/null voms-proxy-info --file '%s'" % x509up).read().find('timeleft  : 191') < 0:
        if os.system("voms-proxy-init -voms cms -valid 192:00 -out '%s'" % x509up):
            raise RuntimeError('error generating x509 user proxy')
    return x509up

def check_success(fileout, nevents):
    os.system("touch '%s'" % fileout)
    import ROOT
    errorIgnoreLevel = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = ROOT.kFatal
    success = False
    try:
        #return os.stat(fileout).st_size >= 12*4096  # fast check
        tfile = ROOT.TFile(fileout)
        success = tfile.Get('Events').GetEntries() == nevents  # reliable check
    except Exception:
        #traceback.print_exc()
        pass
    ROOT.gErrorIgnoreLevel = errorIgnoreLevel
    return success

def eos_to_xrd(path):
    if path[:4] == '/eos': return 'root://eosuser.cern.ch/' + path
    return path

def request(dataset, prepid, sample, target_nevents=None, dryrun=False, outdir=None):
    jobstr = '''Universe = vanilla
Executable = %s

+ProjectName="cms.org.cern"

X509UP = %s
PROG = %s
NTHREAD = 8
Arguments = $(X509UP) $(PROG) $(NEVENT) $(NTHREAD) $(FILEIN) $(FILEOUT)

request_cpus = 8
request_memory = 4096
use_x509userproxy = True
x509userproxy = $(X509UP)

on_exit_remove        = (ExitBySignal == False) && (ExitCode == 0)
on_exit_hold          = (ExitBySignal == True) || (ExitCode != 0)
on_exit_hold_reason   = strcat("Job held by ON_EXIT_HOLD due to ", ifThenElse((ExitBySignal == True), strcat("exit signal ", ExitSignal), strcat("exit code ", ExitCode)), ".")
periodic_release      = (NumJobStarts < 3) && ((CurrentTime - EnteredCurrentStatus) > 60*60)

+MaxRuntime = 4*60*60

Log    = $(LOGPREFIX).log
Output = $(LOGPREFIX)_1.log
Error  = $(LOGPREFIX)_2.log

should_transfer_files = YES
transfer_input_files = ""
transfer_output_files = ""
Queue NEVENT, FILEIN, FILEOUT, LOGPREFIX from (
%s)'''
    isdata = dataset in '/MET/ /SingleMuon/ /SingleElectron/ /EGamma/'  # [XXX]
    if isdata:
        if '2016' in prepid:
            year = '2016APV' if 'HIPM' in prepid else '2016'
        elif '2017' in prepid:
            year = '2017'
        elif '2018' in prepid:
            year = '2018'
        else:
            raise RuntimeError('year not recognized in prepid: %s' % prepid)
    else:
        if 'UL16' in prepid:
            year = '2016APV' if 'APV' in prepid else '2016'
        elif 'UL17' in prepid:
            year = '2017'
        elif 'UL18' in prepid:
            year = '2018'
        else:
            raise RuntimeError('year not recognized in prepid: %s' % prepid)
    if not outdir:
        user = __import__('getpass').getuser()
        outdir = f'/eos/user/{user[0]}/{user}/CustomizedNanoAOD/V0/{year}/{"Data" if isdata else "MC"}'
    executable = os.path.abspath(os.path.join(basedir, 'scripts', 'x509run'))
    x509up = generate_x509up()
    prog = os.path.abspath(os.path.join(basedir, 'scripts', f'run-{"data" if isdata else "mc"}-{year}.sh'))
    outdir = os.path.join(outdir, dataset, prepid)
    logdir = os.path.join(basedir, 'scripts', 'log', dataset, prepid)
    if os.system("mkdir -p '%s' '%s'" % (outdir, logdir)):
        raise RuntimeError('error making directories')
    queue = ''
    for nevents, filein in sample.select(target_nevents):
        filename = os.path.basename(filein)
        fileout = os.path.join(outdir, filename.replace('MiniAODv2', 'CustomizedNanoAODv9'))
        success = check_success(fileout, nevents)
        print('%s %s' % (('Skipping' if success else 'Adding'), fileout))
        if success: continue
        #os.close(os.open(fileout, os.O_WRONLY | os.O_TRUNC))  # truncate
        #logprefix = os.path.join(logdir, os.path.splitext(filename)[0])
        logprefix = os.path.join(logdir, 'condor')
        filein, fileout = map(eos_to_xrd, (filein, fileout))
        queue += '%s, %s, %s, %s\n' % (nevents, filein, fileout, logprefix)
    jobfile = prepid + '.jdl'
    open(jobfile, 'w').write(jobstr % (executable, x509up, prog, queue))
    ((print() or print) if dryrun else os.system)("condor_submit -file '%s'" % jobfile)

prepid = sys.argv[1]
nevent = (int(sys.argv[2]) if len(sys.argv) > 2 else None) or None
dryrun = eval(sys.argv[3]) if len(sys.argv) > 3 else False
if prepid == 'describe':
    prefix = ' ' * 6
    for dataset, dataset_samples in sorted(samples.items()):
        for prepid, sample in sorted(dataset_samples.items()):
            print('%s- name:' % prefix, sample.mcm_prepid)
            print('%s  dataset:' % prefix, sample.mcm_dataset)
            print('%s  nevent:' % prefix, min(sample.maxevent or (2**128 - 1), sample.count()))
            print('%s  xs-name:' % prefix, sample.xs['MCM'] if sample.xs else 'null')
            print('%s  xs:' % prefix, sample.xs['cross_section'] if sample.xs else 'null')
            print()
    sys.exit(0)
if prepid == 'list':
    for dataset, dataset_samples in sorted(samples.items()):
        print('%s:' % dataset)
        for prepid, sample in sorted(dataset_samples.items()):
            print('  - %s:' % prepid)
            for nevents, filein in sample.select(min(nevent if nevent else (2**128 - 1), sample.maxevent)):
                print('      - %s' % filein)
    sys.exit(0)
for dataset, dataset_samples in samples.items():
    if prepid != 'all' and prepid not in dataset_samples: continue
    for pid in (dataset_samples.keys() if prepid == 'all' else [prepid]):
        request(dataset, pid, dataset_samples[pid], nevent, dryrun)
    if prepid != 'all': break
else:
    if prepid != 'all': raise RuntimeError('prepid not recognized: %s' % prepid)
