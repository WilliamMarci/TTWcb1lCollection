#!/usr/bin/env python3
from __future__ import print_function

import os
import ast
import copy
import time

from runPostProcessing import get_arg_parser, run, tar_cmssw
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

tth_cfgname = 'tthtree_cfg.json'
default_config = {
    'channel': None,
    'usePuppiJets': True,
    'jec': False, 'jes': None, 'jes_source': '', 'jes_uncertainty_file_prefix': 'RegroupedV2_',
    'jer': 'nominal', 'jmr': None, 'met_unclustered': None, 'applyHEMUnc': False,
    'smearMET': False,
    'wcb_analysis': True,
}

# _jes_uncertainty_sources = [
#     'AbsoluteMPFBias', 'AbsoluteScale', 'AbsoluteStat', 'FlavorQCD', 'Fragmentation', 'PileUpDataMC', 'PileUpPtBB',
#     'PileUpPtEC1', 'PileUpPtEC2', 'PileUpPtHF', 'PileUpPtRef', 'RelativeBal', 'RelativeFSR', 'RelativeJEREC1',
#     'RelativeJEREC2', 'RelativeJERHF', 'RelativePtBB', 'RelativePtEC1', 'RelativePtEC2', 'RelativePtHF',
#     'RelativeStatEC', 'RelativeStatFSR', 'RelativeStatHF', 'SinglePionECAL', 'SinglePionHCAL', 'TimePtEta',
# ]
#
# jes_uncertainty_sources = {
#     '2016': _jes_uncertainty_sources + ['RelativeSample'],
#     '2017': _jes_uncertainty_sources + ['RelativeSample'],
#     '2018': _jes_uncertainty_sources,
# }

jes_uncertainty_sources = {
    '2015': ['Absolute', 'Absolute_2016', 'BBEC1', 'BBEC1_2016', 'EC2', 'EC2_2016', 'FlavorQCD', 'HF', 'HF_2016', 'RelativeBal', 'RelativeSample_2016'],
    '2016': ['Absolute', 'Absolute_2016', 'BBEC1', 'BBEC1_2016', 'EC2', 'EC2_2016', 'FlavorQCD', 'HF', 'HF_2016', 'RelativeBal', 'RelativeSample_2016'],
    '2017': ['Absolute', 'Absolute_2017', 'BBEC1', 'BBEC1_2017', 'EC2', 'EC2_2017', 'FlavorQCD', 'HF', 'HF_2017', 'RelativeBal', 'RelativeSample_2017'],
    '2018': ['Absolute', 'Absolute_2018', 'BBEC1', 'BBEC1_2018', 'EC2', 'EC2_2018', 'FlavorQCD', 'HF', 'HF_2018', 'RelativeBal', 'RelativeSample_2018'],
}

golden_json = {
    '2015': 'Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt',
    '2016': 'Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt',
    '2017': 'Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt',
    '2018': 'Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt',
}

ALL_POSSIBLE_SYSTS = ['nominal', 'pdf', 'jer', 'met', 'jes-sources', 'jes-total', 'hem']


def _base_cut(year, channel):
    # FIXME: remember to update this whenever the selections change in tthTreeProducer.py
    # FIXME: why not using ``Electron_mvaFall17V2Iso_WP90'' for 2L? ~40% gain in signal eff.

    cut_dict = {
        # ttH(bb) analysis uses tight electron ID
        # 'ele_cut': 'Electron_pt>15 && abs(Electron_eta)<2.4 && Electron_cutBased==4',
        'ele_cut': 'Electron_pt>15 && abs(Electron_eta)<2.4 && Electron_mvaFall17V2Iso_WP90',
        # 'ele_cut': 'Electron_pt>15 && abs(Electron_eta)<2.5 && Electron_mvaFall17V2Iso_WP90'
                #    ' && (abs(Electron_eta+Electron_deltaEtaSC)<1.4442 || abs(Electron_eta+Electron_deltaEtaSC)>1.5560)',
        'mu_cut': 'Muon_pt>10 && abs(Muon_eta)<2.4 && Muon_tightId && Muon_pfRelIso04_all<0.25',
        # 'mu_cut': 'Muon_pt>10 && abs(Muon_eta)<2.4', #&& Muon_tightId && Muon_pfRelIso04_all<0.25',
        'tight_ele_cut': 'Electron_pt>25 && Electron_mvaFall17V2Iso_WP80',
        'tight_mu_cut': 'Muon_pt>20 && Muon_pfRelIso04_all<0.15',
        'loose_ele_cut': 'Electron_pt>15 && abs(Electron_eta)<2.4 && Electron_mvaFall17V2Iso_WP90',
        'loose_mu_cut': 'Muon_pt>15 && abs(Muon_eta)<2.4 && Muon_pfRelIso04_all<0.25 && Muon_tightId',
        'jet_count': 'Sum$(Jet_pt>15 && abs(Jet_eta)<2.4 && (Jet_jetId & 4))',
        'fatjet_count': 'Sum$(FatJet_pt>150 && abs(FatJet_eta)<2.4)', #&& (FatJet_jetId & 2) && FatJet_msoftdrop>30)',
    }
    basesels = {
        '0L': '{jet_count}>=6',
        # 1L: *one and only one* e/mu w/ pT > 15, but *at least one* e/mu w/ pT above a (year-dependent) higher threshold (here put the lowest of the three years)
        '1L': '(Sum$({ele_cut} && {tight_ele_cut}) + Sum$({mu_cut} && {tight_mu_cut})) >= 1 && '
              '{jet_count} >= 3 && {fatjet_count} >= 1',
            #   '(Sum$({loose_ele_cut})+ Sum$({loose_mu_cut})) == 1 && '
        # 2L: exactly 2 e/mu w/ pT > 15, and at least 1 e/mu w/ pT > 25 (relaxed to 20 for mu)
        '2L': '(Sum$({ele_cut}) + Sum$({mu_cut})) >= 2 && '
              '(Sum$(Electron_pt>25 && {ele_cut}) + Sum$(Muon_pt>20 && {mu_cut})) >= 1 && '
              '{jet_count}>=3',

        # for TrigSF samples, the muon Rochester correction is not applied, so we apply the exact cuts
        '0L_TrigSF': '{jet_count}>=6',
        '1L_TrigSF': 'Sum$({ele_cut})==1 && Sum$(Muon_pt>15 && {mu_cut})==1 && '
              'Sum$({ele_cut} && {tight_ele_cut})>=1 && Sum$({mu_cut} && {tight_mu_cut})>=1 && '
              '{jet_count}>=4',
        '2L_TrigSF': '(Sum$({ele_cut}) + Sum$(Muon_pt>15 && {mu_cut})) == 2 && '
              '(Sum$(Electron_pt>25 && {ele_cut}) + Sum$(Muon_pt>25 && {mu_cut})) >= 1',
    }
    cut = basesels[channel].format(**cut_dict)
    return cut


def _process(args):
    year = args.year
    channel = args.tth_channel
    default_config['year'] = year
    default_config['channel'] = channel
    if args.training_samples:
        args.sample_dir = 'training_samples'
        default_config['training_samples'] = True
    basename = os.path.basename(args.outputdir) + '_' + year + '_' + channel
    args.outputdir = os.path.join(os.path.dirname(args.outputdir), basename, args.type)
    args.jobdir = os.path.join('jobs_%s' % basename, args.type)
    args.datasets = '%s/%s/%s_%s.yaml' % (args.sample_dir, year, channel, args.type)
    args.cut = _base_cut(year, channel)
    args.imports = [('PhysicsTools.NanoTTH.producers.tthTreeProducer', 'tthTreeFromConfig')]

    systs_to_run = []
    if args.type == 'MC':
        if args.run_syst is None:
            systs_to_run = ['nominal']
        elif args.run_syst == 'full':
            systs_to_run = ['pdf', 'jer', 'met', 'jes-sources', 'hem']
        elif args.run_syst == 'lite':
            systs_to_run = ['nominal', 'jer', 'met', 'jes-total', 'hem']
        else:
            systs_to_run = args.run_syst.split(',')
    elif args.type != 'Data':
        systs_to_run = ['nominal']
    for syst in systs_to_run:
        if syst not in ALL_POSSIBLE_SYSTS:
            raise RuntimeError(f'Unrecognized systematics {syst}. Supported ones: {ALL_POSSIBLE_SYSTS}.')

    if args.type == 'Data':
        logging.info('Start making Data trees...')
        cfg = copy.deepcopy(default_config)
        cfg['runModules'] = False
        cfg['jes'] = None
        cfg['jer'] = None
        cfg['jmr'] = None
        cfg['met_unclustered'] = None
        opts = copy.deepcopy(args)
        opts.extra_transfer = os.path.expandvars(
            '$CMSSW_BASE/src/PhysicsTools/NanoTTH/data/JSON/%s' % golden_json[year])
        opts.json = golden_json[year]
        run(opts, configs={tth_cfgname: cfg})
    else:
        if 'nominal' in systs_to_run:
            logging.info(f'Start making nominal {args.type} trees...')
            cfg = copy.deepcopy(default_config)
            opts = copy.deepcopy(args)
            run(opts, configs={tth_cfgname: cfg})

        # w/ PDF/Scale weights
        if 'pdf' in systs_to_run:
            logging.info('Start making nominal trees with PDF/scale weights...')
            syst_name = 'LHEWeight'
            cfg = copy.deepcopy(default_config)
            opts = copy.deepcopy(args)
            opts.outputdir = os.path.join(os.path.dirname(opts.outputdir), syst_name)
            opts.jobdir = os.path.join(os.path.dirname(opts.jobdir), syst_name)
            opts.branchsel_out = 'keep_and_drop_output_LHEweights.txt'
            run(opts, configs={tth_cfgname: cfg})

        # JER up/down
        if 'jer' in systs_to_run:
            for variation in ['up', 'down']:
                syst_name = 'jer_%s' % variation
                logging.info('Start making %s trees...' % syst_name)
                cfg = copy.deepcopy(default_config)
                cfg['fillSystWeights'] = False
                cfg['jer'] = variation
                opts = copy.deepcopy(args)
                opts.outputdir = os.path.join(os.path.dirname(opts.outputdir), syst_name)
                opts.jobdir = os.path.join(os.path.dirname(opts.jobdir), syst_name)
                opts.branchsel_out = 'keep_and_drop_output_forSystTrees.txt'
                run(opts, configs={tth_cfgname: cfg})

        # MET unclustEn up/down
        if 'met' in systs_to_run:
            for variation in ['up', 'down']:
                syst_name = 'met_%s' % variation
                logging.info('Start making %s trees...' % syst_name)
                cfg = copy.deepcopy(default_config)
                cfg['fillSystWeights'] = False
                cfg['met_unclustered'] = variation
                opts = copy.deepcopy(args)
                opts.outputdir = os.path.join(os.path.dirname(opts.outputdir), syst_name)
                opts.jobdir = os.path.join(os.path.dirname(opts.jobdir), syst_name)
                opts.branchsel_out = 'keep_and_drop_output_forSystTrees.txt'
                run(opts, configs={tth_cfgname: cfg})

        # split JES sources
        if 'jes-sources' in systs_to_run:
            for source in jes_uncertainty_sources[year]:
                for variation in ['up', 'down']:
                    syst_name = 'jes_%s_%s' % (source, variation)
                    logging.info('Start making %s trees...' % syst_name)
                    cfg = copy.deepcopy(default_config)
                    cfg['fillSystWeights'] = False
                    cfg['jes_source'] = source
                    cfg['jes'] = variation
                    opts = copy.deepcopy(args)
                    opts.outputdir = os.path.join(os.path.dirname(opts.outputdir), syst_name)
                    opts.jobdir = os.path.join(os.path.dirname(opts.jobdir), syst_name)
                    opts.branchsel_out = 'keep_and_drop_output_forSystTrees.txt'
                    run(opts, configs={tth_cfgname: cfg})

        # JES (Total) up/down
        if 'jes-total' in systs_to_run:
            for variation in ['up', 'down']:
                syst_name = 'jes_%s' % variation
                logging.info('Start making %s trees...' % syst_name)
                cfg = copy.deepcopy(default_config)
                cfg['fillSystWeights'] = False
                cfg['jes'] = variation
                opts = copy.deepcopy(args)
                opts.outputdir = os.path.join(os.path.dirname(opts.outputdir), syst_name)
                opts.jobdir = os.path.join(os.path.dirname(opts.jobdir), syst_name)
                opts.branchsel_out = 'keep_and_drop_output_forSystTrees.txt'
                run(opts, configs={tth_cfgname: cfg})

        # HEM15/16 unc for 2018
        if year == '2018' and 'hem' in systs_to_run:
            for variation in ['down']:
                syst_name = 'HEMIssue_%s' % variation
                logging.info('Start making %s trees...' % syst_name)
                cfg = copy.deepcopy(default_config)
                cfg['fillSystWeights'] = False
                cfg['applyHEMUnc'] = True
                opts = copy.deepcopy(args)
                opts.outputdir = os.path.join(os.path.dirname(opts.outputdir), syst_name)
                opts.jobdir = os.path.join(os.path.dirname(opts.jobdir), syst_name)
                opts.branchsel_out = 'keep_and_drop_output_forSystTrees.txt'
                run(opts, configs={tth_cfgname: cfg})


def _main(args):

    if not (args.post or args.add_weight or args.merge):
        tar_cmssw(args.tarball_suffix)

    if args.tth_all:
        years = ['2015', '2016', '2017', '2018']
        channels = ['0L', '1L', '2L']
        categories = ['Data', 'MC']
    else:
        years = args.year.split(',')
        channels = args.tth_channel.split(',')
        categories = args.type.split(',')

    for year in years:
        for chn in channels:
            for cat in categories:
                opts = copy.deepcopy(args)
                if cat == 'Data':
                    opts.nfiles_per_job *= 2
                if opts.inputdir:
                    opts.inputdir = opts.inputdir.rstrip('/').replace('_YEAR_', year)
                    assert (year in opts.inputdir)
                    base_dir_name = 'Data' if cat == 'Data' else 'MC'
                    if opts.inputdir.rsplit('/', 1)[1] not in ['Data', 'MC']:
                        opts.inputdir = os.path.join(opts.inputdir, base_dir_name)
                    assert (opts.inputdir.endswith(base_dir_name))
                opts.type = cat
                opts.year = year
                opts.tth_channel = chn
                logging.info('inputdir=%s, year=%s, channel=%s, cat=%s, syst=%s', opts.inputdir, opts.year,
                             opts.tth_channel, opts.type, opts.run_syst)
                _process(opts)


def main():
    parser = get_arg_parser()

    parser.add_argument(
        '--year', required=False,
        help='Year: 2015 (2016 preVFP), 2016 (2016 postVFP), 2017, 2018, or comma separated list e.g., `2016,2017,2018`')

    parser.add_argument(
        '--type', type=str,
        help='Run `Data` or `mc` or `syst`, or a combination of them with a comma-separated string (e.g., `Data,mc`)')

    parser.add_argument('--tth-channel',
                        required=False,
                        help='VH channel: 0L, 1L, 2L, 0L_TrigSF, 1L_TrigSF, 2L_TrigSF'
                        )

    parser.add_argument('--run-syst', default=None,
                        help='Run the systematic trees. The options include: '
                        '`full`: include PDF weights and use individual JEC sources; '
                        '`lite`: exclude PDF weights (but keep scale/PS weights) and use total JEC uncertainty; '
                        'a comma separated list of the systematics to run, e.g., `jer,met`. '
                        f'The available systematics include {ALL_POSSIBLE_SYSTS}. '
                        'Default: %(default)s'
                        )

    parser.add_argument('--tth-all',
                        action='store_true', default=False,
                        help='Run over all three years and all channels. Default: %(default)s'
                        )

    parser.add_argument('--sample-dir',
                        type=str,
                        default='custom_samples',
                        help='Directory of the sample list files. Default: %(default)s'
                        )

    parser.add_argument('--training-samples',
                        action='store_true', default=False,
                        help='Run over samples for event classifier training. Default: %(default)s'
                        )

    parser.add_argument(
        '--wait', type=int, default=-1,
        help='To be used together with `--post --batch` to keep the postprocessing waiting until all jobs finished.'
        ' The value is the number of seconds to wait between two trials.')

    parser.add_argument('--po', '--producer-option', dest='producer_option',
                        nargs=2, action='append', default=[],
                        help='options to pass to the producer, e.g., `--po apply_tight_selection False`')

    args = parser.parse_args()

    if args.wait > 0:
        if args.post or args.add_weight or args.merge:
            while True:
                _main(args)
                print('... waiting ...')
                time.sleep(args.wait)
    else:
        producer_options = {k: ast.literal_eval(v) for k, v in args.producer_option}
        if len(producer_options):
            logging.info(f'Updating default_config with options: {producer_options}')
            default_config.update(producer_options)

        _main(args)


if __name__ == '__main__':
    main()
