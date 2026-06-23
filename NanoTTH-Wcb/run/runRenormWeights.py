#!/usr/bin/env python3
from __future__ import print_function

import os
import ast
import copy
import time

from runPostProcessing import get_arg_parser, run, tar_cmssw
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

tth_cfgname = 'tthrenorm_cfg.json'
default_config = {
}

def _process(args):
    year = args.year
    default_config['year'] = year
    basename = os.path.basename(args.outputdir) + '_' + year
    args.outputdir = os.path.join(os.path.dirname(args.outputdir), basename, args.type)
    args.jobdir = os.path.join('jobs_%s' % basename, args.type)
    args.datasets = '%s/%s/%s.yaml' % (args.sample_dir, year, args.type)
    args.imports = [('PhysicsTools.NanoTTH.producers.renormWeightProducer', 'renormWeightFromConfig')]
    args.cut = "(1)"

    # w/ PDF/Scale weights
    logging.info('Start making nominal trees with PDF/scale weights...')
    syst_name = 'LHEWeight'
    opts = copy.deepcopy(args)
    cfg = copy.deepcopy(default_config)
    opts.outputdir = os.path.join(os.path.dirname(opts.outputdir), syst_name)
    opts.jobdir = os.path.join(os.path.dirname(opts.jobdir), syst_name)
    opts.branchsel_out = 'keep_and_drop_output_renorm.txt'
    run(opts, configs={tth_cfgname: cfg})


def _main(args):

    if not (args.post or args.add_weight or args.merge):
        tar_cmssw(args.tarball_suffix)

    years = args.year.split(',')
    categories = args.type.split(',')

    for year in years:
        for cat in categories:
            opts = copy.deepcopy(args)
            if opts.inputdir:
                opts.inputdir = opts.inputdir.rstrip('/').replace('_YEAR_', year)
                assert (year in opts.inputdir)
                base_dir_name = 'MC'
                if opts.inputdir.rsplit('/', 1)[1] not in ['MC']:
                    opts.inputdir = os.path.join(opts.inputdir, base_dir_name)
                assert (opts.inputdir.endswith(base_dir_name))
            opts.type = cat
            opts.year = year
            logging.info('inputdir=%s, year=%s, syst=%s', opts.inputdir, opts.year, opts.type)
            _process(opts)


def main():
    parser = get_arg_parser()

    parser.add_argument(
        '--year', required=False,
        help='Year: 2015 (2016 preVFP), 2016 (2016 postVFP), 2017, 2018, or comma separated list e.g., `2016,2017,2018`')

    parser.add_argument(
        '--type', type=str,
        help='Run `mc`, or a combination of them with a comma-separated string (e.g., `mc`)')

    parser.add_argument('--sample-dir',
                        type=str,
                        default='samples',
                        help='Directory of the sample list files. Default: %(default)s'
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
