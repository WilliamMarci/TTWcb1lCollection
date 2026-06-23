#!/usr/bin/env python3
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def add_weight_branch(file, xsec, lumi=1000., treename='Events', wgtbranch='xsecWeight'):
    from array import array
    import ROOT
    ROOT.PyConfig.IgnoreCommandLineOptions = True

    def _get_sum(tree, wgtvar):
        htmp = ROOT.TH1D('htmp', 'htmp', 1, 0, 10)
        tree.Project('htmp', '1.0', wgtvar)
        return float(htmp.Integral())

    def _fill_const_branch(tree, branch_name, buff, lenVar=None):
        if lenVar is not None:
            b = tree.Branch(branch_name, buff, '%s[%s]/F' % (branch_name, lenVar))
            b_lenVar = tree.GetBranch(lenVar)
            buff_lenVar = array('I', [0])
            b_lenVar.SetAddress(buff_lenVar)
        else:
            b = tree.Branch(branch_name, buff, branch_name + '/F')

        b.SetBasketSize(tree.GetEntries() * 2)  # be sure we do not trigger flushing
        for i in range(tree.GetEntries()):
            if lenVar is not None:
                b_lenVar.GetEntry(i)
            b.Fill()

        b.ResetAddress()
        if lenVar is not None:
            b_lenVar.ResetAddress()

    f = ROOT.TFile(file, 'UPDATE')
    run_tree = f.Get('Runs')
    tree = f.Get(treename)

    # fill cross section weights to the 'Events' tree
    sumwgts = _get_sum(run_tree, 'genEventSumw')
    xsecwgt = xsec * lumi / sumwgts
    xsec_buff = array('f', [xsecwgt])
    _fill_const_branch(tree, wgtbranch, xsec_buff)

    # fill LHE weight re-normalization factors
    if tree.GetBranch('LHEScaleWeight'):
        run_tree.GetEntry(0)
        nScaleWeights = run_tree.nLHEScaleSumw
        scale_weight_norm_buff = array('f',
                                       [sumwgts / _get_sum(run_tree, 'LHEScaleSumw[%d]*genEventSumw' % i)
                                        for i in range(nScaleWeights)])
        logging.info('LHEScaleWeightNorm: ' + str(scale_weight_norm_buff))
        _fill_const_branch(tree, 'LHEScaleWeightNorm', scale_weight_norm_buff, lenVar='nLHEScaleWeight')

    if tree.GetBranch('LHEPdfWeight'):
        run_tree.GetEntry(0)
        nPdfWeights = run_tree.nLHEPdfSumw
        pdf_weight_norm_buff = array('f',
                                     [sumwgts / _get_sum(run_tree, 'LHEPdfSumw[%d]*genEventSumw' % i)
                                      for i in range(nPdfWeights)])
        logging.info('LHEPdfWeightNorm: ' + str(pdf_weight_norm_buff))
        _fill_const_branch(tree, 'LHEPdfWeightNorm', pdf_weight_norm_buff, lenVar='nLHEPdfWeight')

    # fill PS weight re-normalization factors
    if tree.GetBranch('PSWeight') and run_tree.GetBranch('PSSumw'):
        run_tree.GetEntry(0)
        nPSWeights = run_tree.nPSSumw
        ps_weight_norm_buff = array('f',
                                    [sumwgts / _get_sum(run_tree, 'PSSumw[%d]*genEventSumw' % i)
                                     for i in range(nPSWeights)])
        logging.info('PSWeightNorm: ' + str(ps_weight_norm_buff))
        _fill_const_branch(tree, 'PSWeightNorm', ps_weight_norm_buff, lenVar='nPSWeight')

    tree.Write(treename, ROOT.TObject.kOverwrite)
    f.Close()


def parse_sample_xsec(cfgfile):
    xsec_dict = {}
    with open(cfgfile) as f:
        for l in f:
            l = l.strip()
            if not l or l.startswith('#'):
                continue
            pieces = l.split()
            samp = None
            xsec = None
            isData = False
            for s in pieces:
                if '/MINIAOD' in s or '/NANOAOD' in s:
                    samp = s.split('/')[1]
                    if '/MINIAODSIM' not in s and '/NANOAODSIM' not in s:
                        isData = True
                        break
                else:
                    try:
                        xsec = float(s)
                    except ValueError:
                        try:
                            import numexpr
                            xsec = numexpr.evaluate(s).item()
                        except:
                            pass
            if samp is None:
                logging.warning('Ignore line:\n%s' % l)
            elif not isData and xsec is None:
                logging.error('Cannot find cross section:\n%s' % l)
            else:
                if samp in xsec_dict and xsec_dict[samp] != xsec:
                    print(f"xsecdict {xsec_dict[samp]},xsec {xsec}")
                    raise RuntimeError('Inconsistent entries for sample %s' % samp)
                xsec_dict[samp] = xsec
                if 'PSweights_' in samp:
                    xsec_dict[samp.replace('PSweights_', '')] = xsec
    return xsec_dict

def main(args):
    xsec_dict = parse_sample_xsec(args.cfgfile)
    outfile = args.input
    samp = args.sample
    try:
        xsec = xsec_dict[samp]
        if xsec is not None:
            logging.info('Adding xsec weight to file %s, xsec=%f' % (outfile, xsec))
            add_weight_branch(outfile, xsec)
    except KeyError as e:
        if '-' not in samp and '_' not in samp:
            # data
            logging.info('Not adding weight to sample %s' % samp)
        else:
            raise e
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add cross section weight branch to NanoAOD file')
    parser.add_argument('--input', required=True, help='Input NanoAOD file')
    parser.add_argument('--sample', required=True, help='Sample name as in the xsec cfg file')
    parser.add_argument('--cfgfile', required=True, help='Cross section configuration file')
    args = parser.parse_args()
    main(args)