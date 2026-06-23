import os
import re
import numpy as np
import math
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from ..helpers.utils import deltaPhi, deltaR, deltaR2, deltaEta, closest, polarP4, sumP4, transverseMass, minValue, configLogger, getDigit, closest_pair
from ..helpers.ortHelper import ONNXRuntimeHelper
from ..helpers.nnHelper import convert_prob
from ..helpers.jetmetCorrector import JetMETCorrector, rndSeed
from ..helpers.muonCorrector import MuonScaleResCorrector
from ..helpers.triggerHelper import passTrigger

from .flavTagSFProducer import FlavTagSFProducer
from .leptonSFProducer import TriggerSF, ElectronSFProducer, MuonSFProducer
from .puWeightProducer import PileupWeightProducer
from .eventVetoMapProducer import EventVetoMapProducer
from .topPtWeightProducer import TopPtWeightProducer
from .topSystReweightingProducer import TopSystReweightingProducer
from .renormWeightSFProducer import RenormWeightSFProducer

import logging
logger = logging.getLogger('nano')
configLogger('nano', loglevel=logging.INFO)

lumi_dict = {2015: 19.52, 2016: 16.81, 2017: 41.48, 2018: 59.83}
dataset_dict = {
    'SingleMuon': 100001,
    'SingleElectron': 100002,
    'MuonEG': 100003,
    'DoubleMuon': 100004,
    'DoubleEG': 100005,
    'EGamma': 100006,
    'JetHT': 100007,
    'MET': 100008,
    'BtagCSV': 100009,
}

from .decotaKit import timeit, getentries

class METObject(Object):

    def p4(self):
        return polarP4(self, eta=None, mass=None)


class TTHTreeProducer(Module, object):

    def __init__(self, channel, **kwargs):
        self._channel = channel  # '0L', '1L', '2L', '0L_TrigSF', '1L_TrigSF', '2L_TrigSF'
        self._year = int(kwargs['year'])
        self._usePuppiJets = kwargs['usePuppiJets']
        self._jmeSysts = {'jec': False, 'jes': None, 'jes_source': '', 'jes_uncertainty_file_prefix': '',
                          'jer': 'nominal', 'jmr': None, 'met_unclustered': None, 'applyHEMUnc': False,
                          'smearMET': False}
        self._opts = {
            'min_num_ak4_jets_1L': 4,
            'min_num_ak8_jets_1L': 0,
            'min_num_b_jets': 1,
            'min_num_b_or_c_jets': 3,
            'min_ht_0L': 500,
            'apply_tight_selection': True,
            'apply_score_selection': True, 'apply_qcd_cut': 1e-4,
            'eval_nn': True, 'eval_nn_da_op1': False, 'eval_nn_da_op2': False, 'eval_mlp': False,
            'muon_scale': 'nominal',
            'fillJetTaggingScores': False, 'fillEventVars': True, 'fillBDTVars': False,
            'runModules': True, 'fillSystWeights': True, 'fillRenormWeights': True, 'fillExtendedPSWeights': True,
        }
        if kwargs.get('training_samples'):
            self._opts.update({
                'min_ht_0L': 300,
                'apply_tight_selection': False,
                'apply_score_selection': False, 'apply_qcd_cut': None,
                'eval_nn': False, 'eval_nn_da_op1': False, 'eval_nn_da_op2': False, 'eval_mlp': False,
                'fillEventVars': False,
                'runModules': False, 'fillSystWeights': False, 'fillRenormWeights': False, 'fillExtendedPSWeights': True,
            })
        # [XXX] default config for wcb analysis
        if kwargs.get('wcb_analysis'):
            self._opts.update({
                'min_num_ak4_jets_1L': 3,
                'min_num_ak8_jets_1L': 1,
                'min_num_b_jets': 1,
                'min_num_b_or_c_jets': 1,
                'apply_tight_selection': False,
                'apply_score_selection': False, 'apply_qcd_cut': None,
                'eval_nn': True, 'eval_nn_da_op1': False, 'eval_nn_da_op2': False, 'eval_mlp': False,
                'muon_scale': 'nominal',
                'fillJetTaggingScores': True, 'fillEventVars': True, 'fillBDTVars': False,
                'runModules': True, 'fillSystWeights': True, 'fillRenormWeights': True, 'fillExtendedPSWeights': True,
            })

        for k, v in kwargs.items():
            if k in self._jmeSysts:
                self._jmeSysts[k] = v
            else:
                self._opts[k] = v

        if self._channel in ('1L_TrigSF', '2L_TrigSF'):
            # no need to evaluate NNs for TrigSF samples
            self._opts['min_num_b_or_c_jets'] = 0
            self._opts['apply_tight_selection'] = False
            self._opts['apply_score_selection'] = False
            self._opts['apply_qcd_cut'] = None
            self._opts['eval_nn'] = False
            self._opts['eval_nn_da_op1'] = False
            self._opts['eval_nn_da_op2'] = False
            self._opts['eval_mlp'] = False
            self._opts['muon_scale'] = None

        self._needsJMECorr = any([self._jmeSysts['jec'], self._jmeSysts['jes'],
                                  self._jmeSysts['jer'], self._jmeSysts['jmr'],
                                  self._jmeSysts['met_unclustered'], self._jmeSysts['applyHEMUnc']])

        logger.info('Running %s channel for year %s with JME systematics %s, other options %s',
                    self._channel, str(self._year), str(self._jmeSysts), str(self._opts))

        if self._needsJMECorr:
            self.jetmetCorr = JetMETCorrector(
                year=self._year, jetType="AK4PFPuppi" if self._usePuppiJets else "AK4PFchs", **self._jmeSysts)

        if self._channel in ('0L', '1L', '2L'):
            self.trigSF = TriggerSF(self._year, self._channel)

        if self._opts['muon_scale']:
            self.muonCorr = MuonScaleResCorrector(year=self._year, corr=self._opts['muon_scale'])

        # ParticleNetAK4 -- exclusive b- and c-tagging categories
        # 5x: b-tagged; 4x: c-tagged; 0: light
        if self._year in (2017, 2018):
            self.jetTagWPs = {
                54: '(pn_b_plus_c>0.5) & (pn_b_vs_c>0.99)',
                53: '(pn_b_plus_c>0.5) & (0.96<pn_b_vs_c<=0.99)',
                52: '(pn_b_plus_c>0.5) & (0.88<pn_b_vs_c<=0.96)',
                51: '(pn_b_plus_c>0.5) & (0.70<pn_b_vs_c<=0.88)',
                50: '(pn_b_plus_c>0.5) & (0.40<pn_b_vs_c<=0.70)',

                44: '(pn_b_plus_c>0.5) & (pn_b_vs_c<=0.05)',
                43: '(pn_b_plus_c>0.5) & (0.05<pn_b_vs_c<=0.15)',
                42: '(pn_b_plus_c>0.5) & (0.15<pn_b_vs_c<=0.40)',
                41: '(0.2<pn_b_plus_c<=0.5)',
                40: '(0.1<pn_b_plus_c<=0.2)',

                0: '(pn_b_plus_c<=0.1)',
            }
        elif self._year in (2015, 2016):
            self.jetTagWPs = {
                54: '(pn_b_plus_c>0.35) & (pn_b_vs_c>0.99)',
                53: '(pn_b_plus_c>0.35) & (0.96<pn_b_vs_c<=0.99)',
                52: '(pn_b_plus_c>0.35) & (0.88<pn_b_vs_c<=0.96)',
                51: '(pn_b_plus_c>0.35) & (0.70<pn_b_vs_c<=0.88)',
                50: '(pn_b_plus_c>0.35) & (0.40<pn_b_vs_c<=0.70)',

                44: '(pn_b_plus_c>0.35) & (pn_b_vs_c<=0.05)',
                43: '(pn_b_plus_c>0.35) & (0.05<pn_b_vs_c<=0.15)',
                42: '(pn_b_plus_c>0.35) & (0.15<pn_b_vs_c<=0.40)',
                41: '(0.17<pn_b_plus_c<=0.35)',
                40: '(0.1<pn_b_plus_c<=0.17)',

                0: '(pn_b_plus_c<=0.1)',
            }

        if self._usePuppiJets:
            self.puID_WP = None
        else:
            # https://twiki.cern.ch/twiki/bin/view/CMS/PileupJetIDUL
            # NOTE [27.04.2022]: switched back to tight PU ID
            # self.puID_WP = {2015: 1, 2016: 1, 2017: 4, 2018: 4}[self._year]  # L
            # self.puID_WP = {2015: 3, 2016: 3, 2017: 6, 2018: 6}[self._year]  # M
            self.puID_WP = {2015: 7, 2016: 7, 2017: 7, 2018: 7}[self._year]  # T

        logger.info('Running %s channel for year %s with jet tagging WPs %s, jet PU ID WPs %s',
                    self._channel, str(self._year), str(self.jetTagWPs), str(self.puID_WP))

        if self._opts['eval_nn']:
            prefix = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoTTH/data')
            if '0L' in self._channel:
                self.nn_helper = ONNXRuntimeHelper(
                    preprocess_file='%s/nn/0L/v3/preprocess.json' % prefix,
                    model_files=['%s/nn/0L/v3/net.%d.onnx' % (prefix, idx) for idx in range(5)])
            elif '1L' in self._channel:
                self.nn_helper = ONNXRuntimeHelper(
                    preprocess_file='%s/nn/1L/v3/preprocess.json' % prefix,
                    model_files=['%s/nn/1L/v3/net.%d.onnx' % (prefix, idx) for idx in range(5)])
            elif '2L' in self._channel:
                self.nn_helper = ONNXRuntimeHelper(
                    preprocess_file='%s/nn/2L/v3/preprocess.json' % prefix,
                    model_files=['%s/nn/2L/v3/net.%d.onnx' % (prefix, idx) for idx in range(5)])

        if self._opts['eval_nn_da_op1']:
            prefix = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoTTH/data')
            if '0L' in self._channel:
                self.nn_da_op1_helper = ONNXRuntimeHelper(
                    preprocess_file='%s/nn_da_op1/0L/preprocess.json' % prefix,
                    model_files=['%s/nn_da_op1/0L/net.%d.onnx' % (prefix, idx) for idx in range(5)],
                    output_prefix='score_da_op1')
            elif '1L' in self._channel:
                self.nn_da_op1_helper = ONNXRuntimeHelper(
                    preprocess_file='%s/nn_da_op1/1L/preprocess.json' % prefix,
                    model_files=['%s/nn_da_op1/1L/net.%d.onnx' % (prefix, idx) for idx in range(5)],
                    output_prefix='score_da_op1')
            elif '2L' in self._channel:
                self.nn_da_op1_helper = ONNXRuntimeHelper(
                    preprocess_file='%s/nn_da_op1/2L/preprocess.json' % prefix,
                    model_files=['%s/nn_da_op1/2L/net.%d.onnx' % (prefix, idx) for idx in range(5)],
                    output_prefix='score_da_op1')

        if self._opts['eval_nn_da_op2']:
            prefix = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoTTH/data')
            if '0L' in self._channel:
                self.nn_da_op2_helper = ONNXRuntimeHelper(
                    preprocess_file='%s/nn_da_op2/0L/preprocess.json' % prefix,
                    model_files=['%s/nn_da_op2/0L/net.%d.onnx' % (prefix, idx) for idx in range(5)],
                    output_prefix='score_da_op2')
            elif '1L' in self._channel:
                self.nn_da_op2_helper = ONNXRuntimeHelper(
                    preprocess_file='%s/nn_da_op2/1L/preprocess.json' % prefix,
                    model_files=['%s/nn_da_op2/1L/net.%d.onnx' % (prefix, idx) for idx in range(5)],
                    output_prefix='score_da_op2')
            elif '2L' in self._channel:
                self.nn_da_op2_helper = ONNXRuntimeHelper(
                    preprocess_file='%s/nn_da_op2/2L/preprocess.json' % prefix,
                    model_files=['%s/nn_da_op2/2L/net.%d.onnx' % (prefix, idx) for idx in range(5)],
                    output_prefix='score_da_op2')

        if self._opts['eval_mlp']:
            prefix = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoTTH/data')
            if '0L' in self._channel:
                raise NotImplementedError
                self.mlp_helper = ONNXRuntimeHelper(
                    preprocess_file='%s/mlp/0L/v1/preprocess.json' % prefix,
                    model_files=['%s/mlp/0L/v1/net.%d.onnx' % (prefix, idx) for idx in range(5)],
                    output_prefix='mlp')
            elif '1L' in self._channel:
                self.mlp_helper = ONNXRuntimeHelper(
                    preprocess_file='%s/mlp/1L/v1/preprocess.json' % prefix,
                    model_files=['%s/mlp/1L/v1/net.%d.onnx' % (prefix, idx) for idx in range(5)],
                    output_prefix='mlp')
            elif '2L' in self._channel:
                self.mlp_helper = ONNXRuntimeHelper(
                    preprocess_file='%s/mlp/2L/v1/preprocess.json' % prefix,
                    model_files=['%s/mlp/2L/v1/net.%d.onnx' % (prefix, idx) for idx in range(5)],
                    output_prefix='mlp')

        # start with modules that should always run, regardless of data or MC
        self._modules = {
            'jetVetomapEventVeto': EventVetoMapProducer,
        }
        # then add those only needed for MC
        if self._opts['runModules']:
            self._modules.update({
                'flavTagSF': FlavTagSFProducer,
                'electronSF': ElectronSFProducer,
                'muonSF': MuonSFProducer,
                'puWeight': PileupWeightProducer,
                'topPtWeight': TopPtWeightProducer,
                'topSystReweighter': TopSystReweightingProducer,
                'renormWeighter': RenormWeightSFProducer,
            })
        for k, cls in self._modules.items():
            self._modules[k] = cls(
                self._year, fillSystWeights=self._opts['fillSystWeights'],
                usePuppiJets=self._opts['usePuppiJets'])

    def evalJetTag(self, j, default=0):
        for wp, expr in self.jetTagWPs.items():
            if eval(expr, j.__dict__):
                return wp
        return default

    def beginJob(self):
        if self._needsJMECorr:
            self.jetmetCorr.beginJob()
        for mod in self._modules.values():
            mod.beginJob()

    def endJob(self):
        for mod in self._modules.values():
            mod.endJob()

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        self.hasParticleNetAK4 = 'privateNano' if inputTree.GetBranch(
            'Jet_ParticleNetAK4_probb') else 'jmeNano' if inputTree.GetBranch('Jet_particleNetAK4_B') else None
        if not self.hasParticleNetAK4 and 'TrigSF' not in self._channel:
            raise RuntimeError('No ParticleNetAK4 scores in the input NanoAOD!')
        self.rho_branch_name = 'Rho_fixedGridRhoFastjetAll' if bool(
            inputTree.GetBranch('Rho_fixedGridRhoFastjetAll')) else 'fixedGridRhoFastjetAll'

        self.dataset = None
        r = re.search(('MC' if self.isMC else 'Data') + r'\/([a-zA-Z0-9_\-]+)\/', inputFile.GetName())
        if r:
            self.dataset = r.groups()[0]

        self.out = wrappedOutputTree

        # NOTE: branch names must start with a lower case letter
        # check keep_and_drop_output.txt
        self.out.branch("dataset", "I", title=', '.join([f'{k}={v}' for k, v in dataset_dict.items()]))
        self.out.branch("year", "I")
        self.out.branch("channel", "I")
        self.out.branch("lumiwgt", "F")

        self.out.branch("passmetfilters", "O")

        # triggers for 0L
        self.out.branch("passTrig0L", "O")
        self.out.branch("passTrig0L_ext", "O")

        # triggers for 1L
        self.out.branch("passTrigEl", "O")
        self.out.branch("passTrigMu", "O")
        self.out.branch("passTrigEl1", "O") # [DEBUG] for checking HLT  
        self.out.branch("passTrigEl2", "O") # [DEBUG]
        if self._channel == "0L_TrigSF":
            self.out.branch("passTrigMu_nIso", "O")

        # triggers for 2L
        self.out.branch("passTrigElEl", "O")
        self.out.branch("passTrigElMu", "O")
        self.out.branch("passTrigMuMu", "O")
        # extra single e/mu trigger for 2L
        self.out.branch("passTrig2L_extEl", "O")
        self.out.branch("passTrig2L_extMu", "O")

        # MET triggers for 2L_TrigSF
        if self._channel == '2L_TrigSF':
            self.out.branch("passTrigMET", "O")

        if self.isMC:
            self.out.branch("l1PreFiringWeight", "F", limitedPrecision=10)
            if self._opts['fillSystWeights']:
                self.out.branch("l1PreFiringWeightUp", "F", limitedPrecision=10)
                self.out.branch("l1PreFiringWeightDown", "F", limitedPrecision=10)

            self.out.branch("trigEffWeight", "F", limitedPrecision=10)
            if self._opts['fillSystWeights']:
                self.out.branch("trigEffWeightUp", "F", limitedPrecision=10)
                self.out.branch("trigEffWeightDown", "F", limitedPrecision=10)

        self.out.branch("met", "F")
        self.out.branch("met_phi", "F")

        # V boson
        self.out.branch("v_pt", "F", limitedPrecision=10)
        self.out.branch("v_eta", "F", limitedPrecision=10)
        self.out.branch("v_phi", "F", limitedPrecision=10)
        self.out.branch("v_mass", "F", limitedPrecision=10)

        # leptons
        self.out.branch("n_lep", "I")

        self.out.branch("lep1_pt", "F")
        self.out.branch("lep1_eta", "F")
        self.out.branch("lep1_phi", "F")
        self.out.branch("lep1_mass", "F")
        self.out.branch("lep1_etaSC", "F", limitedPrecision=10)
        self.out.branch("lep1_pdgId", "I")

        self.out.branch("lep2_pt", "F")
        self.out.branch("lep2_eta", "F")
        self.out.branch("lep2_phi", "F")
        self.out.branch("lep2_mass", "F")
        self.out.branch("lep2_etaSC", "F", limitedPrecision=10)
        self.out.branch("lep2_pdgId", "I")

        # event level
        self.out.branch("ht", "F", limitedPrecision=10)

        # ak4 jets
        self.out.branch("n_btag", "I")
        self.out.branch("n_ctag", "I")
        self.out.branch("n_btagM", "I")
        self.out.branch("n_btagT", "I")
        self.out.branch("n_ctagM", "I")
        self.out.branch("n_ctagT", "I")

        # self.out.branch("b_idx", "I", 10, lenVar="n_btag")
        # self.out.branch("c_idx", "I", 10, lenVar="n_ctag")
        # self.out.branch("l_idx", "I", 10, lenVar="n_l_ak4")

        self.out.branch("ak4_pt", "F", 20, lenVar="n_ak4")
        self.out.branch("ak4_eta", "F", 20, lenVar="n_ak4")
        self.out.branch("ak4_phi", "F", 20, lenVar="n_ak4")
        self.out.branch("ak4_mass", "F", 20, lenVar="n_ak4")
        self.out.branch("ak4_tag", "F", 20, lenVar="n_ak4")
        if self.isMC:
            self.out.branch("ak4_hflav", "I", 20, lenVar="n_ak4")

        if self._opts['fillJetTaggingScores']:
            self.out.branch("ak4_bdisc", "F", 20, lenVar="n_ak4")
            self.out.branch("ak4_cvbdisc", "F", 20, lenVar="n_ak4")
            self.out.branch("ak4_cvldisc", "F", 20, lenVar="n_ak4")
            if self.hasParticleNetAK4:
                self.out.branch("ak4_pn_b", "F", 20, lenVar="n_ak4")
                self.out.branch("ak4_pn_c", "F", 20, lenVar="n_ak4")
                self.out.branch("ak4_pn_uds", "F", 20, lenVar="n_ak4")
                self.out.branch("ak4_pn_g", "F", 20, lenVar="n_ak4")

        # ak8 jets
        self.out.branch("n_ak8", "I")
        self.out.branch("ak8_pt", "F", 10, lenVar="n_ak8")
        self.out.branch("ak8_eta", "F", 10, lenVar="n_ak8")
        self.out.branch("ak8_phi", "F", 10, lenVar="n_ak8")
        self.out.branch("ak8_sdmass", "F", 10, lenVar="n_ak8")
        self.out.branch("ak8_rawFactor", "F", 10, lenVar="n_ak8")
        self.out.branch("ak8_tau21", "F", 10, lenVar="n_ak8")
        self.out.branch("ak8_tau32", "F", 10, lenVar="n_ak8")
        self.out.branch("ak8_gpt_bb", "F", 10, lenVar="n_ak8")
        self.out.branch("ak8_gpt_cc", "F", 10, lenVar="n_ak8")
        self.out.branch("ak8_gpt_bc", "F", 10, lenVar="n_ak8")
        self.out.branch("ak8_gpt_qcd", "F", 10, lenVar="n_ak8")
        self.out.branch("ak8_gpt_bs", "F", 10, lenVar="n_ak8")
        self.out.branch("ak8_gpt_cs", "F", 10, lenVar="n_ak8")
        self.out.branch("ak8_gpt_qq", "F", 10, lenVar="n_ak8")
        self.out.branch("ak8_gpt_topbw", "F", 10, lenVar="n_ak8")
        # self.out.branch("ak8_gpt_bqq", "F", 10, lenVar="n_ak8")
        self.out.branch("ak8_gpt_topw", "F", 10, lenVar="n_ak8")

        self.out.branch("ak8_nConstituents", "I", 10, lenVar="n_ak8")

        if 'TrigSF' in self._channel:
            self.out.branch("n_PV", "I")
            self.out.branch("n_PV_good", "I")
            self.out.branch("rho_Calo", "F")
            self.out.branch("rho_ChargedPileUp", "F")

        if self._opts['fillEventVars']:
            self.out.branch("ht_b", "F", limitedPrecision=10)
            self.out.branch("ht_c", "F", limitedPrecision=10)
            self.out.branch("ht_bc", "F", limitedPrecision=10)
            self.out.branch("minDR_b", "F", limitedPrecision=10)
            self.out.branch("mass_minDR_b", "F", limitedPrecision=10)
            self.out.branch("maxMass_b", "F", limitedPrecision=10)
            self.out.branch("minDR_c", "F", limitedPrecision=10)
            self.out.branch("mass_minDR_c", "F", limitedPrecision=10)
            self.out.branch("maxMass_c", "F", limitedPrecision=10)
            self.out.branch("minDR_bc", "F", limitedPrecision=10)
            self.out.branch("mass_minDR_bc", "F", limitedPrecision=10)
            self.out.branch("maxMass_bc", "F", limitedPrecision=10)

        if self._opts['eval_nn']:
            for name in self.nn_helper.output_names:
                self.out.branch(name, "F")
                # self.out.branch(name.replace('score', 'cat'), "O")
            self.out.branch('nn_category', "I", title=','.join(
                [n.replace('score_', '%d=' % idx) for idx, n in enumerate(self.nn_helper.output_names)]))

        if self._opts['eval_nn_da_op1']:
            for name in self.nn_da_op1_helper.output_names:
                self.out.branch(name, "F")

        if self._opts['eval_nn_da_op2']:
            for name in self.nn_da_op2_helper.output_names:
                self.out.branch(name, "F")

        if self._opts['eval_mlp']:
            for name in self.mlp_helper.output_names:
                self.out.branch(name, "F")

        if self._opts['fillBDTVars']:
            self.out.branch("Sum_Htb_ak4", "F")
            self.out.branch("Sum_pt_ak4_l", "F")
            self.out.branch("d2_ak4", "F")

            self.out.branch("djavg_ak4", "F")
            self.out.branch("mjavg_ak4", "F")
            self.out.branch("dbavg_ak4", "F")
            self.out.branch("dbmin_ak4", "F")
            self.out.branch("d_b_avg2_ak4", "F")

            self.out.branch("deltaravebb_ak4", "F")
            self.out.branch("Clb_ak4", "F")
            self.out.branch("Clj_ak4", "F")

            self.out.branch("mbbave_ak4", "F")
            self.out.branch("mbbclosestH_ak4", "F")
            self.out.branch("mbbmindr_ak4", "F")
            self.out.branch("mbbmax_ak4", "F")
            self.out.branch("deltarminbb_ak4", "F")
            self.out.branch("sumptbbmindr_ak4", "F")
            self.out.branch("deltaetamaxbb_ak4", "F")

            self.out.branch("mblmindr_ak4", "F")
            self.out.branch("deltarminbl_ak4", "F")

            self.out.branch("deltaravebj_ak4", "F")
            self.out.branch("mbjmindr_ak4", "F")
            self.out.branch("deltaravejj_ak4", "F")
            self.out.branch("deltarminjj_ak4", "F")
            self.out.branch("sumptjjmindr_ak4", "F")
            self.out.branch("deltaetamaxjj_ak4", "F")
            self.out.branch("deltaRmaxjj_ak4", "F")
            self.out.branch("mjjmindr_ak4", "F")

            self.out.branch("mjjj_ak4", "F")
            self.out.branch("njjmH_ak4", "F")

        # gen matching
        # [TODO]: set gen match branches 
        self.out.branch("n_genTops", "I")
        self.out.branch("tt_category", "I")
        self.out.branch("higgs_decay", "I")
        self.out.branch("w_decay", "I")
        self.out.branch("z_decay", "I")
        
        # match efficiency branches[DEBUG]
        # self.out.branch("n_maskLevel", "I")
        # self.out.branch("n_allak8", "I")
        # self.out.branch("n_maskak8jet", "I", 30, lenVar="n_allak8")

        # [XXX] define gen match branches here
        if self.isMC:
            self.out.branch("ak8_type", "I", 10, lenVar="n_ak8")
            self.out.branch("ak8_n_b_in_jet", "I", 10, lenVar="n_ak8")
            self.out.branch("ak8_n_c_in_jet", "I", 10, lenVar="n_ak8")
            self.out.branch("ak8_n_in_jet", "I", 10, lenVar="n_ak8")
            self.out.branch("ak8_is_wbc", "I", 10, lenVar="n_ak8")

            self.out.branch("ak8_match_wqq", "I", 10, lenVar="n_ak8")
            self.out.branch("ak8_match_wcq", "I", 10, lenVar="n_ak8")
            self.out.branch("ak8_match_top_bq", "I", 10, lenVar="n_ak8")
            self.out.branch("ak8_match_top_bc", "I", 10, lenVar="n_ak8")
            self.out.branch("ak8_match_top_bqq", "I", 10, lenVar="n_ak8")
            self.out.branch("ak8_match_top_bcq", "I", 10, lenVar="n_ak8")
            self.out.branch("ak8_match_wqq_wcb", "I", 10, lenVar="n_ak8")
            self.out.branch("ak8_match_tbq_wcb", "I", 10, lenVar="n_ak8")
            self.out.branch("ak8_match_tbqq_wcb", "I", 10, lenVar="n_ak8")
            self.out.branch("ak8_match_non", "I", 10, lenVar="n_ak8")
            self.out.branch("ak8_match_qcd", "I", 10, lenVar="n_ak8")

            # self.out.branch("genInfo_Nb_top", "I")
            # self.out.branch("genInfo_Nc_W", "I")
            # self.out.branch("genInfo_Nb_W", "I")
            # self.out.branch("genInfo_Nextra_b", "I")
            # self.out.branch("genInfo_Nextra_c", "I")Flag_eeBadScFilter
            # self.out.branch("genInfo_Nextra_b_hadMult", "I")
            # self.out.branch("genInfo_Nextra_c_hadMult", "I")
            self.out.branch("genEventClassifier", "I")

            self.out.branch("h2bb", "O")
            self.out.branch("h2cc", "O")
            self.out.branch("h2tautau", "O")

            self.out.branch("genH_pt", "F", limitedPrecision=10)
            self.out.branch("genZ_pt", "F", limitedPrecision=10)
            self.out.branch("genW_pt", "F", limitedPrecision=10)

            self.out.branch("hdau_jetidx1", "I")
            self.out.branch("hdau_jetidx2", "I")
            self.out.branch("topb_jetidx1", "I")
            self.out.branch("topb_jetidx2", "I")

            self.out.branch("n_b_genjets", "I")
            self.out.branch("n_genjets", "I")
            self.out.branch("n_add_genjets", "I")
            self.out.branch("n_fromTop_genjets", "I")
            self.out.branch("n_fromW_genjets", "I")

            self.out.branch("n_b_genjets_fromW","I")
            self.out.branch("n_c_genjets_fromW", "I")
            self.out.branch("n_q_genjets_fromW", "I")

            self.out.branch("pdfSumWgt", "F")
            self.out.branch("pdfSumWgtWAlphaS", "F")

        for mod in self._modules.values():
            mod.beginFile(inputFile, outputFile, inputTree, wrappedOutputTree)

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        for mod in self._modules.values():
            mod.endFile(inputFile, outputFile, inputTree, wrappedOutputTree)

    def _correctJetAndMET(self, event):
        if self._needsJMECorr:
            rho = getattr(event, self.rho_branch_name)
            # correct AK4 jets and MET
            self.jetmetCorr.setSeed(rndSeed(event, event._allJets))
            self.jetmetCorr.correctJetAndMET(
                jets=event._allJets,
                lowPtJets=Collection(event, "CorrT1METJet"),
                met=event.met,
                rawMET=METObject(event, "RawPuppiMET") if self._usePuppiJets else METObject(event, "RawMET"),
                defaultMET=METObject(event, "PuppiMET") if self._usePuppiJets else METObject(event, "MET"),
                rho=rho, genjets=Collection(event, 'GenJet') if self.isMC else None,
                isMC=self.isMC, runNumber=event.run)
            event._allJets = sorted(event._allJets, key=lambda x: x.pt, reverse=True)  # sort by pt after updating

    #@timeit
    def _selectLeptons(self, event):
        # do lepton selection
        event.looseLeptons = []  # used for jet lepton cleaning & lepton counting

        electrons = Collection(event, "Electron")
        for el in electrons:
            el.etaSC = el.eta + el.deltaEtaSC
            # ttH(bb) analysis uses tight electron ID
            # if el.pt > 15 and abs(el.eta) < 2.4 and el.cutBased == 4:
            # NOTE: try mvaFall17V2Iso_WP90
            if 1.4442 <= abs(el.etaSC) <= 1.5560:
                continue
            if el.pt > 15 and abs(el.eta) < 2.4 and el.mvaFall17V2Iso_WP90:
                el._wp_ID = 'wp90iso'
                event.looseLeptons.append(el)

        muons = Collection(event, "Muon")
        for mu in muons:
            if self._opts['muon_scale']:
                self.muonCorr.correct(event, mu, self.isMC)
            if mu.pt > 15 and abs(mu.eta) < 2.4 and mu.tightId and mu.pfRelIso04_all < 0.25:
                mu._wp_ID = 'TightID'
                mu._wp_Iso = 'LooseRelIso'
                event.looseLeptons.append(mu)

        event.looseLeptons.sort(key=lambda x: x.pt, reverse=True)
    #@timeit
    def _preSelect(self, event):
        event.selectedLeptons = []  # used for reconstructing the top quarks
        if self._channel == '0L_TrigSF':
            return True
        elif self._channel == '0L':
            if len(event.looseLeptons) != 0:
                return False
        elif self._channel == '1L' or self._channel == '1L_TrigSF':
            NUM_LEP = 2 if self._channel == '1L_TrigSF' else 1
            if len(event.looseLeptons) != NUM_LEP:
                return False
            electrons = []
            muons = []
            for lep in event.looseLeptons:
                if abs(lep.pdgId) == 13:
                    # mu (26/29/26 GeV)
                    muPtCut = 29 if self._year == 2017 else 26
                    if lep.pt > muPtCut and lep.tightId and lep.pfRelIso04_all < 0.15:
                        lep._wp_Iso = 'TightRelIso'
                        event.selectedLeptons.append(lep)
                        muons.append(lep)
                else:
                    # ele (29/30/30 GeV)
                    ePtCut = 30 if self._year in (2017, 2018) else 29
                    # ttH(bb) analysis uses tight electron ID
                    # if lep.pt > ePtCut and lep.cutBased == 4:
                    # NOTE: try mvaFall17V2Iso_WP80
                    if lep.pt > ePtCut and lep.mvaFall17V2Iso_WP80:
                        lep._wp_ID = 'wp80iso'
                        event.selectedLeptons.append(lep)
                        electrons.append(lep)
            if len(event.selectedLeptons) != NUM_LEP:
                return False
            if self._channel == '1L_TrigSF':
                # =1 e + =1 mu
                if not (len(electrons) == 1 and len(muons) == 1):
                    return False
                # always put electron first in the list
                event.selectedLeptons = [electrons[0], muons[0]]
        elif self._channel == '2L' or self._channel == '2L_TrigSF':
            if len(event.looseLeptons) != 2:
                return False
            for lep in event.looseLeptons:
                event.selectedLeptons.append(lep)
            if len(event.selectedLeptons) != 2:
                return False
            if event.selectedLeptons[0].pt < 25:
                return False
            if event.selectedLeptons[0].pdgId * event.selectedLeptons[1].pdgId > 0:
                # keep only opposite-sign
                return False
            if event.selectedLeptons[0].pdgId + event.selectedLeptons[1].pdgId == 0:
                # if (opposite-sign) same-flavor: apply additional cuts
                Vboson = sumP4(event.selectedLeptons[0], event.selectedLeptons[1])
                if Vboson.M() < 20:
                    return False
                if Vboson.M() > 76 and Vboson.M() < 106:
                    return False

        return True
    
    #@timeit
    def _cleanObjects(self, event):
        event.ak4jets = []
        for j in event._allJets:
            if not (j.pt > 25 and abs(j.eta) < 2.4 and (j.jetId & 4)):
                # NOTE: ttH(bb) uses jets w/ pT > 30 GeV, loose PU Id
                # pt, eta, tightIdLepVeto, loose PU ID
                continue
            if not self._usePuppiJets and not (j.pt > 50 or j.puId >= self.puID_WP):
                # apply jet puId only for CHS jets
                continue
            if closest(j, event.looseLeptons)[1] < 0.4:
                continue
            j.btagDeepFlavC = j.btagDeepFlavB * j.btagDeepFlavCvB / (
                1 - j.btagDeepFlavCvB) if (j.btagDeepFlavCvB >= 0 and j.btagDeepFlavCvB < 1) else -1
            if self.hasParticleNetAK4 == 'privateNano':
                # attach ParticleNet scores
                j.pn_b = convert_prob(j, ['b', 'bb'], ['c', 'cc', 'uds', 'g'], 'ParticleNetAK4_prob')
                j.pn_c = convert_prob(j, ['c', 'cc'], ['b', 'bb', 'uds', 'g'], 'ParticleNetAK4_prob')
                j.pn_uds = convert_prob(j, 'uds', ['b', 'bb', 'c', 'cc', 'g'], 'ParticleNetAK4_prob')
                j.pn_g = convert_prob(j, 'g', ['b', 'bb', 'c', 'cc', 'uds'], 'ParticleNetAK4_prob')
                j.pn_b_plus_c = j.pn_b + j.pn_c
                j.pn_b_vs_c = j.pn_b / j.pn_b_plus_c
                j.tag = self.evalJetTag(j)
            elif self.hasParticleNetAK4 == 'jmeNano':
                # attach ParticleNet scores
                j.pn_b = j.particleNetAK4_B
                j.pn_c = j.particleNetAK4_B * j.particleNetAK4_CvsB / (
                    1 - j.particleNetAK4_CvsB) if (j.particleNetAK4_CvsB >= 0 and j.particleNetAK4_CvsB < 1) else -1
                j.pn_uds = np.clip(1 - j.pn_b - j.pn_c, 0, 1) * j.particleNetAK4_QvsG if (
                    j.particleNetAK4_QvsG >= 0 and j.particleNetAK4_QvsG < 1) else -1
                j.pn_g = np.clip(1 - j.pn_b - j.pn_c - j.pn_uds, 0, 1) if (
                    j.particleNetAK4_QvsG >= 0 and j.particleNetAK4_QvsG < 1) else -1
                j.pn_b_plus_c = j.pn_b + j.pn_c
                j.pn_b_vs_c = j.pn_b / j.pn_b_plus_c
                j.tag = self.evalJetTag(j)
            else:
                j.tag = 0
            event.ak4jets.append(j)

        event.ak4_b_or_c_jets = []
        event.ak4_b_jets = []
        event.ak4_c_jets = []

        for jet_idx, j in enumerate(event.ak4jets):
            j.idx = jet_idx
            if j.tag > 0:
                event.ak4_b_or_c_jets.append(j)
                if j.tag >= 50:
                    event.ak4_b_jets.append(j)
                if 40 <= j.tag < 50:
                    event.ak4_c_jets.append(j)

        # ak8 jets
        event.ak8jets = []
        for j in event._allFatJets:
            if not (j.pt > 200 and abs(j.eta) < 2.4 and (j.jetId & 2)):  #[DIFF] tt is 2
                continue
            if closest(j, event.looseLeptons)[1] < 0.8:
                continue
            # attach GloParT scores
            j.gpt_bb = j.inclParTMDV2_probHbb
            j.gpt_cc = j.inclParTMDV2_probHcc
            j.gpt_bc = j.inclParTMDV2_probHbc
            j.gpt_qcd = convert_prob(j, None, ['QCDbb', 'QCDb', 'QCDcc', 'QCDc', 'QCDothers'], 'inclParTMDV2_prob')
            j.gpt_topbw = convert_prob(j, None, ['TopbWcs', 'TopbWqq', 'TopbWq', 'TopbWs', 'TopbWc', 'TopbWev', 'TopbWmv', 'TopbWtauev', 'TopbWtauhv', 'TopbWtaumv'], 'inclParTMDV2_prob')
            j.gpt_topw = convert_prob(j, None, ['TopWcs', 'TopWqq', 'TopWev', 'TopWmv', 'TopWtauev', 'TopWtauhv', 'TopWtaumv'], 'inclParTMDV2_prob')
            j.gpt_bs = j.inclParTMDV2_probHbs
            j.gpt_cs = j.inclParTMDV2_probHcs
            j.gpt_qq = j.inclParTMDV2_probHqq
            # j.gpt_bqq = j.inclParTMDV2_probTopbWqq
            event.ak8jets.append(j)
    #@timeit
    def _selectEvent(self, event):
        # logger.debug('processing event %d' % event.event)
        event.Vboson = None
        if '0L' in self._channel:
            event.Vboson = event.met.p4()
        elif '1L' in self._channel:
            event.Vboson = polarP4(event.selectedLeptons[0]) + (event.met.p4())
        elif '2L' in self._channel:
            event.Vboson = sumP4(event.selectedLeptons[0], event.selectedLeptons[1])

        # channel specific selections
        if '0L' in self._channel:
            # >=6j, >=1b, >=? medium b or c (=6j used for validation)
            if len(event.ak4jets) < 6:
                return False
            if len(event.ak4_b_jets) < self._opts['min_num_b_jets']:
                return False
            if len(event.ak4_b_or_c_jets) < self._opts['min_num_b_or_c_jets']:
                return False
            if event.ak4jets[5].pt < 40:
                # min pT of the 6th jet
                return False
            if sum([j.pt for j in event.ak4jets]) < self._opts['min_ht_0L']:
                # HT > 500
                return False
            if self._opts['apply_tight_selection']:
                # TODO
                if len(event.ak4jets) < 7:
                    return False
                if sum(j.tag >= 51 for j in event.ak4_b_jets) < 1:
                    return False
                if sum(j.tag >= 51 for j in event.ak4_b_jets) + sum(j.tag >= 41 for j in event.ak4_c_jets) < 3:
                    return False
        elif self._channel == '1L':
            # >=4j, >=1b, >=? medium b or c (=4j used for validation)
            if len(event.ak4jets) < self._opts['min_num_ak4_jets_1L']:
                return False
            if len(event.ak8jets) < self._opts['min_num_ak8_jets_1L']:
                return False
            if len(event.ak4_b_jets) < self._opts['min_num_b_jets']:
                return False
            if len(event.ak4_b_or_c_jets) < self._opts['min_num_b_or_c_jets']:
                return False
            if event.met.pt < 20:
                return False
            if self._opts['apply_tight_selection']:
                if len(event.ak4jets) < 5:
                    return False
                if sum(j.tag >= 51 for j in event.ak4_b_jets) < 1:
                    return False
                if sum(j.tag >= 51 for j in event.ak4_b_jets) + sum(j.tag >= 41 for j in event.ak4_c_jets) < 3:
                    return False
        elif self._channel == '2L':
            # >=3j, >=1b, >=? medium b or c (=3j used for validation)
            if len(event.ak4jets) < 3:
                return False
            if len(event.ak4_b_jets) < self._opts['min_num_b_jets']:
                return False
            if len(event.ak4_b_or_c_jets) < self._opts['min_num_b_or_c_jets']:
                return False
            if event.met.pt < 20:
                return False
            if self._opts['apply_tight_selection']:
                if len(event.ak4jets) < 4:
                    return False
                if sum(j.tag >= 51 for j in event.ak4_b_jets) < 1:
                    return False
                if sum(j.tag >= 51 for j in event.ak4_b_jets) + sum(j.tag >= 41 for j in event.ak4_c_jets) < 3:
                    return False
        elif self._channel == '1L_TrigSF':
            if len(event.ak4jets) < 4:
                return False
        elif self._channel == '2L_TrigSF':
            pass

        # return True if passes selection
        return True

    #@timeit
    def _fillGenMatch(self, event):
        if not self.isMC:
            self.out.fillBranch("tt_category", -1)
            self.out.fillBranch("higgs_decay", -1)
            return
        
        # genNb_top = getDigit(event.genTtbarId, 2)
        # genNc_W = getDigit(event.genTtbarId, 4)
        # genNb_W = getDigit(event.genTtbarId, 3)  # for BSM models giving W -> b decays
        # genNextra_b = 0
        # genNextra_b = 0
        # genNextra_c = 0
        # genNextra_b_hadMult = 0
        # genNextra_c_hadMult = 0
        # if getDigit(event.genTtbarId, 1) == 5:
        #     genNextra_b = 2 if getDigit(event.genTtbarId, 0) > 2 else 1
        #     genNextra_b_hadMult = getDigit(event.genTtbarId, 0)
        # elif getDigit(event.genTtbarId, 1) == 4:
        #     genNextra_c = 2 if getDigit(event.genTtbarId, 0) > 2 else 1
        #     genNextra_c_hadMult = getDigit(event.genTtbarId, 0)

        # self.out.fillBranch("genInfo_Nb_top", genNb_top)
        # self.out.fillBranch("genInfo_Nc_W", genNc_W)
        # self.out.fillBranch("genInfo_Nb_W", genNb_W)
        # self.out.fillBranch("genInfo_Nextra_b", genNextra_b)
        # self.out.fillBranch("genInfo_Nextra_c", genNextra_c)
        # self.out.fillBranch("genInfo_Nextra_b_hadMult", genNextra_b_hadMult)
        # self.out.fillBranch("genInfo_Nextra_c_hadMult", genNextra_c_hadMult)
        # [TODO]: Need match to gen jets to count these properly

        # genparts --> genparts.dauIdx
        # for each genpart, dauIdx is a list of indices of its daughters in genparts
        try:                                                
            genparts = event.genparts
        except RuntimeError as e:
            genparts = Collection(event, "GenPart")
            for idx, gp in enumerate(genparts):
                if 'dauIdx' not in gp.__dict__:
                    gp.dauIdx = []
                if gp.genPartIdxMother >= 0:
                    mom = genparts[gp.genPartIdxMother]
                    if 'dauIdx' not in mom.__dict__:
                        mom.dauIdx = [idx]
                    else:
                        mom.dauIdx.append(idx)
            event.genparts = genparts

        # if genparts is hadronic decay (quark daughters > 0)
        def isHadronic(gp):
            if len(gp.dauIdx) == 0:
                logger.warning(f'Particle (pdgId={gp.pdgId}) has no daughters!')
            for idx in gp.dauIdx:
                if abs(genparts[idx].pdgId) < 6:
                    return True
            return False

        # get the final copy of a particle (same pdgId, no further decay)
        def getFinal(gp):
            for idx in gp.dauIdx:
                dau = genparts[idx]
                if dau.pdgId == gp.pdgId:
                    return getFinal(dau)
            return gp

        nGenTops = 0
        nGenWs = 0
        nGenZs = 0
        nGenHs = 0

        lepGenTops = []         # Top quarks: t -> W(lv) b
        hadGenTops = []         # Top quarks: t -> W(qq) b
        bFromTops = []          # B quarks: t -> W `b`
        partonsFromW = []       # Partons: W -> {q,l}
        hadGenWs = []           # W bosons: W -> q X
        hadGenZs = []           # Z bosons: Z -> q X
        hadGenHs = []           # Higgs bosons: H -> q X
        lepGenHs = []           # Higgs bosons: H -> l l

        for gp in genparts:
            if gp.statusFlags & (1 << 13) == 0:
                continue
            if abs(gp.pdgId) == 6:
                nGenTops += 1
                for idx in gp.dauIdx:
                    dau = genparts[idx]
                    if abs(dau.pdgId) == 24:
                        genW = getFinal(dau)
                        gp.genW = genW
                        if isHadronic(genW):
                            hadGenTops.append(gp)
                        else:
                            lepGenTops.append(gp)
                    elif abs(dau.pdgId) in (1, 3, 5):
                        gp.genB = dau
                        bFromTops.append(dau)
            elif abs(gp.pdgId) == 24:
                nGenWs += 1
                if isHadronic(gp):
                    hadGenWs.append(gp)         
                for idx in gp.dauIdx:
                    dau = genparts[idx]
                    if abs(dau.pdgId) in (1, 2, 3, 4, 5, 11, 13, 15):
                        partonsFromW.append(dau)
            elif abs(gp.pdgId) == 23:
                nGenZs += 1
                if isHadronic(gp):
                    hadGenZs.append(gp)
            elif abs(gp.pdgId) == 25:
                nGenHs += 1
                if isHadronic(gp):
                    hadGenHs.append(gp)
                else:
                    lepGenHs.append(gp)

        def get_daughters(parton):
            try:
                if abs(parton.pdgId) == 6:
                    return (parton.genB, genparts[parton.genW.dauIdx[0]], genparts[parton.genW.dauIdx[1]])
                elif abs(parton.pdgId) in (23, 24, 25):
                    return (genparts[parton.dauIdx[0]], genparts[parton.dauIdx[1]])
            except IndexError:
                logger.warning(f'Failed to get daughter for particle (pdgId={parton.pdgId})!')
                return tuple()

        event.hadGenTops = hadGenTops
        event.lepGenTops = lepGenTops
        event.bFromTops = bFromTops
        event.partonsFromW = partonsFromW
        event.hadGenWs = hadGenWs
        event.hadGenZs = hadGenZs
        event.hadGenHs = hadGenHs
        event.lepGenHs = lepGenHs

        event.genH = hadGenHs[0] if len(hadGenHs) else None
        # 0: no hadronic H; else: max pdgId of the daughters
        event.hdecay_ = abs(get_daughters(event.genH)[0].pdgId) if event.genH else 0

        event.genHlep = lepGenHs[0] if len(lepGenHs) else None
        # 0: no hadronic H; else: max pdgId of the daughters
        event.hdecaylep_ = abs(get_daughters(event.genHlep)[0].pdgId) if event.genHlep else 0

        event.genZ = hadGenZs[0] if len(hadGenZs) else None
        # 0: no hadronic Z; else: max pdgId of the daughters
        event.zdecay_ = abs(get_daughters(event.genZ)[0].pdgId) if event.genZ else 0

        event.genW = hadGenWs[0] if len(hadGenWs) else None
        # 0: no hadronic W; else: max pdgId of the daughters
        event.wdecay_ = max([abs(d.pdgId) for d in get_daughters(event.genW)], default=0) if event.genW else 0

        event.tt_category = getDigit(event.genTtbarId, 1)

        self.out.fillBranch("n_genTops", nGenTops)
        self.out.fillBranch("tt_category", event.tt_category)
        self.out.fillBranch("higgs_decay", event.hdecay_ if event.hdecay_ != 0 else event.hdecaylep_)
        self.out.fillBranch("w_decay", event.wdecay_)
        self.out.fillBranch("z_decay", event.zdecay_)

        self.out.fillBranch("h2bb", event.hdecay_ == 5)
        self.out.fillBranch("h2cc", event.hdecay_ == 4)
        self.out.fillBranch("h2tautau", event.hdecaylep_ == 15)

        self.out.fillBranch("genH_pt", event.genH.pt if event.genH else -1)
        self.out.fillBranch("genZ_pt", event.genZ.pt if event.genZ else -1)
        self.out.fillBranch("genW_pt", event.genW.pt if event.genW else -1)


        # jet-parton matching
        hdau_jetidx = [-1, -1]
        if event.genH:
            for idx, dau in enumerate(get_daughters(event.genH)):
                if idx >= 2:
                    break
                for j in event.ak4jets:
                    if deltaR(j, dau) < 0.4:
                        hdau_jetidx[idx] = j.idx
                        break
        self.out.fillBranch("hdau_jetidx1", hdau_jetidx[0])
        self.out.fillBranch("hdau_jetidx2", hdau_jetidx[1])

        top_bjetidx = [-1, -1]
        for idx, dau in enumerate(bFromTops):
            if idx >= 2:
                break
            for j in event.ak4jets:
                if deltaR(j, dau) < 0.4:
                    top_bjetidx[idx] = j.idx
                    break
        self.out.fillBranch("topb_jetidx1", top_bjetidx[0])
        self.out.fillBranch("topb_jetidx2", top_bjetidx[1])

        # https://github.com/cms-sw/cmssw/blob/master/TopQuarkAnalysis/TopTools/plugins/GenTtbarCategorizer.cc
        genEventClassifier_id = -99
        if nGenHs > 0:  # there is a gen level Higgs
            if event.hdecay_ == 4:
                genEventClassifier_id = 0  # higgs to cc
            elif event.hdecay_ == 5:
                genEventClassifier_id = 1  # higgs to bb
            elif event.hdecaylep_ == 15:
                genEventClassifier_id = 2  # higgs to tautau
        else:  # tt categories
            if getDigit(event.genTtbarId, 1) == 0:  # no HFs accompanying tt
                genEventClassifier_id = 3  # tt+LF
            if getDigit(event.genTtbarId, 1) == 4:  # c's accompanying tt
                if getDigit(event.genTtbarId, 0) == 1:  # tt+c
                    genEventClassifier_id = 4
                if getDigit(event.genTtbarId, 0) == 2:  # tt+2c, 1 c jet with 2 hadrons
                    genEventClassifier_id = 5
                if getDigit(event.genTtbarId, 0) > 2:  # tt+cc
                    genEventClassifier_id = 6
            if getDigit(event.genTtbarId, 1) == 5:  # b's accompanying tt
                if getDigit(event.genTtbarId, 0) == 1:  # tt+b
                    genEventClassifier_id = 7
                if getDigit(event.genTtbarId, 0) == 2:  # tt+2b, 1 b jet with 2 hadrons
                    genEventClassifier_id = 8
                if getDigit(event.genTtbarId, 0) > 2:  # tt+bb
                    genEventClassifier_id = 9

        self.out.fillBranch("genEventClassifier", genEventClassifier_id)

        try:
            genjets = event.genjets
        except RuntimeError as e:
            genjets = Collection(event, "GenJet")
            for jet_idx, j in enumerate(genjets):
                j.idx = jet_idx
            event.genjets = genjets

        # get the gen b jets from top decays
        bjetIdxsFromTop = []
        for idx, dau in enumerate(bFromTops):
            dau = getFinal(dau)
            for j in event.genjets:
                if deltaR(j, dau) < 0.4:
                    bjetIdxsFromTop.append(j.idx)
                    break

        # get the gen b jets from w decays
        bjetIdxsFromW = []
        for idx, dau in enumerate(partonsFromW):
            if abs(dau.pdgId) != 5:
                continue
            dau = getFinal(dau)
            for j in event.genjets:
                if deltaR(j, dau) < 0.4:
                    bjetIdxsFromW.append(j.idx)
                    break
        # get the gen c jets from w decays
        cjetIdxsFromW = []
        for idx, dau in enumerate(partonsFromW):
            if abs(dau.pdgId) != 4:
                continue
            dau = getFinal(dau)
            for j in event.genjets:
                if deltaR(j, dau) < 0.4:
                    cjetIdxsFromW.append(j.idx)
                    break
        # get the gen light jets from w decays
        qjetIdxsFromW = []
        for idx, dau in enumerate(partonsFromW):
            if abs(dau.pdgId) in (4, 5):
                continue
            dau = getFinal(dau)
            for j in event.genjets:
                if deltaR(j, dau) < 0.4:
                    qjetIdxsFromW.append(j.idx)
                    break

        # get the gen jets from w decays
        jetIdxsFromW = []
        for idx, dau in enumerate(partonsFromW):
            dau = getFinal(dau)
            for j in event.genjets:
                if deltaR(j, dau) < 0.4:
                    jetIdxsFromW.append(j.idx)
                    break

        n_genjets = 0
        n_fromTop_genjets = 0
        n_fromW_genjets = 0
        n_add_genjets = 0  # ones not from top decay
        n_b_genjets = 0  # inclusively, as same for all channels
        n_b_genjets_fromW = 0
        n_c_genjets_fromW = 0
        n_q_genjets_fromW = 0
        n_b_genjets_fromTop = 0
        n_c_genjets_fromTop = 0
        n_q_genjets_fromTop = 0
        for idx, genjet in enumerate(event.genjets):
            if genjet.pt > 20 and abs(genjet.eta) < 2.4:
                n_genjets += 1
                if abs(genjet.hadronFlavour) == 5:
                    n_b_genjets += 1
                if idx not in bjetIdxsFromTop and idx not in jetIdxsFromW:
                    n_add_genjets += 1
                if idx in bjetIdxsFromTop:
                    n_fromTop_genjets += 1
                if idx in jetIdxsFromW:
                    n_fromW_genjets += 1
                if idx in bjetIdxsFromW:
                    n_b_genjets_fromW += 1
                if idx in cjetIdxsFromW:
                    n_c_genjets_fromW += 1
                if idx in qjetIdxsFromW:
                    n_q_genjets_fromW += 1

        self.out.fillBranch("n_genjets", n_genjets)
        self.out.fillBranch("n_add_genjets", n_add_genjets)
        self.out.fillBranch("n_b_genjets", n_b_genjets)
        self.out.fillBranch("n_fromTop_genjets", n_fromTop_genjets)
        self.out.fillBranch("n_fromW_genjets", n_fromW_genjets)
        self.out.fillBranch("n_b_genjets_fromW", n_b_genjets_fromW)
        self.out.fillBranch("n_c_genjets_fromW", n_c_genjets_fromW)
        self.out.fillBranch("n_q_genjets_fromW", n_q_genjets_fromW)


        self.out.fillBranch("genEventClassifier", genEventClassifier_id)

        # fill pdf weights summed / we could shrink the trees even further...
        event.pdfSumWgt = 0.
        event.pdfSumWgtWAlphaS = 0.

        try:
            if event.nLHEPdfWeight > 101:  # default NNPDF 3.1 hessian set
                for iPDF in range(1, 101):
                    event.pdfSumWgt = event.pdfSumWgt + ((event.LHEPdfWeight[iPDF] / event.LHEPdfWeight[0] - 1.)**2.)
                for iPDF in range(1, 103):
                    event.pdfSumWgtWAlphaS = event.pdfSumWgtWAlphaS + (
                        (event.LHEPdfWeight[iPDF] / event.LHEPdfWeight[0] - 1.)**2.)
                event.pdfSumWgt = np.sqrt(event.pdfSumWgt)
                event.pdfSumWgtWAlphaS = np.sqrt(event.pdfSumWgt)
            else:  # replica set for ttbb
                mean_w = np.mean(np.array([event.LHEPdfWeight[iPDF] for iPDF in range(event.nLHEPdfWeight)]))
                for iPDF in range(event.nLHEPdfWeight):
                    event.pdfSumWgt = event.pdfSumWgt + ((event.LHEPdfWeight[iPDF] - mean_w)**2.)
                event.pdfSumWgt = np.sqrt(event.pdfSumWgt / (event.nLHEPdfWeight - 1))
                event.pdfSumWgtWAlphaS = event.pdfSumWgt
        except ZeroDivisionError:
            event.pdfSumWgt = 0.
            event.pdfSumWgtWAlphaS = 0.
        except RuntimeError: #when nLHEPdfWeight does not exist
            event.pdfSumWgt = 0.
            event.pdfSumWgtWAlphaS = 0.

        # if event.nLHEPdfWeight == 103:  # default NNPDF 3.1 hessian set
        #     for iPDF in range(1, 101):
        #         event.pdfSumWgt = event.pdfSumWgt + ((event.LHEPdfWeight[iPDF] / event.LHEPdfWeight[0] - 1.) ** 2)
        #     event.pdfSumWgtWAlphaS = event.pdfSumWgt + (
        #         (0.5 * abs(event.LHEPdfWeight[102] - event.LHEPdfWeight[101]) / event.LHEPdfWeight[0]) ** 2)
        #     event.pdfSumWgt = 1 + math.sqrt(event.pdfSumWgt)
        #     event.pdfSumWgtWAlphaS = 1 + math.sqrt(event.pdfSumWgtWAlphaS)
        # elif event.nLHEPdfWeight == 101:  # replica set for ttbb
        #     pdfwgts = np.array([event.LHEPdfWeight[iPDF] for iPDF in range(1, event.nLHEPdfWeight)])
        #     event.pdfSumWgt = 1 + np.std(pdfwgts) / event.LHEPdfWeight[0]
        #     event.pdfSumWgtWAlphaS = event.pdfSumWgt

        self.out.fillBranch("pdfSumWgt", event.pdfSumWgt)
        self.out.fillBranch("pdfSumWgtWAlphaS", event.pdfSumWgtWAlphaS)
        # self.out.fillBranch("n_maskLevel", event.maskLevel)  # [mask] fill mask level
        
    #@timeit
    def _fillAk8Match(self, event):
        if self.isMC:
            # genparts --> genparts.dauIdx
            # for each genpart, dauIdx is a list of indices of its daughters in genparts
            try:                                                
                genparts = event.genparts
                hadGenTops = event.hadGenTops
                hadGenWs = event.hadGenWs
            except RuntimeError as e:
                raise ValueError('hadGenTops or hadGenWs or genpart not found, please run `_fillGenMatch` first!')
            def getFinal(gp):
            # get the final copy of a particle (same pdgId, no further decay)
                for idx in gp.dauIdx:
                    dau = genparts[idx]
                    if dau.pdgId == gp.pdgId:
                        return getFinal(dau)
                return gp


            # initialize match variables in ak8 jets

            ak8jets = event.ak8jets       # for RECO level
            for jet_idx, j in enumerate(ak8jets):
                j.idx = jet_idx
                j.type = -1
                j.is_wbc = 0
                j.n_b_in_jet = 0
                j.n_c_in_jet = 0
                j.n_in_jet = 0
                j.match_wqq_wcb = 0
                j.match_tbq_wcb = 0
                j.match_tbqq_wcb = 0
                j.match_top_bq = 0
                j.match_top_bqq = 0
                j.match_top_bcq = 0
                j.match_top_bc = 0
                j.match_wqq = 0
                j.match_non = 0
                j.match_wcq = 0
                j.match_qcd = 0

            for j in ak8jets:
                n_b_in_jet , n_c_in_jet , n_in_jet = -1, -1, -1
                is_wbc = False
                is_w = True
                is_non = True
                for wboson in hadGenWs:
                    n_w_b_gen = sum(1 for idx in wboson.dauIdx if abs(genparts[idx].pdgId) == 5)
                    n_w_c_gen = sum(1 for idx in wboson.dauIdx if abs(genparts[idx].pdgId) == 4)
                    is_wbc = (n_w_b_gen == 1 and n_w_c_gen == 1) or is_wbc

                # for idtop, tquark in enumerate(hadGenTops):
                #     if deltaR(j, tquark.genB) < 0.8:
                #         n_in_jet = 0
                #         n_b_in_jet = 0
                #         n_c_in_jet = 0
                #         for idx in tquark.genW.dauIdx:
                #             dau = getFinal(genparts[idx])
                #             if deltaR(j, dau) < 0.8:
                #                 n_in_jet += 1
                #                 if abs(dau.pdgId) in (4, 5):
                #                     n_b_in_jet += abs(dau.pdgId) == 5
                #                     n_c_in_jet += abs(dau.pdgId) == 4
                #         if n_in_jet >= 1:
                #             is_w = False
                #             is_non = False
                # for idw, wboson in enumerate(hadGenWs):
                #     n_w_b_gen = sum(1 for idx in wboson.dauIdx if abs(genparts[idx].pdgId) == 5)
                #     n_w_c_gen = sum(1 for idx in wboson.dauIdx if abs(genparts[idx].pdgId) == 4)
                #     is_wbc = (n_w_b_gen == 1 and n_w_c_gen == 1) or is_wbc
                #     if is_w:
                #         n_in_jet = 0
                #         n_b_in_jet = 0
                #         n_c_in_jet = 0
                #         for idx in wboson.dauIdx:
                #             dau = getFinal(genparts[idx])
                #             if deltaR(j, dau) < 0.8:
                #                 n_in_jet += 1
                #                 if abs(dau.pdgId) in (4, 5):
                #                     n_b_in_jet += abs(dau.pdgId) == 5
                #                     n_c_in_jet += abs(dau.pdgId) == 4
                #         if n_in_jet >= 2:
                #             is_non = False

                best_top_idx = -1
                best_dr_b = 999.
                for idtop, tquark in enumerate(hadGenTops):
                    if not hasattr(tquark, "genB") or not hasattr(tquark, "genW"):
                        continue
                    drb = deltaR(j, tquark.genB)
                    if drb < 0.8 and drb < best_dr_b:
                        best_dr_b = drb
                        best_top_idx = idtop

                if best_top_idx >= 0:
                    tmp_n_in = 0
                    tmp_n_b = 0
                    tmp_n_c = 0
                    tquark = hadGenTops[best_top_idx]
                    for idx in tquark.genW.dauIdx:
                        dau = getFinal(genparts[idx])
                        if deltaR(j, dau) < 0.8:
                            tmp_n_in += 1
                            if abs(dau.pdgId) == 5:
                                tmp_n_b += 1
                            elif abs(dau.pdgId) == 4:
                                tmp_n_c += 1
                    if tmp_n_in >= 1:
                        is_w = False
                        is_non = False
                        n_in_jet = tmp_n_in
                        n_b_in_jet = tmp_n_b
                        n_c_in_jet = tmp_n_c

                # only if not top-like, try W-like
                # choose the W with the largest daughter overlap
                if is_w:
                    best_w_idx = -1
                    best_w_n_in = -1
                    best_w_n_b = -1
                    best_w_n_c = -1

                    for idw, wboson in enumerate(hadGenWs):
                        tmp_n_in = 0
                        tmp_n_b = 0
                        tmp_n_c = 0
                        for idx in wboson.dauIdx:
                            dau = getFinal(genparts[idx])
                            if deltaR(j, dau) < 0.8:
                                tmp_n_in += 1
                                if abs(dau.pdgId) == 5:
                                    tmp_n_b += 1
                                elif abs(dau.pdgId) == 4:
                                    tmp_n_c += 1

                        if tmp_n_in > best_w_n_in:
                            best_w_idx = idw
                            best_w_n_in = tmp_n_in
                            best_w_n_b = tmp_n_b
                            best_w_n_c = tmp_n_c

                    if best_w_idx >= 0 and best_w_n_in >= 2:
                        is_non = False
                        n_in_jet = best_w_n_in
                        n_b_in_jet = best_w_n_b
                        n_c_in_jet = best_w_n_c


                # topo: w(xx) --> 1, t(bx) --> 2, t(bxx) --> 4
                j.type = 0 if is_non else 1 if is_w else 2 if n_in_jet == 1 else 4 if n_in_jet == 2 else -1
                if (len(hadGenTops) == 0) and (len(hadGenWs) == 0):
                    j.type = -1  # no hadronic top or w in the events

                # flavor: b, c, light
                j.n_b_in_jet = n_b_in_jet
                j.n_c_in_jet = n_c_in_jet
                j.n_in_jet = n_in_jet
                j.is_wbc = 1 if is_wbc else 0
                                
                # identify jet type 
                if j.type == 0:
                    j.match_non = 1
                elif j.type == 1:
                    if j.n_b_in_jet + j.n_c_in_jet == 0:
                        j.match_wqq = 1
                    elif j.n_c_in_jet == 1 and j.n_b_in_jet == 0:
                        j.match_wcq = 1
                    elif j.n_b_in_jet == 1 and j.n_c_in_jet == 1:
                        j.match_wqq_wcb = 1
                    else:
                        j.match_non = -1
                elif j.type == 2:
                    if is_wbc and j.n_b_in_jet + j.n_c_in_jet == 1:
                        j.match_tbq_wcb = 1
                    elif j.n_c_in_jet == 1 and j.n_b_in_jet == 0:
                        j.match_top_bc = 1
                    elif j.n_in_jet - j.n_b_in_jet - j.n_c_in_jet == 1 :
                        j.match_top_bq = 1
                    else:
                        j.match_non = -1
                elif j.type == 4:
                    if is_wbc and j.n_b_in_jet == 1 and j.n_c_in_jet == 1:
                        j.match_tbqq_wcb = 1
                    elif j.n_c_in_jet == 1 and j.n_b_in_jet == 0:
                        j.match_top_bcq = 1
                    elif j.n_in_jet - j.n_b_in_jet - j.n_c_in_jet == 2:
                        j.match_top_bqq = 1
                    else:
                        j.match_non = -1
                elif j.type == -1:
                    j.match_qcd = 1
                else:
                    j.match_non = -1  # undefined case
            self.out.fillBranch("ak8_type", [j.type for j in ak8jets])
            self.out.fillBranch("ak8_n_b_in_jet", [j.n_b_in_jet for j in ak8jets])
            self.out.fillBranch("ak8_n_c_in_jet", [j.n_c_in_jet for j in ak8jets])
            self.out.fillBranch("ak8_n_in_jet", [j.n_in_jet for j in ak8jets])
            self.out.fillBranch("ak8_is_wbc", [j.is_wbc for j in ak8jets])

            self.out.fillBranch("ak8_match_wqq", [j.match_wqq for j in ak8jets])
            self.out.fillBranch("ak8_match_wcq", [j.match_wcq for j in ak8jets])
            self.out.fillBranch("ak8_match_top_bq", [j.match_top_bq for j in ak8jets])
            self.out.fillBranch("ak8_match_top_bc", [j.match_top_bc for j in ak8jets])
            self.out.fillBranch("ak8_match_top_bqq", [j.match_top_bqq for j in ak8jets])
            self.out.fillBranch("ak8_match_non", [j.match_non for j in ak8jets])
            self.out.fillBranch("ak8_match_top_bcq", [j.match_top_bcq for j in ak8jets])
            self.out.fillBranch("ak8_match_wqq_wcb", [j.match_wqq_wcb for j in ak8jets])
            self.out.fillBranch("ak8_match_tbq_wcb", [j.match_tbq_wcb for j in ak8jets])
            self.out.fillBranch("ak8_match_tbqq_wcb", [j.match_tbqq_wcb for j in ak8jets])
            self.out.fillBranch("ak8_match_qcd", [j.match_qcd for j in ak8jets])

            
    #@timeit
    def _selectTriggers(self, event):
        out_data = {}

        # !!! NOTE: make sure to update `keep_and_drop_input.txt` !!!
        if self._year <= 2016:
            out_data["passTrig0L"] = passTrigger(event, [
                'HLT_PFHT450_SixJet40_BTagCSV_p056',
                'HLT_PFHT400_SixJet30_DoubleBTagCSV_p056',
                'HLT_PFJet450',
                'HLT_PFHT900',  # extra
            ])
            out_data["passTrig0L_ext"] = False
            # out_data["passTrig0L_ext"] = passTrigger(event, [
            #     'HLT_DoubleJet90_Double30_TripleBTagCSV_p087',  # BTagCSV
            #     'HLT_QuadJet45_TripleBTagCSV_p087',  # BTagCSV
            # ])
            out_data["passTrigEl"] = passTrigger(event, 'HLT_Ele27_WPTight_Gsf')
            out_data["passTrigMu"] = passTrigger(event, ['HLT_IsoMu24', 'HLT_IsoTkMu24'])
            out_data["passTrigElEl"] = passTrigger(event, 'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ')
            out_data["passTrigElMu"] = passTrigger(event,
                                                   ['HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
                                                    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
                                                    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL',
                                                    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ'])
            out_data["passTrigMuMu"] = (
                passTrigger(event, ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL', 'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL'])
                if event.run <= 280385 else  # Run2016G
                passTrigger(event, ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ', 'HLT_Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ'])
            )
            out_data["passTrig2L_extEl"] = passTrigger(event, 'HLT_Ele27_WPTight_Gsf')
            out_data["passTrig2L_extMu"] = passTrigger(event, ['HLT_IsoMu24', 'HLT_IsoTkMu24'])

            if self._channel == "0L_TrigSF":
                out_data["passTrigMu_nIso"] = passTrigger(event, ['HLT_Mu50', 'HLT_TkMu50'])
            elif self._channel == '2L_TrigSF':
                out_data["passTrigMET"] = passTrigger(event, [
                    'HLT_PFMET120_PFMHT120_IDTight',
                    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight',
                    'HLT_MET200',
                    'HLT_PFMET300',
                    'HLT_PFMET170_HBHECleaned',
                    # 'HLT_PFHT300_PFMET110',  # ? HTMHT dataset
                ])

        elif self._year == 2017:
            out_data["passTrig0L"] = passTrigger(event, [
                'HLT_PFHT430_SixJet40_BTagCSV_p080',  # RunB
                'HLT_PFHT430_SixPFJet40_PFBTagCSV_1p5',  # RunC-F
                'HLT_PFHT380_SixJet32_DoubleBTagCSV_p075',  # RunB
                'HLT_PFHT380_SixPFJet32_DoublePFBTagCSV_2p2',  # RunC-F
                'HLT_PFHT1050'])
            out_data["passTrig0L_ext"] = passTrigger(event, [
                'HLT_HT300PT30_QuadJet_75_60_45_40_TripeCSV_p07',  # RunB, BTagCSV
                'HLT_PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0',  # RunC-F, BTagCSV
            ])

            flagL1DoubleEG = False
            for obj in Collection(event, "TrigObj"):
                if (obj.id == 11) and (obj.filterBits & 1024):
                    # 1024 = 1e (32_L1DoubleEG_AND_L1SingleEGOr)
                    flagL1DoubleEG = True
                    break
            event.HLT_Ele32_WPTight_Gsf_L1DoubleEG_L1Flag = event.HLT_Ele32_WPTight_Gsf_L1DoubleEG and flagL1DoubleEG

            out_data["passTrigEl"] = passTrigger(
                event, ['HLT_Ele32_WPTight_Gsf_L1DoubleEG_L1Flag', 'HLT_Ele28_eta2p1_WPTight_Gsf_HT150'])
            out_data["passTrigEl1"] = passTrigger(event, 'HLT_Ele32_WPTight_Gsf_L1DoubleEG_L1Flag')     #[DEBUG]
            out_data["passTrigEl2"] = passTrigger(event, 'HLT_Ele28_eta2p1_WPTight_Gsf_HT150')          #[DEBUG]
            out_data["passTrigMu"] = passTrigger(event, 'HLT_IsoMu27')
            out_data["passTrigElEl"] = passTrigger(event, ['HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL',
                                                           'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ'])
            out_data["passTrigElMu"] = passTrigger(event,
                                                   ['HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
                                                    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
                                                    'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ',
                                                    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ'])
            out_data["passTrigMuMu"] = (
                passTrigger(event, 'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ')
                if event.run <= 299329 else  # Run2017B
                passTrigger(event, 'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8'))
            out_data["passTrig2L_extEl"] = passTrigger(event, 'HLT_Ele32_WPTight_Gsf_L1DoubleEG_L1Flag')
            out_data["passTrig2L_extMu"] = passTrigger(event, ['HLT_IsoMu24_eta2p1', 'HLT_IsoMu27'])

            if self._channel == "0L_TrigSF":
                out_data["passTrigMu_nIso"] = passTrigger(event, ['HLT_Mu50', 'HLT_OldMu100', 'HLT_TkMu100'])
            elif self._channel == '2L_TrigSF':
                out_data["passTrigMET"] = passTrigger(event, [
                    # 'HLT_PFMET250_HBHECleaned',  # only present for 36.75fb-1
                    'HLT_PFMET120_PFMHT120_IDTight',
                    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight',
                    'HLT_PFHT500_PFMET100_PFMHT100_IDTight',
                    'HLT_PFHT700_PFMET85_PFMHT85_IDTight',
                    'HLT_PFHT800_PFMET75_PFMHT75_IDTight',
                ])

        elif self._year == 2018:
            out_data["passTrig0L"] = passTrigger(event, [
                'HLT_PFHT430_SixPFJet40_PFBTagCSV_1p5',  # Runs 315252-315974
                'HLT_PFHT430_SixPFJet40_PFBTagDeepCSV_1p5',  # Runs 315974-317509
                'HLT_PFHT450_SixPFJet36_PFBTagDeepCSV_1p59',  # Runs 317509-end, MC
                'HLT_PFHT380_SixPFJet32_DoublePFBTagDeepCSV_2p2',  # Runs 315252-317509
                'HLT_PFHT400_SixPFJet32_DoublePFBTagDeepCSV_2p94',  # Runs 317509-end, MC
                'HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5',
                'HLT_PFHT1050'])
            out_data["passTrig0L_ext"] = False
            out_data["passTrigEl"] = passTrigger(event,
                                                 ['HLT_Ele32_WPTight_Gsf',
                                                  'HLT_Ele28_eta2p1_WPTight_Gsf_HT150'])
            out_data["passTrigMu"] = passTrigger(event, 'HLT_IsoMu24')
            out_data["passTrigElEl"] = passTrigger(event,
                                                   ['HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL',
                                                    'HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ'])
            out_data["passTrigElMu"] = passTrigger(event,
                                                   ['HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL',
                                                    'HLT_Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ',
                                                    'HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ',
                                                    'HLT_Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ'])
            out_data["passTrigMuMu"] = passTrigger(event,
                                                   ['HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8',
                                                    'HLT_Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8'])
            out_data["passTrig2L_extEl"] = passTrigger(event, 'HLT_Ele32_WPTight_Gsf')
            out_data["passTrig2L_extMu"] = passTrigger(event, 'HLT_IsoMu24')

            if self._channel == "0L_TrigSF":
                out_data["passTrigMu_nIso"] = passTrigger(event, ['HLT_Mu50', 'HLT_OldMu100', 'HLT_TkMu100'])
            elif self._channel == '2L_TrigSF':
                out_data["passTrigMET"] = passTrigger(event, [
                    'HLT_PFMET120_PFMHT120_IDTight',
                    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight',
                    'HLT_PFMET120_PFMHT120_IDTight_PFHT60',
                    'HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60',
                    'HLT_PFMET200_HBHE_BeamHaloCleaned',
                    'HLT_PFMETTypeOne200_HBHE_BeamHaloCleaned',
                    'HLT_PFHT500_PFMET100_PFMHT100_IDTight',
                    'HLT_PFHT700_PFMET85_PFMHT85_IDTight',
                    'HLT_PFHT800_PFMET75_PFMHT75_IDTight',
                ])

        # apply trigger selections on data
        if not self.isMC and self.dataset is not None and 'TrigSF' not in self._channel:
            if self._channel == '0L':
                passTrig0L = False
                if self.dataset == 'JetHT':
                    passTrig0L = out_data['passTrig0L']
                elif self.dataset == 'BTagCSV':
                    passTrig0L = (not out_data['passTrig0L']) and out_data['passTrig0L_ext']
                if not passTrig0L:
                    return False

            elif self._channel == '1L':
                passTrig1L = False
                if self.dataset in ('EGamma', 'SingleElectron'):
                    passTrig1L = out_data['passTrigEl']
                elif self.dataset == 'SingleMuon':
                    passTrig1L = out_data['passTrigMu']
                if not passTrig1L:
                    return False

            elif self._channel == '2L':
                passTrig2L = False
                if abs(event.selectedLeptons[0].pdgId) == 11 and abs(event.selectedLeptons[1].pdgId) == 11:
                    # ee channel
                    if self._year == 2018:
                        if self.dataset == 'EGamma':
                            passTrig2L = out_data["passTrigElEl"] or out_data['passTrig2L_extEl']
                    else:
                        if self.dataset == 'DoubleEG':
                            passTrig2L = out_data["passTrigElEl"]
                        elif self.dataset == 'SingleElectron':
                            passTrig2L = (not out_data["passTrigElEl"]) and out_data["passTrig2L_extEl"]
                elif abs(event.selectedLeptons[0].pdgId) == 13 and abs(event.selectedLeptons[1].pdgId) == 13:
                    # mumu channel
                    if self.dataset == 'DoubleMuon':
                        passTrig2L = out_data["passTrigMuMu"]
                    elif self.dataset == 'SingleMuon':
                        passTrig2L = (not out_data["passTrigMuMu"]) and out_data["passTrig2L_extMu"]
                else:
                    # emu channel
                    if self.dataset == 'MuonEG':
                        passTrig2L = out_data["passTrigElMu"]
                    elif self.dataset == 'SingleMuon':
                        passTrig2L = (not out_data["passTrigElMu"]) and out_data["passTrig2L_extMu"]
                    elif self.dataset in ('EGamma', 'SingleElectron'):
                        passTrig2L = (not out_data["passTrigElMu"]) and (
                            not out_data["passTrig2L_extMu"]) and out_data["passTrig2L_extEl"]

                if not passTrig2L:
                    return False

        for key in out_data:
            self.out.fillBranch(key, out_data[key])
        return True

    def _fillEventInfo(self, event):
        out_data = {}

        out_data["dataset"] = dataset_dict[self.dataset] if self.dataset in dataset_dict else 1 if self.dataset else 0
        out_data["year"] = self._year
        out_data["channel"] = int(self._channel[0])
        out_data["lumiwgt"] = lumi_dict[self._year]

        # met filters -- updated for UL
        met_filters = bool(
            event.Flag_goodVertices and
            event.Flag_globalSuperTightHalo2016Filter and
            event.Flag_HBHENoiseFilter and
            event.Flag_HBHENoiseIsoFilter and
            event.Flag_EcalDeadCellTriggerPrimitiveFilter and
            event.Flag_BadPFMuonFilter and
            event.Flag_BadPFMuonDzFilter and
            event.Flag_eeBadScFilter
        )
        if self._year in (2017, 2018):
            met_filters = met_filters and event.Flag_ecalBadCalibFilter
        out_data["passmetfilters"] = met_filters

        if self.isMC:
            # L1 prefire weights
            out_data["l1PreFiringWeight"] = event.L1PreFiringWeight_Nom
            if self._opts['fillSystWeights']:
                out_data["l1PreFiringWeightUp"] = event.L1PreFiringWeight_Up
                out_data["l1PreFiringWeightDown"] = event.L1PreFiringWeight_Dn

            # trigger SFs
            if self._channel in ('0L', '1L', '2L'):
                trigWgt = self.trigSF.get_trigger_sf(event)
                out_data["trigEffWeight"] = trigWgt[0]
                if self._opts['fillSystWeights']:
                    out_data["trigEffWeightUp"] = trigWgt[1]
                    out_data["trigEffWeightDown"] = trigWgt[2]

        # met
        out_data["met"] = event.met.pt
        out_data["met_phi"] = event.met.phi

        # V boson
        out_data["v_pt"] = event.Vboson.Pt()
        out_data["v_eta"] = event.Vboson.Eta()
        out_data["v_phi"] = event.Vboson.Phi()
        out_data["v_mass"] = event.Vboson.M()

        # leptons
        out_data["n_lep"] = len(event.looseLeptons)
        if len(event.selectedLeptons) > 0:
            out_data["lep1_pt"] = event.selectedLeptons[0].pt
            out_data["lep1_eta"] = event.selectedLeptons[0].eta
            out_data["lep1_etaSC"] = event.selectedLeptons[0].etaSC if abs(
                event.selectedLeptons[0].pdgId) == 11 else -999
            out_data["lep1_phi"] = event.selectedLeptons[0].phi
            out_data["lep1_mass"] = event.selectedLeptons[0].mass
            out_data["lep1_pdgId"] = event.selectedLeptons[0].pdgId

        if len(event.selectedLeptons) > 1:
            out_data["lep2_pt"] = event.selectedLeptons[1].pt
            out_data["lep2_eta"] = event.selectedLeptons[1].eta
            out_data["lep2_etaSC"] = event.selectedLeptons[1].etaSC if abs(
                event.selectedLeptons[1].pdgId) == 11 else -999
            out_data["lep2_phi"] = event.selectedLeptons[1].phi
            out_data["lep2_mass"] = event.selectedLeptons[1].mass
            out_data["lep2_pdgId"] = event.selectedLeptons[1].pdgId

        # event level
        out_data["ht"] = sum([j.pt for j in event.ak4jets])

        # AK4 jets, cleaned vs leptons
        out_data["n_btag"] = len(event.ak4_b_jets)
        out_data["n_ctag"] = len(event.ak4_c_jets)

        out_data["n_btagM"] = sum(j.tag >= 51 for j in event.ak4_b_jets)
        out_data["n_btagT"] = sum(j.tag >= 52 for j in event.ak4_b_jets)

        out_data["n_ctagM"] = sum(j.tag >= 41 for j in event.ak4_c_jets)
        out_data["n_ctagT"] = sum(j.tag >= 42 for j in event.ak4_c_jets)

        ak4_pt = []
        ak4_eta = []
        ak4_phi = []
        ak4_mass = []
        ak4_tag = []
        ak4_hflav = []
        ak4_bdisc = []
        ak4_cvbdisc = []
        ak4_cvldisc = []
        ak4_pn_b = []
        ak4_pn_c = []
        ak4_pn_uds = []
        ak4_pn_g = []

        for j in event.ak4jets:
            ak4_pt.append(j.pt)
            ak4_eta.append(j.eta)
            ak4_phi.append(j.phi)
            ak4_mass.append(j.mass)
            ak4_tag.append(j.tag)
            if self.isMC:
                ak4_hflav.append(j.hadronFlavour)

            if self._opts['fillJetTaggingScores']:
                ak4_bdisc.append(j.btagDeepFlavB)
                ak4_cvbdisc.append(j.btagDeepFlavCvB)
                ak4_cvldisc.append(j.btagDeepFlavCvL)
                if self.hasParticleNetAK4:
                    ak4_pn_b.append(j.pn_b)
                    ak4_pn_c.append(j.pn_c)
                    ak4_pn_uds.append(j.pn_uds)
                    ak4_pn_g.append(j.pn_g)

        out_data["ak4_pt"] = ak4_pt
        out_data["ak4_eta"] = ak4_eta
        out_data["ak4_phi"] = ak4_phi
        out_data["ak4_mass"] = ak4_mass
        out_data["ak4_tag"] = ak4_tag
        if self.isMC:
            out_data["ak4_hflav"] = ak4_hflav

        if self._opts['fillJetTaggingScores']:
            out_data["ak4_bdisc"] = ak4_bdisc
            out_data["ak4_cvbdisc"] = ak4_cvbdisc
            out_data["ak4_cvldisc"] = ak4_cvldisc
            if self.hasParticleNetAK4:
                out_data["ak4_pn_b"] = ak4_pn_b
                out_data["ak4_pn_c"] = ak4_pn_c
                out_data["ak4_pn_uds"] = ak4_pn_uds
                out_data["ak4_pn_g"] = ak4_pn_g

        # out_data["b_idx"] = [j.idx for j in event.ak4_b_jets]
        # out_data["c_idx"] = [j.idx for j in event.ak4_c_jets]
        # out_data["l_idx"] = [j.idx for j in event.ak4jets if j.tag == 0]

        # ak8 jets
        ak8_pt = []
        ak8_eta = []
        ak8_phi = []
        ak8_sdmass = []
        ak8_rawFactor = []
        ak8_tau21 = []
        ak8_tau32 = []
        ak8_gpt_bb = []
        ak8_gpt_cc = []
        ak8_gpt_bc = []
        ak8_gpt_qcd = []
        ak8_gpt_topbw = []
        ak8_gpt_topw = []
        ak8_gpt_bs = []
        ak8_gpt_cs = []
        ak8_gpt_qq = []
        ak8_nConstituents = []

        for j in event.ak8jets:
            ak8_pt.append(j.pt)
            ak8_eta.append(j.eta)
            ak8_phi.append(j.phi)
            ak8_sdmass.append(j.msoftdrop)
            ak8_rawFactor.append(j.rawFactor)
            ak8_tau21.append(j.tau2 / j.tau1 if j.tau1 > 0 else 99)
            ak8_tau32.append(j.tau3 / j.tau2 if j.tau2 > 0 else 99)
            ak8_gpt_bb.append(j.gpt_bb)
            ak8_gpt_cc.append(j.gpt_cc)
            ak8_gpt_bc.append(j.gpt_bc)
            ak8_gpt_qcd.append(j.gpt_qcd)
            ak8_gpt_topbw.append(j.gpt_topbw)
            ak8_gpt_topw.append(j.gpt_topw)
            ak8_gpt_bs.append(j.gpt_bs)
            ak8_gpt_cs.append(j.gpt_cs)
            ak8_gpt_qq.append(j.gpt_qq)
            ak8_nConstituents.append(j.nConstituents)


        out_data["n_ak8"] = len(event.ak8jets)
        out_data["ak8_pt"] = ak8_pt
        out_data["ak8_eta"] = ak8_eta
        out_data["ak8_phi"] = ak8_phi
        out_data["ak8_sdmass"] = ak8_sdmass
        out_data["ak8_rawFactor"] = ak8_rawFactor
        out_data["ak8_tau21"] = ak8_tau21
        out_data["ak8_tau32"] = ak8_tau32
        out_data["ak8_gpt_bb"] = ak8_gpt_bb
        out_data["ak8_gpt_cc"] = ak8_gpt_cc
        out_data["ak8_gpt_bc"] = ak8_gpt_bc
        out_data["ak8_gpt_qcd"] = ak8_gpt_qcd
        out_data["ak8_gpt_topbw"] = ak8_gpt_topbw
        out_data["ak8_gpt_topw"] = ak8_gpt_topw
        out_data["ak8_gpt_bs"] = ak8_gpt_bs
        out_data["ak8_gpt_cs"] = ak8_gpt_cs
        out_data["ak8_gpt_qq"] = ak8_gpt_qq
        out_data["ak8_nConstituents"] = ak8_nConstituents

        if 'TrigSF' in self._channel:
            try:
                out_data['rho_Calo'] = event.Rho_fixedGridRhoFastjetCentralCalo
                out_data['rho_ChargedPileUp'] = event.Rho_fixedGridRhoFastjetCentralChargedPileUp
            except RuntimeError:
                out_data['rho_Calo'] = event.fixedGridRhoFastjetCentralCalo
                out_data['rho_ChargedPileUp'] = event.fixedGridRhoFastjetCentralChargedPileUp

            out_data['n_PV'] = event.PV_npvs
            out_data['n_PV_good'] = event.PV_npvsGood

        if self._opts['fillEventVars']:
            out_data["ht_b"] = sum([j.pt for j in event.ak4_b_jets])
            out_data["ht_c"] = sum([j.pt for j in event.ak4_c_jets])
            out_data["ht_bc"] = sum([j.pt for j in event.ak4_b_or_c_jets])

            for name, jets in zip(['b', 'c', 'bc'], [event.ak4_b_jets, event.ak4_c_jets, event.ak4_b_or_c_jets]):
                if len(jets) >= 2:
                    (i, j), min_dr2 = closest_pair(jets)
                    out_data[f"minDR_{name}"] = math.sqrt(min_dr2) if min_dr2 >= 0 else min_dr2
                    out_data[f"mass_minDR_{name}"] = sumP4(jets[i], jets[j]).mass()
                    _, max_mass = closest_pair(jets, lambda a, b: sumP4(a, b).mass(), reverse=True)
                    out_data[f"maxMass_{name}"] = max_mass
                else:
                    out_data[f"minDR_{name}"] = -1
                    out_data[f"mass_minDR_{name}"] = -1
                    out_data[f"maxMass_{name}"] = -1

        for key in out_data:
            self.out.fillBranch(key, out_data[key])
    #@timeit
    def _fillNN(self, event, daMode=None):
        # ====== fill jet vars ======
        inputs = {
            'ak4_pt_log': [],
            'ak4_energy_log': [],
            'ak4_eta': [],
            'ak4_tag_B4': [],
            'ak4_tag_B3': [],
            'ak4_tag_B2': [],
            'ak4_tag_B1': [],
            'ak4_tag_B0': [],
            'ak4_tag_C4': [],
            'ak4_tag_C3': [],
            'ak4_tag_C2': [],
            'ak4_tag_C1': [],
            'ak4_tag_C0': [],
            'ak4_px': [],
            'ak4_py': [],
            'ak4_pz': [],
            'ak4_energy': [],
        }
        for j in event.ak4jets:
            jet_p4 = polarP4(j)
            inputs['ak4_pt_log'].append(math.log(jet_p4.pt()))
            inputs['ak4_energy_log'].append(math.log(jet_p4.energy()))
            inputs['ak4_eta'].append(jet_p4.eta())
            inputs['ak4_tag_B4'].append(j.tag == 54)
            inputs['ak4_tag_B3'].append(j.tag == 53)
            inputs['ak4_tag_B2'].append(j.tag == 52)
            inputs['ak4_tag_B1'].append(j.tag == 51)
            inputs['ak4_tag_B0'].append(j.tag == 50)
            inputs['ak4_tag_C4'].append(j.tag == 44)
            inputs['ak4_tag_C3'].append(j.tag == 43)
            inputs['ak4_tag_C2'].append(j.tag == 42)
            inputs['ak4_tag_C1'].append(j.tag == 41)
            inputs['ak4_tag_C0'].append(j.tag == 40)
            inputs['ak4_px'].append(jet_p4.px())
            inputs['ak4_py'].append(jet_p4.py())
            inputs['ak4_pz'].append(jet_p4.pz())
            inputs['ak4_energy'].append(jet_p4.energy())
        # add jet mask
        inputs['ak4_mask'] = np.ones_like(inputs['ak4_energy'])

        # ====== fill lepton+MET vars ======
        n_leps = int(self._channel[0])
        if n_leps > 0:
            inputs.update(**{
                'lep_pt_log': [],
                'lep_energy_log': [],
                'lep_eta': [],
                'lep_isMu': [],
                'lep_isEl': [],
                'lep_px': [],
                'lep_py': [],
                'lep_pz': [],
                'lep_energy': [],
            })
            for lep in event.selectedLeptons[:n_leps]:
                lep_p4 = polarP4(lep)
                inputs['lep_pt_log'].append(math.log(lep_p4.pt()))
                inputs['lep_energy_log'].append(math.log(lep_p4.energy()))
                inputs['lep_eta'].append(lep_p4.eta())
                inputs['lep_isMu'].append(abs(lep.pdgId) == 13)
                inputs['lep_isEl'].append(abs(lep.pdgId) == 11)
                inputs['lep_px'].append(lep_p4.px())
                inputs['lep_py'].append(lep_p4.py())
                inputs['lep_pz'].append(lep_p4.pz())
                inputs['lep_energy'].append(lep_p4.energy())
            # MET
            met_p4 = event.met.p4()
            inputs['lep_pt_log'].append(math.log(met_p4.pt()))
            inputs['lep_energy_log'].append(math.log(met_p4.energy()))
            inputs['lep_eta'].append(0)
            inputs['lep_isMu'].append(0)
            inputs['lep_isEl'].append(0)
            inputs['lep_px'].append(met_p4.px())
            inputs['lep_py'].append(met_p4.py())
            inputs['lep_pz'].append(met_p4.pz())
            inputs['lep_energy'].append(met_p4.energy())
            # add lepton mask
            inputs['lep_mask'] = np.ones_like(inputs['lep_energy'])

        # TODO: update
        if daMode == None:
            model_idx = event.event % self.nn_helper.k_fold if self._year == 2017 else None
            outputs = self.nn_helper.predict(inputs, model_idx)
            category = max(outputs, key=outputs.get)
            category_id = None
            for idx, name in enumerate(self.nn_helper.output_names):
                self.out.fillBranch(name, outputs[name])
                # self.out.fillBranch(name.replace('score', 'cat'), name == category)
                if name == category:
                    category_id = idx
            assert category_id is not None
            self.out.fillBranch('nn_category', category_id)
            # [TODO] change this to our score
            event.signalScore_nn = (outputs["score_ttHcc"] + outputs["score_ttHbb"] + outputs["score_ttZcc"] +
                                    outputs["score_ttZbb"]) / (1. - outputs["score_ttZqq"])
            event.bkgScore_nn = (outputs["score_ttLF"]) / (1. - outputs["score_ttZqq"])
            if self._channel == '0L' and self._opts['apply_qcd_cut'] is not None:
                event.qcdScore_nn = outputs["score_qcd"]
        elif daMode == 1:
            model_idx = event.event % self.nn_da_op1_helper.k_fold if self._year == 2018 else None
            outputs = self.nn_da_op1_helper.predict(inputs, model_idx)
            category = max(outputs, key=outputs.get)
            for idx, name in enumerate(self.nn_da_op1_helper.output_names):
                self.out.fillBranch(name, outputs[name])
            event.signalScore_nn_da_op1 = (
                outputs["score_da_op1_ttHcc"] + outputs["score_da_op1_ttHbb"] + outputs["score_da_op1_ttZcc"] +
                outputs["score_da_op1_ttZbb"]) / (
                1. - outputs["score_da_op1_ttZqq"])
            event.bkgScore_nn_da_op1 = (outputs["score_da_op1_ttLF"]) / (1. - outputs["score_da_op1_ttZqq"])
            if self._channel == '0L' and self._opts['apply_qcd_cut'] is not None:
                event.qcdScore_nn_da_op1 = outputs["score_da_op1_qcd"]
        elif daMode == 2:
            model_idx = event.event % self.nn_da_op2_helper.k_fold if self._year == 2018 else None
            outputs = self.nn_da_op2_helper.predict(inputs, model_idx)
            category = max(outputs, key=outputs.get)
            for idx, name in enumerate(self.nn_da_op2_helper.output_names):
                self.out.fillBranch(name, outputs[name])
            event.signalScore_nn_da_op2 = (
                outputs["score_da_op2_ttHcc"] + outputs["score_da_op2_ttHbb"] + outputs["score_da_op2_ttZcc"] +
                outputs["score_da_op2_ttZbb"]) / (
                1. - outputs["score_da_op2_ttZqq"])
            event.bkgScore_nn_da_op2 = (outputs["score_da_op2_ttLF"]) / (1. - outputs["score_da_op2_ttZqq"])
            if self._channel == '0L' and self._opts['apply_qcd_cut'] is not None:
                event.qcdScore_nn_da_op2 = outputs["score_da_op2_qcd"]
    #@timeit
    def _fillMLP(self, event):
        # ====== fill jet vars ======
        inputs = {
            'ak4_pt_log': [],
            'ak4_energy_log': [],
            'ak4_eta': [],
            'ak4_btag_L': [],
            'ak4_btag_M': [],
            'ak4_btag_T': [],
            'ak4_ctag_L': [],
            'ak4_ctag_M': [],
            'ak4_ctag_T': [],
        }
        for j in event.ak4jets:
            jet_p4 = polarP4(j)
            inputs['ak4_pt_log'].append(math.log(jet_p4.pt()))
            inputs['ak4_energy_log'].append(math.log(jet_p4.energy()))
            inputs['ak4_eta'].append(jet_p4.eta())
            inputs['ak4_btag_L'].append(j.btag_L)
            inputs['ak4_btag_M'].append(j.btag_M)
            inputs['ak4_btag_T'].append(j.btag_T)
            inputs['ak4_ctag_L'].append(j.ctag_L)
            inputs['ak4_ctag_M'].append(j.ctag_M)
            inputs['ak4_ctag_T'].append(j.ctag_T)

        # ====== fill lepton+MET vars ======
        n_leps = int(self._channel[0])
        if n_leps > 0:
            inputs.update(**{
                'lep_pt_log': [],
                'lep_energy_log': [],
                'lep_eta': [],
                'lep_isMu': [],
                'lep_isEl': [],
            })
            for lep in event.selectedLeptons[:n_leps]:
                lep_p4 = polarP4(lep)
                inputs['lep_pt_log'].append(math.log(lep_p4.pt()))
                inputs['lep_energy_log'].append(math.log(lep_p4.energy()))
                inputs['lep_eta'].append(lep_p4.eta())
                inputs['lep_isMu'].append(abs(lep.pdgId) == 13)
                inputs['lep_isEl'].append(abs(lep.pdgId) == 11)
            # MET
            met_p4 = event.met.p4()
            inputs['lep_pt_log'].append(math.log(met_p4.pt()))
            inputs['lep_energy_log'].append(math.log(met_p4.energy()))
            inputs['lep_eta'].append(0)
            inputs['lep_isMu'].append(0)
            inputs['lep_isEl'].append(0)

        # ====== fill event vars ======
        def _get(name):
            return self.out._branches[name].buff[0]

        inputs['ht_log'] = [math.log(_get('ht'))]
        for name in ['b', 'c', 'bc']:
            inputs[f'ht_{name}_log'] = [math.log(_get(f'ht_{name}')) if _get(f'ht_{name}') > 0 else 0]
            inputs[f'minDR_{name}'] = [_get(f'minDR_{name}')]
            inputs[f'mass_minDR_{name}_log'] = [
                math.log(_get(f'mass_minDR_{name}')) if _get(f'mass_minDR_{name}') > 0 else 0]
            inputs[f'maxMass_{name}_log'] = [math.log(_get(f'maxMass_{name}')) if _get(f'maxMass_{name}') > 0 else 0]

        # TODO: update
        model_idx = event.event % self.mlp_helper.k_fold if self._year == 2018 else None
        outputs = self.mlp_helper.predict(inputs, model_idx)
        for name in outputs:
            self.out.fillBranch(name, outputs[name])

    def _fillBDTVars(self, event):
        deltaravebj = float(0)
        deltarminbj = float(99)  # just for m of the pair with min DR, not to be filled

        deltarminbb = float(99)
        deltaravebb = float(0)
        nbb = 0
        deltaetamaxbb = float(-99)
        sumbavg2 = float(0)
        jb1 = 0
        jb2 = 0
        mbbmindr = -99
        mbbmax = -99
        mbbave = float(0)
        mbbclosestH = float(99999)
        sumptbbmindr = -99
        mbjmindr = -99
        deltaravebj = float(0)
        nbj = 0
        mblmindr = -99
        deltarminbl = float(99)
        njjmH = 0
        # print ("new event")
        AAak4_Htb = float(0.0)
        AASumpt_ak4_l = float(0.0)
        sum_ak4_b_bdisc = float(0)
        sum_ak4_bdisc = float(0)

        sum_ak4_mass = float(0)
        Clj = float(0)
        Elj = float(0)
        Clb = float(0)
        Elb = float(0)

        for j in event.ak4_b_jets:
            AAak4_Htb += j.pt
            sum_ak4_b_bdisc += j.btagDeepFlavB
            if (deltaR(event.selectedLeptons[0], j) < deltarminbl):
                deltarminbl = deltaR(event.selectedLeptons[0], j)
                mblmindr = sumP4(j, event.selectedLeptons[0]).M()
            # print ("first jet index "+str(jb1))
            Clb += j.pt
            Elb += polarP4(j).E()
            for l in event.selectedLeptons:
                Clb += l.pt
                Elb += polarP4(l).E()
            sumbavg2 += (j.btagDeepFlavB - sum_ak4_b_bdisc / len(event.ak4_b_jets)
                         ) * (j.btagDeepFlavB - sum_ak4_b_bdisc / len(event.ak4_b_jets))
            for j2 in event.ak4_b_jets:
                if (jb2 <= jb1):
                    jb2 += 1
                    continue
                # print ("second jet index " +str(jb2))
                deltaravebb += deltaR(j2, j)
                nbb += 1
                # print ("deltar 1 2 "+str(deltaR(j2,j)))
                mbbave += sumP4(j, j2).M()
                if (deltaR(j2, j) < deltarminbb):
                    deltarminbb = deltaR(j2, j)
                    mbbmindr = sumP4(j, j2).M()
                    sumptbbmindr = j.pt + j2.pt
                    # print (mbbmindr)
                if (deltaEta(j2, j) > deltaetamaxbb):
                    deltaetamaxbb = deltaEta(j2, j)
                if (sumP4(j, j2).M() > mbbmax):
                    mbbmax = sumP4(j, j2).M()
                if (abs(sumP4(j, j2).M() - 125) < abs(mbbclosestH - 125)):
                    mbbclosestH = sumP4(j, j2).M()
                jb2 += 1
            for j2 in event.ak4jets:
                deltaravebj += deltaR(j2, j)
                nbj += 1
                if (sumP4(j, j2).M() < 140 and sumP4(j, j2).M() > 100):
                    njjmH += 1
                if (deltaR(j2, j) < deltarminbj):
                    deltarminbj = deltaR(j2, j)
                    mbjmindr = sumP4(j, j2).M()
            jb1 += 1
        deltaravebb = deltaravebb / nbb if nbb != 0 else -99
        deltaravebj = deltaravebj / nbj if nbj != 0 else -99
        mbbave = mbbave / nbb if nbb != 0 else -99
        Clb = Clb / Elb if Elb != 0 else 0

        ak4jets_sortedb = sorted(event.ak4jets, key=lambda x: x.btagDeepFlavB, reverse=True)
        ak4_b_jets_sortedb = sorted(event.ak4_b_jets, key=lambda x: x.btagDeepFlavB, reverse=True)
#        print ("new event")
#        for j in ak4jets_sortedb:
#            print (j.btagDeepFlavB)
#        print ("same for b jets")
#        for j in ak4_b_jets_sortedb:
#            print (j.btagDeepFlavB)

        deltarminjj = float(99)
        deltaravejj = float(0)
        njj = 0
        deltaRmaxjj = float(-99)
        deltaetamaxjj = float(-99)
        jj1 = 0
        jj2 = 0
        mjjmindr = -99
        sumptjjmindr = -99
        # print ("new event")
        for j in event.ak4jets:
            sum_ak4_mass += j.mass
            sum_ak4_bdisc += j.btagDeepFlavB
            AASumpt_ak4_l += j.pt
            Clj += j.pt
            Elj += polarP4(j).E()
            for l in event.selectedLeptons:
                AASumpt_ak4_l += l.pt
                Clj += l.pt
                Elj += polarP4(l).E()
            # print ("first jet index "+str(jj1))
            for j2 in event.ak4jets:
                if (jj2 <= jj1):
                    jj2 += 1
                    continue
                # print ("second jet index " +str(jj2))
                deltaravejj += deltaR(j2, j)
                njj += 1
                # print ("deltar 1 2 "+str(deltaR(j2,j)))
                if (deltaR(j2, j) < deltarminjj):
                    deltarminjj = deltaR(j2, j)
                    mjjmindr = sumP4(j, j2).M()
                    sumptjjmindr = j.pt + j2.pt
                    # print (mjjmindr)
                if (deltaEta(j2, j) > deltaetamaxjj):
                    deltaetamaxjj = deltaEta(j2, j)
                if (deltaR(j2, j) > deltaRmaxjj):
                    deltaRmaxjj = deltaR(j2, j)
                jj2 += 1
            jj1 += 1
        deltaravejj = deltaravejj / njj if njj != 0 else -99
        Clj = Clj / Elj if Elj != 0 else 0
        if (len(event.ak4jets)) > 1:
            self.out.fillBranch("d2_ak4", ak4jets_sortedb[1].btagDeepFlavB)
        else:
            self.out.fillBranch("d2_ak4", -9)
        if (len(event.ak4jets)) > 0:
            self.out.fillBranch("Sum_pt_ak4_l", AASumpt_ak4_l)
            self.out.fillBranch("djavg_ak4", sum_ak4_bdisc / len(event.ak4jets))
            self.out.fillBranch("mjavg_ak4", sum_ak4_mass / len(event.ak4jets))

        else:
            self.out.fillBranch("djavg_ak4", -9)
            self.out.fillBranch("mjavg_ak4", 0)
            self.out.fillBranch("Sum_pt_ak4_l", 0)
        self.out.fillBranch("deltaravebb_ak4", deltaravebb)
        self.out.fillBranch("mbbave_ak4", mbbave)
        self.out.fillBranch("mbbclosestH_ak4", mbbclosestH)
        self.out.fillBranch("mbbmindr_ak4", mbbmindr)
        self.out.fillBranch("mbbmax_ak4", mbbmax)
        self.out.fillBranch("deltarminbb_ak4", deltarminbb)
        self.out.fillBranch("sumptbbmindr_ak4", sumptbbmindr)
        self.out.fillBranch("deltaetamaxbb_ak4", deltaetamaxbb)
        self.out.fillBranch("Clb_ak4", Clb)
        self.out.fillBranch("Clj_ak4", Clj)
        self.out.fillBranch("mblmindr_ak4", mblmindr)
        self.out.fillBranch("deltarminbl_ak4", deltarminbl)
        self.out.fillBranch("deltaravebj_ak4", deltaravebj)
        self.out.fillBranch("mbjmindr_ak4", mbjmindr)
        self.out.fillBranch("njjmH_ak4", njjmH)
        self.out.fillBranch(
            "mjjj_ak4", sumP4(event.ak4jets[1],
                              event.ak4jets[0],
                              event.ak4jets[2]).M() if len(event.ak4jets) > 2 else 0)

        self.out.fillBranch("deltaravejj_ak4", deltaravejj)
        self.out.fillBranch("mjjmindr_ak4", mjjmindr)
        self.out.fillBranch("deltarminjj_ak4", deltarminjj)
        self.out.fillBranch("sumptjjmindr_ak4", sumptjjmindr)
        self.out.fillBranch("deltaetamaxjj_ak4", deltaetamaxjj)
        self.out.fillBranch("deltaRmaxjj_ak4", deltaRmaxjj)

        if (len(event.ak4_b_jets)) > 0:
            self.out.fillBranch("Sum_Htb_ak4", AAak4_Htb)
            self.out.fillBranch("d_b_avg2_ak4", sumbavg2 / len(event.ak4_b_jets))
            self.out.fillBranch("dbavg_ak4", sum_ak4_b_bdisc / len(event.ak4_b_jets))
            self.out.fillBranch("dbmin_ak4", ak4_b_jets_sortedb[len(event.ak4_b_jets) - 1].btagDeepFlavB)
        else:
            self.out.fillBranch("Sum_Htb_ak4", 0)
            self.out.fillBranch("d_b_avg2_ak4", -9)
            self.out.fillBranch("dbavg_ak4", -9)
            self.out.fillBranch("dbmin_ak4", -9)
    #@timeit
    @getentries
    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        event.idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
        event._allJets = Collection(event, "Jet")
        event._allFatJets = Collection(event, "FatJet")
        event.met = METObject(event, "PuppiMET") if self._usePuppiJets else METObject(event, "MET")

        self._selectLeptons(event)
        if self._preSelect(event) is False:
            return False
        if self._selectTriggers(event) is False:
            return False

        self._correctJetAndMET(event)
        self._cleanObjects(event)
        if self._selectEvent(event) is False:
            return False
        # fill
        self._fillEventInfo(event)
        self._fillGenMatch(event)
        self._fillAk8Match(event)
        if self._opts['eval_nn']:
            self._fillNN(event)
        if self._opts['eval_nn_da_op1']:
            self._fillNN(event, daMode=1)
        if self._opts['eval_nn_da_op2']:
            self._fillNN(event, daMode=2)
        if self._opts['eval_mlp']:
            self._fillMLP(event)
        if self._opts['fillBDTVars']:
            self._fillBDTVars(event)

        # apply cuts on the NN output scores to reduce tree size
        # sum ttH+ttZ scores > 0.4
        # ttLF score < 0.1
        if self._opts['apply_score_selection']:
            keepEvent = (
                self._opts['eval_nn'] and event.signalScore_nn > 0.4 and event.bkgScore_nn < 0.1) or (
                self._opts['eval_nn_da_op1'] and event.signalScore_nn_da_op1 > 0.4 and event.bkgScore_nn_da_op1 < 0.1) or (
                self._opts['eval_nn_da_op2'] and event.signalScore_nn_da_op2 > 0.4 and event.bkgScore_nn_da_op2 < 0.1)
            c = self._opts['apply_qcd_cut']
            if self._channel == '0L' and c is not None:
                keepEvent = keepEvent and (
                    (self._opts['eval_nn'] and event.qcdScore_nn < c) or
                    (self._opts['eval_nn_da_op1'] and event.qcdScore_nn_da_op1 < c) or
                    (self._opts['eval_nn_da_op2'] and event.qcdScore_nn_da_op2 < c)
                )
            if keepEvent == False:
                return False

        for mod in self._modules.values():
            assert mod.analyze(event)

        return True


def tthTreeFromConfig():
    import yaml
    with open('tthtree_cfg.json') as f:
        cfg = yaml.safe_load(f)
    return TTHTreeProducer(**cfg)
