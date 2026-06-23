import math
import os
import numpy as np
import onnxruntime

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from ..helpers.utils import polarP4


class TopSystReweightingProducer(Module, object):
    '''https://twiki.cern.ch/twiki/bin/viewauth/CMS/MLReweighting'''

    def __init__(self, year, **kwargs):
        self.year = year
        self._opts = {
            'fillSystWeights': True,
        }
        self._opts.update(**kwargs)

        if self._opts['fillSystWeights']:
            prefix = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoTTH/data')
            models = {
                'topHdampWeightUp': f'{prefix}/gen/mymodel12_hdamp_up_13TeV.onnx',
                'topHdampWeightDown': f'{prefix}/gen/mymodel12_hdamp_down_13TeV.onnx',
                'bFragWeightNom': f'{prefix}/gen/mymodel12_rB_nom_CP5_2M.onnx',
                'bFragWeightUp': f'{prefix}/gen/mymodel12_rB_up_CP5_2M.onnx',
            }

            options = onnxruntime.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sessions = {
                k: onnxruntime.InferenceSession(model_path, sess_options=options, providers=['CPUExecutionProvider'])
                for k, model_path in models.items()}

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        if self.isMC:
            self.out = wrappedOutputTree

            if self._opts['fillSystWeights']:
                self.out.branch('topHdampWeightUp', "F")
                self.out.branch('topHdampWeightDown', "F")
                self.out.branch('bFragWeightNom', "F")
                self.out.branch('bFragWeightUp', "F")

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        if not self.isMC:
            return True

        if not self._opts['fillSystWeights']:
            return True

        wgts = {
            'topHdampWeightUp': 1,
            'topHdampWeightDown': 1,
            'bFragWeightNom': 1,
            'bFragWeightUp': 1,
        }

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

        genTopFirst = None
        genAntiTopFirst = None
        for gp in genparts:
            # 12 : isFirstCopy
            if gp.statusFlags & (1 << 12):
                if gp.pdgId == 6:
                    genTopFirst = gp
                elif gp.pdgId == -6:
                    genAntiTopFirst = gp

        if genTopFirst and genAntiTopFirst:
            p_top = polarP4(genTopFirst)
            p_tbar = polarP4(genAntiTopFirst)
            if (p_top + p_tbar).pt() < 1000:
                hdamp = 1.379  # hdamp value of the sample / 172.5
                maxM = 243.9517
                inputs = np.array(
                    [[math.log10(p_top.pt()),
                      p_top.Rapidity(),
                      p_top.phi(),
                      p_top.mass() / maxM, 0.1, hdamp],
                     [math.log10(p_tbar.pt()),
                     p_tbar.Rapidity(),
                     p_tbar.phi(),
                     p_tbar.mass() / maxM, 0.2, hdamp],
                     ], dtype='float32')

                for k in ('topHdampWeightUp', 'topHdampWeightDown'):
                    pred = self.sessions[k].run([], {'input': inputs[None,]})[0][0]
                    wgts[k] = pred[0] / pred[1]

        del genTopFirst, genAntiTopFirst

        genTopLast = None
        genAntiTopLast = None
        for gp in genparts:
            # 13 : isLastCopy
            if gp.statusFlags & (1 << 13) == 0:
                continue
            if abs(gp.pdgId) == 6:
                for idx in gp.dauIdx:
                    dau = genparts[idx]
                    if abs(dau.pdgId) == 24:
                        gp.genW = dau
                    elif abs(dau.pdgId) in (1, 3, 5):
                        gp.genB = dau
                if gp.pdgId == 6:
                    genTopLast = gp
                elif gp.pdgId == -6:
                    genAntiTopLast = gp

        if genTopLast and genAntiTopLast:
            bHadronFromTop = None
            bHadronFromAntiTop = None
            for idx in genTopLast.genB.dauIdx:
                dau = genparts[idx]
                if 500 < abs(dau.pdgId) < 600 or 5000 < abs(dau.pdgId) < 6000:
                    bHadronFromTop = dau
            for idx in genAntiTopLast.genB.dauIdx:
                dau = genparts[idx]
                if 500 < abs(dau.pdgId) < 600 or 5000 < abs(dau.pdgId) < 6000:
                    bHadronFromAntiTop = dau
            if bHadronFromTop and bHadronFromAntiTop:
                p_top = polarP4(genTopLast)
                p_tbar = polarP4(genAntiTopLast)
                p_top_w = polarP4(genTopLast.genW)
                p_tbar_w = polarP4(genAntiTopLast.genW)
                p_top_b_hadron = polarP4(bHadronFromTop)
                p_tbar_b_hadron = polarP4(bHadronFromAntiTop)

                x_top = 2 * p_top_b_hadron.Dot(p_top) / p_top.M2() / (1 - p_top_w.M2() / p_top.M2())
                x_tbar = 2 * p_tbar_b_hadron.Dot(p_tbar) / p_tbar.M2() / (1 - p_tbar_w.M2() / p_tbar.M2())
                r_b = 0.855
                inputs = np.array([
                    [x_top, r_b],
                    [x_tbar, r_b],
                ], dtype='float32')

                for k in ('bFragWeightNom', 'bFragWeightUp'):
                    pred = self.sessions[k].run([], {'input': inputs[None,]})[0][0]
                    wgts[k] = pred[0] / pred[1]

        for k, v in wgts.items():
            self.out.fillBranch(k, v)

        return True


# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
def topSystReweighter():
    return TopSystReweightingProducer(0)
