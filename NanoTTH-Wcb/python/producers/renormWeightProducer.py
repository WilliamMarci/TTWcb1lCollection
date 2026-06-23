import os
import re
import numpy as np
import math
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import onnxruntime

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from ..helpers.utils import deltaPhi, deltaR, deltaR2, deltaEta, closest, polarP4, sumP4, transverseMass, minValue, configLogger, getDigit, closest_pair, clip

from .topPtWeightProducer import TopPtWeightProducer
from .topSystReweightingProducer import TopSystReweightingProducer

import logging
logger = logging.getLogger('nano')
configLogger('nano', loglevel=logging.INFO)

class RenormWeightProducer(Module, object):

    def __init__(self, **kwargs):
        self._year = int(kwargs['year'])
        self._opts = {
        }
        for k in kwargs:
            self._opts[k] = kwargs[k]

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

        self.filename  = inputFile.GetName().split("/")[-5]
        self.isttbar = "TTTo" in self.filename
        self.isttz = "TTZTo" in self.filename
        self.istthiggs = "ttHTo" in self.filename
        self.is4FS = "TTbb_" in self.filename and not "facscfact2p0" in self.filename
        self.is2muF = "TTbb_" in self.filename and "facscfact2p0" in self.filename
        self.isDPS = "bbDPS" in self.filename

        self.labels_ = [
                        "nom", "muR_up", "muR_down", "muF_up", "muF_down", "isr_up", "isr_down", "fsr_up", "fsr_down",
                        "topPt_nom", "topPt_up", "topPt_down", "hdampML_up", "hdampML_down",
                        "pdfSum_up", "pdfSum_down", "pdfSumWAlphaS_up", "pdfSumWAlphaS_down", "alphas_up", "alphas_down",
                        # now the extended PS weights
                        "fsr_G2GG_muR_up", "fsr_G2QQ_muR_up", "fsr_Q2QG_muR_up", "fsr_X2XG_muR_up",
                        "fsr_G2GG_cNS_up", "fsr_G2QQ_cNS_up", "fsr_Q2QG_cNS_up", "fsr_X2XG_cNS_up",
                        "isr_G2GG_muR_up", "isr_G2QQ_muR_up", "isr_Q2QG_muR_up", "isr_X2XG_muR_up",
                        "isr_G2GG_cNS_up", "isr_G2QQ_cNS_up", "isr_Q2QG_cNS_up", "isr_X2XG_cNS_up",

                        "fsr_G2GG_muR_down", "fsr_G2QQ_muR_down", "fsr_Q2QG_muR_down", "fsr_X2XG_muR_down",
                        "fsr_G2GG_cNS_down", "fsr_G2QQ_cNS_down", "fsr_Q2QG_cNS_down", "fsr_X2XG_cNS_down",
                        "isr_G2GG_muR_down", "isr_G2QQ_muR_down", "isr_Q2QG_muR_down", "isr_X2XG_muR_down",
                        "isr_G2GG_cNS_down", "isr_G2QQ_cNS_down", "isr_Q2QG_cNS_down", "isr_X2XG_cNS_down",
                        ]

        self.histos = []

        self.h_genwgt = ROOT.TH1D('sumWgts', 'sumWgts', len(self.labels_), 0, len(self.labels_))
        self.histos.append(self.h_genwgt)

        if self.isttbar or self.is4FS or self.isDPS or self.is2muF:
            self.h_genwgt_nom_ttbb4fs = ROOT.TH1D('sumWgts_ttbb4fs', 'sumWgts_ttbb4fs', len(self.labels_), 0, len(self.labels_))
            self.h_genwgt_nom_ttbb4fs2muF = ROOT.TH1D('sumWgts_ttbb4fs2muF', 'sumWgts_ttbb4fs2muF', len(self.labels_), 0, len(self.labels_))
            self.h_genwgt_nom_ttbj4fs = ROOT.TH1D('sumWgts_ttbj4fs', 'sumWgts_ttbj4fs', len(self.labels_), 0, len(self.labels_))
            self.h_genwgt_nom_ttbj4fs2muF = ROOT.TH1D('sumWgts_ttbj4fs2muF', 'sumWgts_ttbj4fs2muF', len(self.labels_), 0, len(self.labels_))
            self.h_genwgt_nom_ttbb5fs = ROOT.TH1D('sumWgts_ttbb5fs', 'sumWgts_ttbb5fs', len(self.labels_), 0, len(self.labels_))
            self.h_genwgt_nom_ttbj5fs = ROOT.TH1D('sumWgts_ttbj5fs', 'sumWgts_ttbj5fs', len(self.labels_), 0, len(self.labels_))
            self.h_genwgt_nom_ttbbdps = ROOT.TH1D('sumWgts_ttbbdps', 'sumWgts_ttbbdps', len(self.labels_), 0, len(self.labels_))
            self.h_genwgt_nom_ttbjdps = ROOT.TH1D('sumWgts_ttbjdps', 'sumWgts_ttbjdps', len(self.labels_), 0, len(self.labels_))
            self.h_genwgt_nom_ttcc = ROOT.TH1D('sumWgts_ttcc', 'sumWgts_ttcc', len(self.labels_), 0, len(self.labels_))
            self.h_genwgt_nom_ttcj = ROOT.TH1D('sumWgts_ttcj', 'sumWgts_ttcj', len(self.labels_), 0, len(self.labels_))
            self.h_genwgt_nom_ttlf = ROOT.TH1D('sumWgts_ttlf', 'sumWgts_ttlf', len(self.labels_), 0, len(self.labels_))

            self.histos.append(self.h_genwgt_nom_ttbb4fs)
            self.histos.append(self.h_genwgt_nom_ttbb4fs2muF)
            self.histos.append(self.h_genwgt_nom_ttbj4fs)
            self.histos.append(self.h_genwgt_nom_ttbj4fs2muF)
            self.histos.append(self.h_genwgt_nom_ttbb5fs)
            self.histos.append(self.h_genwgt_nom_ttbj5fs)
            self.histos.append(self.h_genwgt_nom_ttbbdps)
            self.histos.append(self.h_genwgt_nom_ttbjdps)
            self.histos.append(self.h_genwgt_nom_ttcc)
            self.histos.append(self.h_genwgt_nom_ttcj)
            self.histos.append(self.h_genwgt_nom_ttlf)

        if self.istthiggs:
            self.h_genwgt_nom_tthbb = ROOT.TH1D('sumWgts_tthbb', 'sumWgts_tthbb', len(self.labels_), 0, len(self.labels_))
            self.h_genwgt_nom_tthcc = ROOT.TH1D('sumWgts_tthcc', 'sumWgts_tthcc', len(self.labels_), 0, len(self.labels_))

            self.histos.append(self.h_genwgt_nom_tthbb)
            self.histos.append(self.h_genwgt_nom_tthcc)

        if self.isttz:
            self.h_genwgt_nom_ttzbb = ROOT.TH1D('sumWgts_ttzbb', 'sumWgts_ttzbb', len(self.labels_), 0, len(self.labels_))
            self.h_genwgt_nom_ttzcc = ROOT.TH1D('sumWgts_ttzcc', 'sumWgts_ttzcc', len(self.labels_), 0, len(self.labels_))

            self.histos.append(self.h_genwgt_nom_ttzbb)
            self.histos.append(self.h_genwgt_nom_ttzcc)

        
        for h_ in self.histos:
            xaxis = h_.GetXaxis()
            xaxis.SetAlphanumeric(True)
            for binX in range(1,h_.GetNbinsX()+1):
                xaxis.SetBinLabel(binX, self.labels_[binX-1])

        self.dataset = None
        r = re.search(('mc' if self.isMC else 'data') + r'\/([a-zA-Z0-9_\-]+)\/', inputFile.GetName())
        if r:
            self.dataset = r.groups()[0]

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        if self.isMC:
            cwd = ROOT.gDirectory
            outputFile.cd()
            for h_ in self.histos:
                h_.Write()
            cwd.cd()


    def _fillEventInfo(self, event):
        out_data = {}
        out_data["year"] = self._year

    def _fillHisto(self, event, h_):
        h_.Fill("nom", event.genWeight)

        h_.Fill("muR_up", event.LHEScaleWeight[7]*event.genWeight)
        h_.Fill("muR_down", event.LHEScaleWeight[1]*event.genWeight)
        h_.Fill("muF_up", event.LHEScaleWeight[5]*event.genWeight)
        h_.Fill("muF_down", event.LHEScaleWeight[3]*event.genWeight)

        h_.Fill("isr_up", event.PSWeight[0]*event.genWeight)
        h_.Fill("isr_down", event.PSWeight[2]*event.genWeight)
        h_.Fill("fsr_up", event.PSWeight[1]*event.genWeight)
        h_.Fill("fsr_down", event.PSWeight[3]*event.genWeight)

        # now the extended PS weights
        # "fsr_G2GG_muR_down", "fsr_G2QQ_muR_down", "fsr_Q2QG_muR_down", "fsr_X2XG_muR_down",
        # "fsr_G2GG_cNS_down", "fsr_G2QQ_cNS_down", "fsr_Q2QG_cNS_down", "fsr_X2XG_cNS_down",
        # "isr_G2GG_muR_down", "isr_G2QQ_muR_down", "isr_Q2QG_muR_down", "isr_X2XG_muR_down",
        # "isr_G2GG_cNS_down", "isr_G2QQ_cNS_down", "isr_Q2QG_cNS_down", "isr_X2XG_cNS_down",

        # indices from here https://github.com/cms-sw/cmssw/blob/1d517cc3ed9bb410dc82614f9be4a20c9dea3f37/Configuration/Generator/python/PSweightsPythia/PythiaPSweightsSettings_cfi.py
        h_.Fill("fsr_G2GG_muR_up", event.PSWeight[13]*event.genWeight)
        h_.Fill("fsr_G2QQ_muR_up", event.PSWeight[15]*event.genWeight)
        h_.Fill("fsr_Q2QG_muR_up", event.PSWeight[17]*event.genWeight)
        h_.Fill("fsr_X2XG_muR_up", event.PSWeight[19]*event.genWeight)
        h_.Fill("fsr_G2GG_cNS_up", event.PSWeight[21]*event.genWeight)
        h_.Fill("fsr_G2QQ_cNS_up", event.PSWeight[23]*event.genWeight)
        h_.Fill("fsr_Q2QG_cNS_up", event.PSWeight[25]*event.genWeight)
        h_.Fill("fsr_X2XG_cNS_up", event.PSWeight[27]*event.genWeight)

        h_.Fill("isr_G2GG_muR_up", event.PSWeight[29]*event.genWeight)
        h_.Fill("isr_G2QQ_muR_up", event.PSWeight[31]*event.genWeight)
        h_.Fill("isr_Q2QG_muR_up", event.PSWeight[33]*event.genWeight)
        h_.Fill("isr_X2XG_muR_up", event.PSWeight[35]*event.genWeight)
        h_.Fill("isr_G2GG_cNS_up", event.PSWeight[37]*event.genWeight)
        h_.Fill("isr_G2QQ_cNS_up", event.PSWeight[39]*event.genWeight)
        h_.Fill("isr_Q2QG_cNS_up", event.PSWeight[41]*event.genWeight)
        h_.Fill("isr_X2XG_cNS_up", event.PSWeight[43]*event.genWeight)

        h_.Fill("fsr_G2GG_muR_down", event.PSWeight[12]*event.genWeight)
        h_.Fill("fsr_G2QQ_muR_down", event.PSWeight[14]*event.genWeight)
        h_.Fill("fsr_Q2QG_muR_down", event.PSWeight[16]*event.genWeight)
        h_.Fill("fsr_X2XG_muR_down", event.PSWeight[18]*event.genWeight)
        h_.Fill("fsr_G2GG_cNS_down", event.PSWeight[20]*event.genWeight)
        h_.Fill("fsr_G2QQ_cNS_down", event.PSWeight[22]*event.genWeight)
        h_.Fill("fsr_Q2QG_cNS_down", event.PSWeight[24]*event.genWeight)
        h_.Fill("fsr_X2XG_cNS_down", event.PSWeight[26]*event.genWeight)

        h_.Fill("isr_G2GG_muR_down", event.PSWeight[28]*event.genWeight)
        h_.Fill("isr_G2QQ_muR_down", event.PSWeight[30]*event.genWeight)
        h_.Fill("isr_Q2QG_muR_down", event.PSWeight[32]*event.genWeight)
        h_.Fill("isr_X2XG_muR_down", event.PSWeight[34]*event.genWeight)
        h_.Fill("isr_G2GG_cNS_down", event.PSWeight[36]*event.genWeight)
        h_.Fill("isr_G2QQ_cNS_down", event.PSWeight[38]*event.genWeight)
        h_.Fill("isr_Q2QG_cNS_down", event.PSWeight[40]*event.genWeight)
        h_.Fill("isr_X2XG_cNS_down", event.PSWeight[42]*event.genWeight)

        h_.Fill("topPt_nom", event.wgts["topptWeight"]*event.genWeight)
        h_.Fill("topPt_up", (2.*event.wgts["topptWeight"]-1.)*event.genWeight)
        h_.Fill("topPt_down", event.genWeight)

        h_.Fill("hdampML_up", event.wgts["topHdampWeightUp"]*event.genWeight)
        h_.Fill("hdampML_down", event.wgts["topHdampWeightDown"]*event.genWeight)

        if event.nLHEPdfWeight > 100:
            h_.Fill("alphas_up", event.LHEPdfWeight[101]*event.genWeight)
            h_.Fill("alphas_down", event.LHEPdfWeight[102]*event.genWeight)

            h_.Fill("pdfSum_up", (1. + event.pdfSumWgt) * event.genWeight)
            h_.Fill("pdfSum_down", (1. - event.pdfSumWgt) * event.genWeight)

            h_.Fill("pdfSumWAlphaS_up", (1. + event.pdfSumWgtWAlphaS) * event.genWeight)
            h_.Fill("pdfSumWAlphaS_down", (1. - event.pdfSumWgtWAlphaS) * event.genWeight)


    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        event.idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)

        self._fillEventInfo(event)

        if not self.isMC:
            return

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

        def isHadronic(gp):
            if len(gp.dauIdx) == 0:
                logger.warning(f'Particle (pdgId={gp.pdgId}) has no daughters!')
            for idx in gp.dauIdx:
                if abs(genparts[idx].pdgId) < 6:
                    return True
            return False

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

        lepGenTops = []
        hadGenTops = []
        bFromTops = []
        hadGenWs = []
        hadGenZs = []
        hadGenHs = []
        lepGenHs = []

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

        genH = hadGenHs[0] if len(hadGenHs) else None
        # 0: no hadronic H; else: max pdgId of the daughters
        hdecay_ = abs(get_daughters(genH)[0].pdgId) if genH else 0

        genHlep = lepGenHs[0] if len(lepGenHs) else None
        # 0: no hadronic H; else: max pdgId of the daughters
        hdecaylep_ = abs(get_daughters(genHlep)[0].pdgId) if genHlep else 0

        genZ = hadGenZs[0] if len(hadGenZs) else None
        # 0: no hadronic Z; else: max pdgId of the daughters
        zdecay_ = abs(get_daughters(genZ)[0].pdgId) if genZ else 0

        genW = hadGenWs[0] if len(hadGenWs) else None
        # 0: no hadronic W; else: max pdgId of the daughters
        wdecay_ = max([abs(d.pdgId) for d in get_daughters(genW)], default=0) if genW else 0

        event.tt_category = getDigit(event.genTtbarId, 1)
        event.higgs_decay = hdecay_ if hdecay_ != 0 else hdecaylep_
        event.z_decay = zdecay_


        event.genZ_pt = genZ.pt if genZ else -1
        genZ_pt = genZ.pt if genZ else -1

        # https://github.com/cms-sw/cmssw/blob/master/TopQuarkAnalysis/TopTools/plugins/GenTtbarCategorizer.cc
        genEventClassifier_id = -99
        if nGenHs > 0:  # there is a gen level Higgs
            if hdecay_ == 4:
                genEventClassifier_id = 0  # higgs to cc
            elif hdecay_ == 5:
                genEventClassifier_id = 1  # higgs to bb
            elif hdecaylep_ == 15:
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
            if getDigit(event.genTtbarId, 1) == 5:  # b's abbompanying tt
                if getDigit(event.genTtbarId, 0) == 1:  # tt+b
                    genEventClassifier_id = 7
                if getDigit(event.genTtbarId, 0) == 2:  # tt+2b, 1 b jet with 2 hadrons
                    genEventClassifier_id = 8
                if getDigit(event.genTtbarId, 0) > 2:  # tt+bb
                    genEventClassifier_id = 9

        event.genEventClassifier = genEventClassifier_id

        # ------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------
        # copy from the top systematics weight producer
        event.wgts = {
            'topHdampWeightUp': 1,
            'topHdampWeightDown': 1,
            'bFragWeightNom': 1,
            'bFragWeightUp': 1,
            'topptWeight': 1,
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
                    event.wgts[k] = pred[0] / pred[1]

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
                if 500 < abs(dau.pdgId) < 600 or 500 < abs(dau.pdgId) < 600:
                    bHadronFromTop = dau
            for idx in genAntiTopLast.genB.dauIdx:
                dau = genparts[idx]
                if 500 < abs(dau.pdgId) < 600 or 500 < abs(dau.pdgId) < 600:
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
                    event.wgts[k] = pred[0] / pred[1]

        # ------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------
        # copy from the top pT weight producer

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

        genTops = []
        for gp in genparts:
            if gp.statusFlags & (1 << 13) == 0:
                # 13: isLastCopy
                continue
            if abs(gp.pdgId) == 6:
                genTops.append(gp)

        if len(genTops) == 2:
            # ttbar (+X ?)

            def wgt(pt):
                return math.exp(0.0615 - 0.0005 * clip(pt, 0, 800))

            def wgt_nnlo(pt):
                x = clip(pt, 0, 2000)
                return 0.103 * math.exp(-0.0118 * x) - 0.000134 * x + 0.973

            event.wgts["topptWeight"] = math.sqrt(wgt_nnlo(genTops[0].pt) * wgt_nnlo(genTops[1].pt))

        # ------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------

        # do the weight stuff here
        event.pdfSumWgt = 0.
        event.pdfSumWgtWAlphaS = 0.

        if event.nLHEPdfWeight > 101:
            for iPDF in range(1, 101):
                event.pdfSumWgt = event.pdfSumWgt + ((event.LHEPdfWeight[iPDF]/event.LHEPdfWeight[0] - 1.)**2.)
            for iPDF in range(1, 103):
                event.pdfSumWgtWAlphaS = event.pdfSumWgtWAlphaS + ((event.LHEPdfWeight[iPDF]/event.LHEPdfWeight[0] - 1.)**2.)
            event.pdfSumWgt = np.sqrt(event.pdfSumWgt)
            event.pdfSumWgtWAlphaS = np.sqrt(event.pdfSumWgt)
        else: #replica set
            mean_w =  np.mean(np.array([event.LHEPdfWeight[iPDF] for iPDF in range(event.nLHEPdfWeight)]))
            for iPDF in range(event.nLHEPdfWeight):
                event.pdfSumWgt = event.pdfSumWgt + ((event.LHEPdfWeight[iPDF] - mean_w)**2.)
            event.pdfSumWgt = np.sqrt(event.pdfSumWgt/(event.nLHEPdfWeight-1))
            event.pdfSumWgtWAlphaS = event.pdfSumWgt

        # find ttbar samples
        if self.is4FS or self.isttbar or self.isDPS or self.is2muF:
            if ((event.tt_category==0) and (event.higgs_decay==0) and (event.z_decay==0) and (event.genZ_pt < 0.) and (self.isttbar)):
                self._fillHisto(event, self.h_genwgt_nom_ttlf)
            elif (((event.genEventClassifier>=4) and (event.genEventClassifier<=5)) and (event.higgs_decay==0) and (event.z_decay==0) and (event.genZ_pt < 0.) and (self.isttbar)):
                self._fillHisto(event, self.h_genwgt_nom_ttcj)
            elif ((event.genEventClassifier==6) and (event.higgs_decay==0) and (event.z_decay==0) and (event.genZ_pt < 0.) and (self.isttbar)):
                self._fillHisto(event, self.h_genwgt_nom_ttcc)
            elif (((event.genEventClassifier>=7) and (event.genEventClassifier<=8)) and (event.higgs_decay==0) and (event.z_decay==0) and (event.genZ_pt < 0.)):
                if self.is4FS:
                    self._fillHisto(event, self.h_genwgt_nom_ttbj4fs)
                elif self.is2muF:
                    self._fillHisto(event, self.h_genwgt_nom_ttbj4fs2muF)
                elif self.isDPS:
                    self._fillHisto(event, self.h_genwgt_nom_ttbjdps)
                elif self.isttbar:
                    self._fillHisto(event, self.h_genwgt_nom_ttbj5fs)
            elif ((event.genEventClassifier==9) and (event.higgs_decay==0) and (event.z_decay==0) and (event.genZ_pt < 0.)):
                if self.is4FS:
                    self._fillHisto(event, self.h_genwgt_nom_ttbb4fs)
                elif self.is2muF:
                    self._fillHisto(event, self.h_genwgt_nom_ttbb4fs2muF)
                elif self.isDPS:
                    self._fillHisto(event, self.h_genwgt_nom_ttbbdps)
                elif self.isttbar:
                    self._fillHisto(event, self.h_genwgt_nom_ttbb5fs)
        elif self.istthiggs:
            if ((event.higgs_decay==5) and (event.z_decay==0) and (event.genZ_pt < 0.)):
                self._fillHisto(event, self.h_genwgt_nom_tthbb)
            elif ((event.higgs_decay==4) and (event.z_decay==0) and (event.genZ_pt < 0.)):
                self._fillHisto(event, self.h_genwgt_nom_tthcc)
        elif self.isttz:
            if ((event.z_decay==5) and (event.higgs_decay==0) and (event.genZ_pt > 0.)):
                self._fillHisto(event, self.h_genwgt_nom_ttzbb)
            elif ((event.z_decay==4) and (event.higgs_decay==0) and (event.genZ_pt > 0.)):
                self._fillHisto(event, self.h_genwgt_nom_ttzcc)

        # always fill this one
        self._fillHisto(event, self.h_genwgt)

        return True


def renormWeightFromConfig():
    import yaml
    with open('tthrenorm_cfg.json') as f:
        cfg = yaml.safe_load(f)
    return RenormWeightProducer(**cfg)
