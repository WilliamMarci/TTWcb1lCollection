import numpy as np
import os
import correctionlib
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from ..helpers.utils import deltaPhi, deltaR, deltaR2, deltaEta, closest, polarP4, sumP4, transverseMass, minValue, configLogger, getDigit, closest_pair

era_dict = {2015: '2016preVFP_UL', 2016: '2016postVFP_UL', 2017: '2017_UL', 2018: '2018_UL'}

class RenormWeightSFProducer(Module, object):

    def __init__(self, year, **kwargs):
        self.era = era_dict[year]
        self._opts = {
            'fillRenormWeights': True,
            'fillExtendedPSWeights': True,
        }
        self._opts.update(**kwargs)
        if self._opts['fillRenormWeights']:
            self.corr = correctionlib.CorrectionSet.from_file(os.path.expandvars(f'$CMSSW_BASE/src/PhysicsTools/NanoTTH/data/renorm_factors/renorm_{self.era}.json'))

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        self.filename  = inputFile.GetName().split("/")[-5]
        self.isttbar = self.filename.startswith("TTTo")
        self.isttz = self.filename.startswith("TTZTo")
        self.is4FS = self.filename.startswith("TTbb_") and "facscfact2p0" not in self.filename
        self.is2muF = self.filename.startswith("TTbb_") and "facscfact2p0" in self.filename
        self.isDPS = "bbDPS" in self.filename
        self.isHdampUp = "_hdampUP" in self.filename
        self.isHdampDown = "_hdampDOWN" in self.filename
        self.isHerwig = "herwig" in self.filename
        self.isFxFx = "amcatnloFXFX" in self.filename

        self.file_key = ""
        self.file_key_nom = ""
        
        if self.filename.startswith("TTTo2L2Nu_TuneCP5"):
            self.file_key = "ttbar-powheg-dl"
        elif self.filename.startswith("TTToSemiLeptonic_TuneCP5"):
            self.file_key = "ttbar-powheg-sl"
        elif self.filename.startswith("TTToHadronic_TuneCP5"):
            self.file_key = "ttbar-powheg-fh"
        elif self.filename.startswith("TTbb_4f_TTTo2L2Nu_TuneCP5"):
            self.file_key = "ttbb-4f-dl"
        elif self.filename.startswith("TTbb_4f_TTToSemiLeptonic_TuneCP5"):
            self.file_key = "ttbb-4f-sl"
        elif self.filename.startswith("TTbb_4f_TTToHadronic_TuneCP5"):
            self.file_key = "ttbb-4f-fh"
        elif self.filename.startswith("TTbb_TTto2L2Nu_4f_facscfact2p0_TuneCP5"):
            self.file_key = "ttbb-4f-facscfact2p0-dl"
        elif self.filename.startswith("TTbb_TTtoLNu2Q_4f_facscfact2p0_TuneCP5"):
            self.file_key = "ttbb-4f-facscfact2p0-sl"
        elif self.filename.startswith("TTbb_TTto4Q_4f_facscfact2p0_TuneCP5"):
            self.file_key = "ttbb-4f-facscfact2p0-fh"
        elif self.filename.startswith("TTTo2L2Nu_bbDPS_TuneCP5"):
            self.file_key = "ttbb-dps-dl"
        elif self.filename.startswith("TTToSemiLeptonic_bbDPS_TuneCP5"):
            self.file_key = "ttbb-dps-sl"
        elif self.filename.startswith("TTToHadronic_bbDPS_TuneCP5"):
            self.file_key = "ttbb-dps-fh"
        elif self.filename.startswith("TT_TuneCH3_13TeV-powheg-herwig"):
            self.file_key = "ttbar-herwig"
        elif self.filename.startswith("TTJets_TuneCP5_13TeV-amcatnloFXFX"):
            self.file_key = "ttbar-fxfx"
        elif self.filename.startswith("TTZToQQ_TuneCP5"):
            self.file_key = "ttZ-qq"
        elif self.filename.startswith("TTTo2L2Nu_hdampUP_TuneCP5"):
            self.file_key = "ttbar-powheg-dl-hdampUP"
            self.file_key_nom = "ttbar-powheg-dl"
        elif self.filename.startswith("TTTo2L2Nu_hdampDOWN_TuneCP5"):
            self.file_key = "ttbar-powheg-dl-hdampDOWN"
            self.file_key_nom = "ttbar-powheg-dl"
        elif self.filename.startswith("TTToSemiLeptonic_hdampUP_TuneCP5"):
            self.file_key = "ttbar-powheg-sl-hdampUP"
            self.file_key_nom = "ttbar-powheg-sl"
        elif self.filename.startswith("TTToSemiLeptonic_hdampDOWN_TuneCP5"):
            self.file_key = "ttbar-powheg-sl-hdampDOWN"
            self.file_key_nom = "ttbar-powheg-sl"
        elif self.filename.startswith("TTToHadronic_hdampUP_TuneCP5"):
            self.file_key = "ttbar-powheg-fh-hdampUP"
            self.file_key_nom = "ttbar-powheg-fh"
        elif self.filename.startswith("TTToHadronic_hdampDOWN_TuneCP5"):
            self.file_key = "ttbar-powheg-dl-hdampDOWN"
            self.file_key_nom = "ttbar-powheg-fh"
        elif self.filename.startswith("TTbb_4f_TTTo2L2Nu_hdampUP_TuneCP5"):
            self.file_key = "ttbb-4f-dl-hdampUP"
            self.file_key_nom = "ttbb-4f-dl"
        elif self.filename.startswith("TTbb_4f_TTTo2L2Nu_hdampDOWN_TuneCP5"):
            self.file_key = "ttbb-4f-dl-hdampDOWN"
            self.file_key_nom = "ttbb-4f-dl"
        elif self.filename.startswith("TTbb_4f_TTToSemiLeptonic_hdampUP_TuneCP5"):
            self.file_key = "ttbb-4f-sl-hdampUP"
            self.file_key_nom = "ttbb-4f-sl"
        elif self.filename.startswith("TTbb_4f_TTToSemiLeptonic_hdampDOWN_TuneCP5"):
            self.file_key = "ttbb-4f-sl-hdampDOWN"
            self.file_key_nom = "ttbb-4f-sl"
        elif self.filename.startswith("TTbb_4f_TTToHadronic_hdampUP_TuneCP5"):
            self.file_key = "ttbb-4f-fh-hdampUP"
            self.file_key_nom = "ttbb-4f-fh"
        elif self.filename.startswith("TTbb_4f_TTToHadronic_hdampDOWN_TuneCP5"):
            self.file_key = "ttbb-4f-fh-hdampDOWN"
            self.file_key_nom = "ttbb-4f-fh"
        
        
        if self.isMC:
            self.out = wrappedOutputTree

            if self._opts['fillRenormWeights']:
                self.out.branch('renormWeight_muR_up', "F")
                self.out.branch('renormWeight_muR_down', "F")
                self.out.branch('renormWeight_muF_up', "F")
                self.out.branch('renormWeight_muF_down', "F")
                self.out.branch('renormWeight_isr_up', "F")
                self.out.branch('renormWeight_isr_down', "F")
                self.out.branch('renormWeight_fsr_up', "F")
                self.out.branch('renormWeight_fsr_down', "F")
                # extended PS weights
                if self._opts['fillExtendedPSWeights']:
                    self.out.branch('renormWeight_fsr_G2GG_muR_up', "F")
                    self.out.branch('renormWeight_fsr_G2QQ_muR_up', "F")
                    self.out.branch('renormWeight_fsr_Q2QG_muR_up', "F")
                    self.out.branch('renormWeight_fsr_X2XG_muR_up', "F")
                    self.out.branch('renormWeight_fsr_G2GG_cNS_up', "F")
                    self.out.branch('renormWeight_fsr_G2QQ_cNS_up', "F")
                    self.out.branch('renormWeight_fsr_Q2QG_cNS_up', "F")
                    self.out.branch('renormWeight_fsr_X2XG_cNS_up', "F")
                    self.out.branch('renormWeight_isr_G2GG_muR_up', "F")
                    self.out.branch('renormWeight_isr_G2QQ_muR_up', "F")
                    self.out.branch('renormWeight_isr_Q2QG_muR_up', "F")
                    self.out.branch('renormWeight_isr_X2XG_muR_up', "F")
                    self.out.branch('renormWeight_isr_G2GG_cNS_up', "F")
                    self.out.branch('renormWeight_isr_G2QQ_cNS_up', "F")
                    self.out.branch('renormWeight_isr_Q2QG_cNS_up', "F")
                    self.out.branch('renormWeight_isr_X2XG_cNS_up', "F")
                    self.out.branch('renormWeight_fsr_G2GG_muR_down', "F")
                    self.out.branch('renormWeight_fsr_G2QQ_muR_down', "F")
                    self.out.branch('renormWeight_fsr_Q2QG_muR_down', "F")
                    self.out.branch('renormWeight_fsr_X2XG_muR_down', "F")
                    self.out.branch('renormWeight_fsr_G2GG_cNS_down', "F")
                    self.out.branch('renormWeight_fsr_G2QQ_cNS_down', "F")
                    self.out.branch('renormWeight_fsr_Q2QG_cNS_down', "F")
                    self.out.branch('renormWeight_fsr_X2XG_cNS_down', "F")
                    self.out.branch('renormWeight_isr_G2GG_muR_down', "F")
                    self.out.branch('renormWeight_isr_G2QQ_muR_down', "F")
                    self.out.branch('renormWeight_isr_Q2QG_muR_down', "F")
                    self.out.branch('renormWeight_isr_X2XG_muR_down', "F")
                    self.out.branch('renormWeight_isr_G2GG_cNS_down', "F")
                    self.out.branch('renormWeight_isr_G2QQ_cNS_down', "F")
                    self.out.branch('renormWeight_isr_Q2QG_cNS_down', "F")
                    self.out.branch('renormWeight_isr_X2XG_cNS_down', "F")

                self.out.branch('renormWeight_topPt_nom', "F")
                self.out.branch('renormWeight_topPt_up', "F")
                self.out.branch('renormWeight_topPt_down', "F")
                self.out.branch('renormWeight_hdampML_up', "F")
                self.out.branch('renormWeight_hdampML_down', "F")
                self.out.branch('renormWeight_hdamp_up', "F")
                self.out.branch('renormWeight_hdamp_down', "F")
                self.out.branch('renormWeight_herwig', "F")
                self.out.branch('renormWeight_fxfx', "F")
                self.out.branch('renormWeight_pdfSum_up', "F")
                self.out.branch('renormWeight_pdfSum_down', "F")
                self.out.branch('renormWeight_pdfSumWAlphaS_up', "F")
                self.out.branch('renormWeight_pdfSumWAlphaS_down', "F")
                self.out.branch('renormWeight_alphas_up', "F")
                self.out.branch('renormWeight_alphas_down', "F")

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass


    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        if not self.isMC:
            return True

        if not self._opts['fillRenormWeights']:
            return True

        self.labels_ = ["nom", "muR_up", "muR_down", "muF_up", "muF_down", "isr_up", "isr_down", "fsr_up", "fsr_down",
                        "topPt_nom", "topPt_up", "topPt_down", "hdampML_up", "hdampML_down",
                        "pdfSum_up", "pdfSum_down", "pdfSumWAlphaS_up", "pdfSumWAlphaS_down", "alphas_up", "alphas_down",
                        ]

        if self._opts['fillExtendedPSWeights']:
            self.labels_ += [
                        "fsr_G2GG_muR_up", "fsr_G2QQ_muR_up", "fsr_Q2QG_muR_up", "fsr_X2XG_muR_up",
                        "fsr_G2GG_cNS_up", "fsr_G2QQ_cNS_up", "fsr_Q2QG_cNS_up", "fsr_X2XG_cNS_up",
                        "isr_G2GG_muR_up", "isr_G2QQ_muR_up", "isr_Q2QG_muR_up", "isr_X2XG_muR_up",
                        "isr_G2GG_cNS_up", "isr_G2QQ_cNS_up", "isr_Q2QG_cNS_up", "isr_X2XG_cNS_up",

                        "fsr_G2GG_muR_down", "fsr_G2QQ_muR_down", "fsr_Q2QG_muR_down", "fsr_X2XG_muR_down",
                        "fsr_G2GG_cNS_down", "fsr_G2QQ_cNS_down", "fsr_Q2QG_cNS_down", "fsr_X2XG_cNS_down",
                        "isr_G2GG_muR_down", "isr_G2QQ_muR_down", "isr_Q2QG_muR_down", "isr_X2XG_muR_down",
                        "isr_G2GG_cNS_down", "isr_G2QQ_cNS_down", "isr_Q2QG_cNS_down", "isr_X2XG_cNS_down"
                        ]

        wgts = {
            'renormWeight_muR_up': 1,
            'renormWeight_muR_down': 1,
            'renormWeight_muF_up': 1,
            'renormWeight_muF_down': 1,
            'renormWeight_isr_up': 1,
            'renormWeight_isr_down': 1,
            'renormWeight_fsr_up': 1,
            'renormWeight_fsr_down': 1,
            'renormWeight_topPt_nom': 1,
            'renormWeight_topPt_up': 1,
            'renormWeight_topPt_down': 1,
            'renormWeight_hdampML_up': 1,
            'renormWeight_hdampML_down': 1,
            'renormWeight_pdfSum_up': 1,
            'renormWeight_pdfSum_down': 1,
            'renormWeight_pdfSumWAlphaS_up': 1,
            'renormWeight_pdfSumWAlphaS_down': 1,
            'renormWeight_alphas_up': 1,
            'renormWeight_alphas_down': 1,

            'renormWeight_hdamp_up': 1,
            'renormWeight_hdamp_down': 1,
            'renormWeight_fxfx': 1,
            'renormWeight_herwig': 1,
        }

        if self._opts['fillExtendedPSWeights']:
            wgts.update(
                {
                'renormWeight_fsr_G2GG_muR_up': 1,
                'renormWeight_fsr_G2QQ_muR_up': 1,
                'renormWeight_fsr_Q2QG_muR_up': 1,
                'renormWeight_fsr_X2XG_muR_up': 1,
                'renormWeight_fsr_G2GG_cNS_up': 1,
                'renormWeight_fsr_G2QQ_cNS_up': 1,
                'renormWeight_fsr_Q2QG_cNS_up': 1,
                'renormWeight_fsr_X2XG_cNS_up': 1,
                'renormWeight_isr_G2GG_muR_up': 1,
                'renormWeight_isr_G2QQ_muR_up': 1,
                'renormWeight_isr_Q2QG_muR_up': 1,
                'renormWeight_isr_X2XG_muR_up': 1,
                'renormWeight_isr_G2GG_cNS_up': 1,
                'renormWeight_isr_G2QQ_cNS_up': 1,
                'renormWeight_isr_Q2QG_cNS_up': 1,
                'renormWeight_isr_X2XG_cNS_up': 1,
                'renormWeight_fsr_G2GG_muR_down': 1,
                'renormWeight_fsr_G2QQ_muR_down': 1,
                'renormWeight_fsr_Q2QG_muR_down': 1,
                'renormWeight_fsr_X2XG_muR_down': 1,
                'renormWeight_fsr_G2GG_cNS_down': 1,
                'renormWeight_fsr_G2QQ_cNS_down': 1,
                'renormWeight_fsr_Q2QG_cNS_down': 1,
                'renormWeight_fsr_X2XG_cNS_down': 1,
                'renormWeight_isr_G2GG_muR_down': 1,
                'renormWeight_isr_G2QQ_muR_down': 1,
                'renormWeight_isr_Q2QG_muR_down': 1,
                'renormWeight_isr_X2XG_muR_down': 1,
                'renormWeight_isr_G2GG_cNS_down': 1,
                'renormWeight_isr_G2QQ_cNS_down': 1,
                'renormWeight_isr_Q2QG_cNS_down': 1,
                'renormWeight_isr_X2XG_cNS_down': 1,
                }
            )

        weights_ = None
        yieldSum_ = None
        yieldSum_nom_ = None
        yield_ = None
        yield_nom_ = None

        if self.file_key:
            # find ttbar samples
            if self.is4FS or self.isttbar or self.isDPS or self.is2muF:
                procs_ = ["ttlf", "ttcj", "ttcc"]
                if self.is4FS:
                    procs_.append("ttbj4fs")
                    procs_.append("ttbb4fs")
                elif self.isDPS:
                    procs_.append("ttbjdps")
                    procs_.append("ttbbdps")
                elif self.is2muF:
                    procs_.append("ttbj4fs2muF")
                    procs_.append("ttbb4fs2muF")
                else:
                    procs_.append("ttbj5fs")
                    procs_.append("ttbb5fs")

                if ((event.tt_category==0) and (event.higgs_decay==0) and (event.z_decay==0) and (event.genZ_pt < 0.)):
                    proc_key = "ttlf"
                elif (((event.genEventClassifier>=4) and (event.genEventClassifier<=5)) and (event.higgs_decay==0) and (event.z_decay==0) and (event.genZ_pt < 0.)):
                    proc_key = "ttcj"
                elif ((event.genEventClassifier==6) and (event.higgs_decay==0) and (event.z_decay==0) and (event.genZ_pt < 0.)):
                    proc_key = "ttcc"
                elif (((event.genEventClassifier>=7) and (event.genEventClassifier<=8)) and (event.higgs_decay==0) and (event.z_decay==0) and (event.genZ_pt < 0.)):
                    if self.is4FS:
                        proc_key = "ttbj4fs"
                    elif self.is2muF:
                        proc_key = "ttbj4fs2muF"
                    elif self.isDPS:
                        proc_key = "ttbjdps"
                    elif self.isttbar:
                        proc_key = "ttbj5fs"
                elif ((event.genEventClassifier==9) and (event.higgs_decay==0) and (event.z_decay==0) and (event.genZ_pt < 0.)):
                    if self.is4FS:
                        proc_key = "ttbb4fs"
                    elif self.is2muF:
                        proc_key = "ttbb4fs2muF"
                    elif self.isDPS:
                        proc_key = "ttbbdps"
                    elif self.isttbar:
                        proc_key = "ttbb5fs"

                weights_ = {label: self.corr["normalization"].evaluate(self.file_key, proc_key, label) for label in self.labels_}
                if self.isHdampUp or self.isHdampDown or self.isFxFx or self.isHerwig:
                    yieldSum_ = np.sum(np.array([self.corr["entries"].evaluate(self.file_key, proc, "nom") for proc in procs_]))
                    yieldSum_nom_ = np.sum(np.array([self.corr["entries"].evaluate(self.file_key_nom, proc, "nom") for proc in procs_]))
                    yield_ = self.corr["entries"].evaluate(self.file_key, proc_key, "nom")
                    yield_nom_ = self.corr["entries"].evaluate(self.file_key_nom, proc_key, "nom")

            elif self.isttz:
                if ((event.z_decay==5) and (event.higgs_decay==0) and (event.genZ_pt > 0.)):
                        weights_ = {label: self.corr["normalization"].evaluate(self.file_key, "ttzbb", label) for label in self.labels_}
                elif ((event.z_decay==4) and (event.higgs_decay==0) and (event.genZ_pt > 0.)):
                        weights_ = {label: self.corr["normalization"].evaluate(self.file_key, "ttzcc", label) for label in self.labels_}

        # for standard variations just get the renormWeight
        if weights_ is not None:
            for label in self.labels_:
                if label != "nom":
                    wgts["renormWeight_"+label] = weights_[label]

        if yieldSum_ is not None:
            if self.isHdampUp:
                wgts["renormWeight_hdamp_up"] = (yield_nom_/yieldSum_nom_)/(yield_/yieldSum_)
            if self.isHdampDown:
                wgts["renormWeight_hdamp_down"] = (yield_nom_/yieldSum_nom_)/(yield_/yieldSum_)
            if self.isFxFx:
                wgts["renormWeight_fxfx"] = (yield_nom_/yieldSum_nom_)/(yield_/yieldSum_)
            if self.isHerwig:
                wgts["renormWeight_herwig"] = (yield_nom_/yieldSum_nom_)/(yield_/yieldSum_)

        for k, v in wgts.items():
            self.out.fillBranch(k, v)

        return True


# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
def renormWeighter():
    return RenormWeightSFProducer(0)
