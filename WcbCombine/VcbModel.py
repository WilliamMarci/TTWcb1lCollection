from HiggsAnalysis.CombinedLimit.PhysicsModel import *
import ROOT


class VcbModel(PhysicsModel):
    def __init__(self):
        PhysicsModel.__init__(self)

        # Default value:
        # r = 0.5 * |Vcb|^2 ~ 0.5 * (0.041)^2 = 0.0008405
        self.r_value = 0.00084

        # Default POI setting:
        #   "mu"                -> lambda_cal is floating but not POI
        #   "lambda_cal"        -> mu is floating but not POI
        #   "mu,lambda_cal"     -> 2D fit
        self.poi_mode = "mu,lambda_cal"

        # Optional debug print
        self.verbose = False

        # Explicit process naming
        self.sig_process = "sig"
        self.wqq_process = "bkg_wqq"
        self.topbc_process = "bkg_topbc"
        self.other_process = "bkg_other"

    def setPhysicsOptions(self, physOptions):
        """
        Supported options:
          --PO r=0.00084
          --PO poi=mu
          --PO poi=lambda_cal
          --PO poi=mu,lambda_cal
          --PO verbose
          --PO sig=sig
          --PO wqq=bkg_wqq
          --PO topbc=bkg_topbc
          --PO other=bkg_other
        """
        for po in physOptions:
            if po.startswith("r="):
                self.r_value = float(po.replace("r=", ""))
            elif po.startswith("poi="):
                self.poi_mode = po.replace("poi=", "").strip()
            elif po == "verbose":
                self.verbose = True
            elif po.startswith("sig="):
                self.sig_process = po.replace("sig=", "")
            elif po.startswith("wqq="):
                self.wqq_process = po.replace("wqq=", "")
            elif po.startswith("topbc="):
                self.topbc_process = po.replace("topbc=", "")
            elif po.startswith("other="):
                self.other_process = po.replace("other=", "")
            else:
                raise RuntimeError(f"Unknown physics option: {po}")

    def doParametersOfInterest(self):
        # Main parameters
        self.modelBuilder.doVar("mu[1,0,2]")
        self.modelBuilder.doVar("lambda_cal[1,0.5,2.0]")
        self.modelBuilder.doVar(f"r[{self.r_value}]")

        # r is fixed external physics input
        self.modelBuilder.out.var("r").setConstant(True)

        # Define POI set
        allowed = ["mu", "lambda_cal", "mu,lambda_cal", "lambda_cal,mu"]
        if self.poi_mode not in allowed:
            raise RuntimeError(
                f"Unsupported poi mode '{self.poi_mode}'. "
                f"Allowed: 'mu', 'lambda_cal', or 'mu,lambda_cal'"
            )

        if self.poi_mode == "mu":
            self.modelBuilder.doSet("POI", "mu")
        elif self.poi_mode == "lambda_cal":
            self.modelBuilder.doSet("POI", "lambda_cal")
        else:
            self.modelBuilder.doSet("POI", "mu,lambda_cal")

        # Scaling functions
        # signal: mu * lambda_cal
        self.modelBuilder.factory_('prod::scale_sig_total(mu,lambda_cal)')

        # wqq background: (1 - mu*r)/(1 - r)
        self.modelBuilder.factory_(
            'expr::scale_wqq("(1-@0*@1)/(1-@1)", mu, r)'
        )

        # topbc background: lambda_cal
        self.modelBuilder.factory_(
            'expr::scale_topbc("@0", lambda_cal)'
        )

        if self.verbose:
            print("== VcbModel configuration ==")
            print(f"  r                = {self.r_value}")
            print(f"  poi_mode         = {self.poi_mode}")
            print(f"  sig_process      = {self.sig_process}")
            print(f"  wqq_process      = {self.wqq_process}")
            print(f"  topbc_process    = {self.topbc_process}")
            print(f"  other_process    = {self.other_process}")
            print("  scale(sig)       = mu * lambda_cal")
            print("  scale(bkg_topbc) = lambda_cal")
            print("  scale(bkg_wqq)   = (1 - mu*r)/(1-r)")
            print("  scale(bkg_other) = 1")

    def getYieldScale(self, bin, process):
        if self.verbose:
            print(f"[VcbModel] getYieldScale called for bin={bin}, process={process}")

        # signal -> mu * lambda_cal
        if process == self.sig_process:
            return "scale_sig_total"

        # in-situ calibration component -> lambda_cal
        if process == self.topbc_process:
            return "scale_topbc"

        # wqq background normalization induced by branching relation
        if process == self.wqq_process:
            return "scale_wqq"

        # other backgrounds unchanged
        if process == self.other_process:
            return 1

        # any additional process left untouched
        return 1


vcbModel = VcbModel()
