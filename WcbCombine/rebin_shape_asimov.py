#!/usr/bin/env python3
import os
import json
import math
import argparse
from array import array

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)
ROOT.gStyle.SetOptStat(0)

REGIONS = ["calib", "sr"]
MC_PROCS = ["sig", "bkg_wqq", "bkg_topbc", "bkg_other"]

COLORS = {
    "sig": ROOT.kRed + 1,
    "bkg_wqq": ROOT.kAzure + 1,
    "bkg_topbc": ROOT.kGreen + 2,
    "bkg_other": ROOT.kMagenta + 1,
    "total_bkg": ROOT.kGray + 2,
    "total_sb": ROOT.kOrange + 7,
}

PRESET_BINNINGS = {
    "equal_6": [0.0, 1/6, 2/6, 3/6, 4/6, 5/6, 1.0],
    "equal_8": [0.0, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 1.0],
    "tail_7a": [0.0, 0.35, 0.55, 0.70, 0.82, 0.90, 0.95, 1.0],
    "tail_7b": [0.0, 0.30, 0.50, 0.65, 0.78, 0.87, 0.94, 1.0],
    "origin": [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0],
}

DEFAULT_PLOT_SYSTS = [
    "PU",
    "L1PreFiring",
    "TrigEff",
    "ElEff",
    "MuEff",
    "Theory_muR",
    "Theory_muF",
    "Theory_isr",
    "Theory_fsr",
    "FT_Stat",
]

# ---------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def sanitize_hist(h):
    for i in range(1, h.GetNbinsX() + 1):
        c = h.GetBinContent(i)
        e = h.GetBinError(i)
        if (not math.isfinite(c)) or c < 0:
            h.SetBinContent(i, 0.0)
        if (not math.isfinite(e)) or h.GetBinContent(i) <= 0:
            h.SetBinError(i, 0.0)

def integral(h):
    return h.Integral(1, h.GetNbinsX())

def max_rel_stat(h):
    vals = []
    for i in range(1, h.GetNbinsX() + 1):
        c = h.GetBinContent(i)
        e = h.GetBinError(i)
        if c > 0:
            vals.append(e / c)
    return max(vals) if vals else 0.0

def get_hist_bin_edges(h):
    xaxis = h.GetXaxis()
    edges = [xaxis.GetBinLowEdge(1)]
    for i in range(1, h.GetNbinsX() + 1):
        edges.append(xaxis.GetBinUpEdge(i))
    return edges

def snap_edges_to_histogram(edges, ref_hist, tol=1e-9, verbose=True):
    old_edges = get_hist_bin_edges(ref_hist)

    snapped = []
    for x in edges:
        best = min(old_edges, key=lambda e: abs(e - x))
        snapped.append(best)

        if verbose and abs(best - x) > tol:
            print(f"[snap] requested edge {x:.6f} -> snapped to {best:.6f}")

    snapped = sorted(set(round(x, 10) for x in snapped))

    if len(snapped) < 2:
        raise RuntimeError(f"Snapped edges are invalid: {snapped}")

    for i in range(len(snapped) - 1):
        if snapped[i+1] <= snapped[i]:
            raise RuntimeError(f"Non-increasing snapped edges: {snapped}")

    return snapped

def list_hists_in_dir(tdir):
    names = []
    for key in tdir.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom("TH1"):
            names.append(obj.GetName())
    return sorted(names)

def load_root_hists(fin):
    out = {}
    for reg in REGIONS:
        tdir = fin.Get(reg)
        if not tdir:
            raise RuntimeError(f"Missing directory '{reg}' in input root")
        out[reg] = {}
        for name in list_hists_in_dir(tdir):
            h = tdir.Get(name)
            if not h:
                continue
            h.SetDirectory(0)
            out[reg][name] = h
    return out

def parse_custom_edges(s):
    edges = [float(x) for x in s.split(",")]
    if len(edges) < 2:
        raise ValueError("Need at least two edges")
    for i in range(len(edges) - 1):
        if edges[i+1] <= edges[i]:
            raise ValueError("Edges must be strictly increasing")
    return edges

def insert_cv_cut(edges, cv_cut, tol=1e-9):
    new_edges = list(edges)
    if not any(abs(x - cv_cut) < tol for x in new_edges):
        new_edges.append(cv_cut)
    new_edges = sorted(set(round(x, 10) for x in new_edges))
    for i in range(len(new_edges) - 1):
        if new_edges[i+1] <= new_edges[i]:
            raise RuntimeError(f"Malformed edges after inserting cv cut: {new_edges}")
    return new_edges

def finalize_edges(edges, cv_cut, ref_hist, verbose=True):
    edges = insert_cv_cut(edges, cv_cut)
    edges = snap_edges_to_histogram(edges, ref_hist, verbose=verbose)
    return edges

def rebin_hist(h, edges, new_name=None):
    arr = array("d", edges)
    hnew = h.Rebin(len(edges) - 1, new_name if new_name else h.GetName(), arr)
    hnew.SetDirectory(0)
    sanitize_hist(hnew)
    return hnew

# ---------------------------------------------------------
# Auto binning
# ---------------------------------------------------------
def build_signal_equalized_edges(hsig, nbins):
    total = integral(hsig)
    if total <= 0:
        raise RuntimeError("Signal histogram has zero total yield, cannot build signal-equalized binning")

    xaxis = hsig.GetXaxis()
    edges = [xaxis.GetXmin()]
    targets = [total * i / nbins for i in range(1, nbins)]
    running = 0.0
    t_idx = 0

    for ib in range(1, hsig.GetNbinsX() + 1):
        running += hsig.GetBinContent(ib)
        while t_idx < len(targets) and running >= targets[t_idx]:
            edges.append(xaxis.GetBinUpEdge(ib))
            t_idx += 1

    xmax = xaxis.GetXmax()
    if edges[-1] != xmax:
        edges.append(xmax)

    clean = [edges[0]]
    for x in edges[1:]:
        if x > clean[-1]:
            clean.append(x)

    if len(clean) < nbins + 1:
        fine_edges = [xaxis.GetBinLowEdge(1)]
        for i in range(1, hsig.GetNbinsX() + 1):
            fine_edges.append(xaxis.GetBinUpEdge(i))
        for x in fine_edges:
            if x not in clean:
                clean.append(x)
            clean = sorted(set(clean))
            if len(clean) >= nbins + 1:
                break

    if len(clean) < nbins + 1:
        raise RuntimeError(f"Could not construct enough edges for signal-equalized {nbins} bins")

    clean = clean[:nbins] + [xmax]
    clean = sorted(set(clean))

    if len(clean) != nbins + 1:
        raise RuntimeError(f"Final signal-equalized edges malformed: {clean}")

    return clean

def build_scheme_edges(name, all_hists, region_for_auto="sr", proc_for_auto="sig"):
    if name in PRESET_BINNINGS:
        return PRESET_BINNINGS[name]

    if name.startswith("signal_eq_"):
        nbins = int(name.split("_")[-1])
        hsig = all_hists[region_for_auto][proc_for_auto]
        return build_signal_equalized_edges(hsig, nbins)

    raise ValueError(f"Unknown binning scheme: {name}")

# ---------------------------------------------------------
# Write rebinned ROOT
# ---------------------------------------------------------
def write_rebinned_root(all_hists, out_root, edges):
    fout = ROOT.TFile(out_root, "RECREATE")
    rebinned = {}

    for reg in REGIONS:
        fout.mkdir(reg)
        fout.cd(reg)
        rebinned[reg] = {}

        for hname, h in all_hists[reg].items():
            hnew = rebin_hist(h, edges, hname)
            hnew.Write(hname)
            rebinned[reg][hname] = hnew

        fout.cd()

    fout.Close()
    return rebinned

# ---------------------------------------------------------
# Totals / systematics
# ---------------------------------------------------------
def build_total_hist(region_hists, include_signal=False):
    procs = ["bkg_wqq", "bkg_topbc", "bkg_other"]
    if include_signal:
        procs = ["sig"] + procs

    htot = None
    for proc in procs:
        if proc not in region_hists:
            continue
        h = region_hists[proc]
        if htot is None:
            htot = h.Clone(f"total_{'sb' if include_signal else 'bkg'}")
            htot.SetDirectory(0)
        else:
            htot.Add(h)
    return htot

def collect_syst_bases(region_hists, proc):
    bases = set()
    prefix = proc + "_"
    for hname in region_hists.keys():
        if not hname.startswith(prefix):
            continue
        tail = hname[len(prefix):]
        if tail.endswith("Up"):
            bases.add(tail[:-2])
        elif tail.endswith("Down"):
            bases.add(tail[:-4])
    return bases

def build_total_syst_unc(region_hists, include_signal=False, allowed_bases=None, verbose=False):
    procs = ["bkg_wqq", "bkg_topbc", "bkg_other"]
    if include_signal:
        procs = ["sig"] + procs

    hnom = build_total_hist(region_hists, include_signal=include_signal)
    if hnom is None:
        return None, None

    nb = hnom.GetNbinsX()

    all_bases = set()
    for proc in procs:
        all_bases |= collect_syst_bases(region_hists, proc)

    if allowed_bases is not None:
        all_bases = {b for b in all_bases if b in allowed_bases}

    syst2 = [0.0] * (nb + 1)

    for base in sorted(all_bases):
        hup = None
        hdn = None

        for proc in procs:
            hnom_proc = region_hists.get(proc, None)
            if hnom_proc is None:
                continue

            hup_proc = region_hists.get(f"{proc}_{base}Up", None)
            hdn_proc = region_hists.get(f"{proc}_{base}Down", None)

            # 如果这个 proc 没有这个 nuisance，就跳过，而不是强行加 nominal
            if hup_proc is None and hdn_proc is None:
                if verbose:
                    print(f"[skip] proc={proc}, base={base} : no variation hist")
                continue

            if hup is None:
                hup = hnom_proc.Clone(f"tmp_up_{base}")
                hup.Reset("ICES")
                hup.SetDirectory(0)

            if hdn is None:
                hdn = hnom_proc.Clone(f"tmp_dn_{base}")
                hdn.Reset("ICES")
                hdn.SetDirectory(0)

            if hup_proc is not None:
                hup.Add(hup_proc)
            else:
                hup.Add(hnom_proc)

            if hdn_proc is not None:
                hdn.Add(hdn_proc)
            else:
                hdn.Add(hnom_proc)

        if hup is None and hdn is None:
            continue

        max_dev = 0.0
        for i in range(1, nb + 1):
            nom = hnom.GetBinContent(i)
            du = abs(hup.GetBinContent(i) - nom) if hup else 0.0
            dd = abs(hdn.GetBinContent(i) - nom) if hdn else 0.0
            syst = max(du, dd)
            syst2[i] += syst * syst
            max_dev = max(max_dev, syst)

        if verbose:
            print(f"[band syst] {base:25s} max_abs_dev = {max_dev:.6f}")

    return hnom, [math.sqrt(x) for x in syst2]

# ---------------------------------------------------------
# Display mapping helpers
# ---------------------------------------------------------
def get_display_edges_from_hist(h, xmin_new, xmax_new):
    old_edges = get_hist_bin_edges(h)
    old_min = old_edges[0]
    old_max = old_edges[-1]

    if abs(old_max - old_min) < 1e-12:
        raise RuntimeError("Histogram has zero x-range")

    disp_edges = []
    for x in old_edges:
        y = xmin_new + (x - old_min) * (xmax_new - xmin_new) / (old_max - old_min)
        disp_edges.append(y)
    return disp_edges

def remap_hist_to_display(h, xmin_new, xmax_new, new_name):
    disp_edges = get_display_edges_from_hist(h, xmin_new, xmax_new)
    arr = array("d", disp_edges)
    hnew = ROOT.TH1F(new_name, new_name, len(disp_edges) - 1, arr)
    hnew.Sumw2()
    hnew.SetDirectory(0)
    hnew.SetStats(0)

    for i in range(1, h.GetNbinsX() + 1):
        hnew.SetBinContent(i, h.GetBinContent(i))
        hnew.SetBinError(i, h.GetBinError(i))
    return hnew

def make_band_from_hist(h, name, color=ROOT.kGray + 1, fillstyle=3004, alpha=0.30):
    g = ROOT.TGraphAsymmErrors(h.GetNbinsX())
    for i in range(1, h.GetNbinsX() + 1):
        x = h.GetBinCenter(i)
        ex = 0.5 * h.GetBinWidth(i)
        y = h.GetBinContent(i)
        ey = h.GetBinError(i)
        g.SetPoint(i - 1, x, y)
        g.SetPointError(i - 1, ex, ex, ey, ey)

    g.SetName(name)
    g.SetFillColorAlpha(color, alpha)
    g.SetLineColor(color)
    g.SetFillStyle(fillstyle)
    return g

def make_total_unc_band(hnom, syst_unc, name, color=ROOT.kRed - 9, fillstyle=1001, alpha=0.18):
    g = ROOT.TGraphAsymmErrors(hnom.GetNbinsX())
    for i in range(1, hnom.GetNbinsX() + 1):
        x = hnom.GetBinCenter(i)
        ex = 0.5 * hnom.GetBinWidth(i)
        y = hnom.GetBinContent(i)
        estat = hnom.GetBinError(i)
        esyst = syst_unc[i]
        etot = math.sqrt(estat * estat + esyst * esyst)
        g.SetPoint(i - 1, x, y)
        g.SetPointError(i - 1, ex, ex, etot, etot)

    g.SetName(name)
    g.SetFillColorAlpha(color, alpha)
    g.SetLineColor(color)
    g.SetFillStyle(fillstyle)
    return g

# ---------------------------------------------------------
# Normal stack plots
# ---------------------------------------------------------
def make_stack_plot(region, region_hists, outpath, title=""):
    c = ROOT.TCanvas(f"c_{region}", "", 800, 800)
    p1 = ROOT.TPad("p1", "", 0.0, 0.30, 1.0, 1.0)
    p2 = ROOT.TPad("p2", "", 0.0, 0.00, 1.0, 0.30)
    p1.SetBottomMargin(0.02)
    p2.SetTopMargin(0.03)
    p2.SetBottomMargin(0.30)
    p1.Draw()
    p2.Draw()

    hs = ROOT.THStack(f"hs_{region}", title if title else region)
    leg = ROOT.TLegend(0.60, 0.53, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)

    draw_order = ["bkg_other", "bkg_topbc", "bkg_wqq", "sig"]
    total = None

    for proc in draw_order:
        if proc not in region_hists:
            continue
        h = region_hists[proc].Clone(f"{region}_{proc}_draw")
        h.SetDirectory(0)
        h.SetFillColor(COLORS[proc])
        h.SetLineColor(ROOT.kBlack)
        hs.Add(h)
        leg.AddEntry(h, proc, "f")

        if total is None:
            total = h.Clone(f"{region}_total")
            total.SetDirectory(0)
        else:
            total.Add(h)

    data = region_hists.get("data_obs", None)
    if data:
        data = data.Clone(f"{region}_data_draw")
        data.SetDirectory(0)
        data.SetMarkerStyle(20)
        data.SetLineColor(ROOT.kBlack)
        leg.AddEntry(data, "data_obs", "lep")

    p1.cd()
    hs.Draw("HIST")
    hs.GetYaxis().SetTitle("Events")
    hs.GetXaxis().SetLabelSize(0)
    ymax = hs.GetMaximum()
    if data:
        ymax = max(ymax, data.GetMaximum())
    hs.SetMaximum(max(1.0, ymax) * 1.6)

    if total:
        total_line = total.Clone(f"{region}_total_line")
        total_line.SetDirectory(0)
        total_line.SetFillStyle(0)
        total_line.SetLineWidth(2)
        total_line.SetLineColor(ROOT.kBlack)
        total_line.Draw("HIST SAME")

    if data:
        data.Draw("E1 SAME")

    leg.Draw()

    p2.cd()
    if data and total:
        ratio = data.Clone(f"{region}_ratio")
        ratio.SetDirectory(0)
        ratio.Divide(total)
        ratio.SetMarkerStyle(20)
        ratio.SetLineColor(ROOT.kBlack)
        ratio.GetYaxis().SetTitle("Data/MC")
        ratio.GetXaxis().SetTitle("CV score")
        ratio.GetYaxis().SetNdivisions(505)
        ratio.GetYaxis().SetTitleSize(0.10)
        ratio.GetYaxis().SetTitleOffset(0.45)
        ratio.GetYaxis().SetLabelSize(0.08)
        ratio.GetXaxis().SetTitleSize(0.10)
        ratio.GetXaxis().SetLabelSize(0.08)
        ratio.SetMinimum(0.4)
        ratio.SetMaximum(1.6)
        ratio.Draw("E1")
    else:
        frame = ROOT.TH1F(f"frame_{region}", "", region_hists["sig"].GetNbinsX(), 0, region_hists["sig"].GetNbinsX())
        frame.SetStats(0)
        frame.GetYaxis().SetTitle("Data/MC")
        frame.GetXaxis().SetTitle("CV score")
        frame.SetMinimum(0.4)
        frame.SetMaximum(1.6)
        frame.Draw()

    c.SaveAs(outpath + ".png")
    c.SaveAs(outpath + ".pdf")
    c.Close()

# ---------------------------------------------------------
# Syst variation plots
# ---------------------------------------------------------
def make_syst_plot(region, proc, syst_base, region_hists, outpath):
    hnom = region_hists.get(proc, None)
    hup = region_hists.get(f"{proc}_{syst_base}Up", None)
    hdn = region_hists.get(f"{proc}_{syst_base}Down", None)

    if hnom is None:
        return
    if hup is None and hdn is None:
        return

    c = ROOT.TCanvas(f"c_{region}_{proc}_{syst_base}", "", 800, 800)
    p1 = ROOT.TPad("p1", "", 0.0, 0.30, 1.0, 1.0)
    p2 = ROOT.TPad("p2", "", 0.0, 0.00, 1.0, 0.30)
    p1.SetBottomMargin(0.02)
    p2.SetTopMargin(0.03)
    p2.SetBottomMargin(0.30)
    p1.Draw()
    p2.Draw()

    p1.cd()
    nominal = hnom.Clone("nominal")
    nominal.SetDirectory(0)
    nominal.SetLineColor(ROOT.kBlack)
    nominal.SetLineWidth(2)
    nominal.SetTitle(f"{region} {proc} {syst_base}")
    nominal.GetYaxis().SetTitle("Events")
    nominal.GetXaxis().SetLabelSize(0)

    ymax = nominal.GetMaximum()
    if hup:
        ymax = max(ymax, hup.GetMaximum())
    if hdn:
        ymax = max(ymax, hdn.GetMaximum())
    nominal.SetMaximum(max(1.0, ymax) * 1.5)
    nominal.Draw("HIST")

    leg = ROOT.TLegend(0.62, 0.62, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(nominal, "nominal", "l")

    up_draw = None
    dn_draw = None

    if hup:
        up_draw = hup.Clone("up_draw")
        up_draw.SetDirectory(0)
        up_draw.SetLineColor(ROOT.kRed)
        up_draw.SetLineWidth(2)
        up_draw.Draw("HIST SAME")
        leg.AddEntry(up_draw, "Up", "l")

    if hdn:
        dn_draw = hdn.Clone("dn_draw")
        dn_draw.SetDirectory(0)
        dn_draw.SetLineColor(ROOT.kBlue)
        dn_draw.SetLineWidth(2)
        dn_draw.Draw("HIST SAME")
        leg.AddEntry(dn_draw, "Down", "l")

    leg.Draw()

    p2.cd()
    frame = nominal.Clone("ratio_frame")
    frame.Reset()
    for i in range(1, frame.GetNbinsX() + 1):
        frame.SetBinContent(i, 1.0)
    frame.GetYaxis().SetTitle("Var/Nom")
    frame.GetXaxis().SetTitle("CV score")
    frame.GetYaxis().SetNdivisions(505)
    frame.GetYaxis().SetTitleSize(0.10)
    frame.GetYaxis().SetTitleOffset(0.45)
    frame.GetYaxis().SetLabelSize(0.08)
    frame.GetXaxis().SetTitleSize(0.10)
    frame.GetXaxis().SetLabelSize(0.08)
    frame.SetMinimum(0.5)
    frame.SetMaximum(1.5)
    frame.Draw("HIST")

    if up_draw:
        rup = up_draw.Clone("rup")
        rup.SetDirectory(0)
        rup.Divide(nominal)
        rup.SetLineColor(ROOT.kRed)
        rup.SetLineWidth(2)
        rup.Draw("HIST SAME")

    if dn_draw:
        rdn = dn_draw.Clone("rdn")
        rdn.SetDirectory(0)
        rdn.Divide(nominal)
        rdn.SetLineColor(ROOT.kBlue)
        rdn.SetLineWidth(2)
        rdn.Draw("HIST SAME")

    c.SaveAs(outpath + ".png")
    c.SaveAs(outpath + ".pdf")
    c.Close()

# ---------------------------------------------------------
# Combined CV-score plotting
# ---------------------------------------------------------
def make_combined_cv_plot(calib_hists, sr_hists, cv_cut, outpath):
    c = ROOT.TCanvas("c_combined", "", 1200, 800)
    c.SetLogy()
    c.SetMargin(0.10, 0.04, 0.12, 0.08)
    allowed_band_systs = {
    "PU",
    "L1PreFiring",
    "TrigEff",
    "ElEff",
    "MuEff",
    "Theory_muR",
    "Theory_muF",
    "Theory_isr",
    "Theory_fsr",
    "FT_Stat",
    }

    # compute y range from all visible hists
    ymax = 0.0
    ymin_pos = 1e30

    for hdict in [calib_hists, sr_hists]:
        for hname, h in hdict.items():
            if not h or not h.InheritsFrom("TH1"):
                continue
            for b in range(1, h.GetNbinsX() + 1):
                y = h.GetBinContent(b)
                if y > ymax:
                    ymax = y
                if y > 0 and y < ymin_pos:
                    ymin_pos = y

    ymin = max(1e-3, ymin_pos * 0.5 if ymin_pos < 1e30 else 1e-3)
    ymax = max(10.0, ymax * 30.0)

    frame = ROOT.TH1D("frame_combined", "", 100, 0.0, 1.0)
    frame.SetStats(0)
    frame.SetMinimum(ymin)
    frame.SetMaximum(ymax)
    frame.GetXaxis().SetTitle("class_score")
    frame.GetYaxis().SetTitle("Events / bin")
    frame.GetXaxis().SetTitleSize(0.045)
    frame.GetYaxis().SetTitleSize(0.045)
    frame.GetXaxis().SetLabelSize(0.035)
    frame.GetYaxis().SetLabelSize(0.035)
    frame.Draw()

    def style_line(h, color, width=3, linestyle=1):
        h.SetLineColor(color)
        h.SetLineWidth(width)
        h.SetLineStyle(linestyle)
        h.SetFillStyle(0)
        h.SetMarkerSize(0)

    def style_data(h, marker=20):
        h.SetMarkerStyle(marker)
        h.SetMarkerSize(0.9)
        h.SetMarkerColor(ROOT.kBlack)
        h.SetLineColor(ROOT.kBlack)
        h.SetLineWidth(1)

    # build total hists
    hcal_bkg = build_total_hist(calib_hists, include_signal=False)
    hcal_sb  = build_total_hist(calib_hists, include_signal=True)
    hsr_bkg  = build_total_hist(sr_hists, include_signal=False)
    hsr_sb   = build_total_hist(sr_hists, include_signal=True)

    # uncertainty bands for total_bkg
    hnom_cal_bkg, syst_cal_bkg = build_total_syst_unc(
    calib_hists,
    include_signal=False,
    allowed_bases=allowed_band_systs,
    verbose=False
    )

    if hnom_cal_bkg:
        htmp = hnom_cal_bkg.Clone("hcal_bkg_unc")
        htmp.SetDirectory(0)
        band_cal_tot = make_total_unc_band(
            htmp, syst_cal_bkg, "band_cal_tot",
            color=ROOT.kRed - 9, fillstyle=1001, alpha=0.18
        )
        band_cal_stat = make_band_from_hist(
            htmp, "band_cal_stat",
            color=ROOT.kGray + 1, fillstyle=3004, alpha=0.30
        )
    else:
        band_cal_tot = None
        band_cal_stat = None

    hnom_sr_bkg, syst_sr_bkg = build_total_syst_unc(
    sr_hists,
    include_signal=False,
    allowed_bases=allowed_band_systs,
    verbose=False
    )

    if hnom_sr_bkg:
        htmp = hnom_sr_bkg.Clone("hsr_bkg_unc")
        htmp.SetDirectory(0)
        band_sr_tot = make_total_unc_band(
            htmp, syst_sr_bkg, "band_sr_tot",
            color=ROOT.kRed - 9, fillstyle=1001, alpha=0.18
        )
        band_sr_stat = make_band_from_hist(
            htmp, "band_sr_stat",
            color=ROOT.kGray + 1, fillstyle=3004, alpha=0.30
        )
    else:
        band_sr_tot = None
        band_sr_stat = None

    # style nominal hists
    draw_map = {
        "sig": COLORS["sig"],
        "bkg_wqq": COLORS["bkg_wqq"],
        "bkg_topbc": COLORS["bkg_topbc"],
        "bkg_other": COLORS["bkg_other"],
    }

    calib_draw = {}
    sr_draw = {}

    for proc, color in draw_map.items():
        if proc in calib_hists:
            h = calib_hists[proc].Clone(f"cal_{proc}_draw")
            h.SetDirectory(0)
            style_line(h, color, width=3, linestyle=1)
            calib_draw[proc] = h

        if proc in sr_hists:
            h = sr_hists[proc].Clone(f"sr_{proc}_draw")
            h.SetDirectory(0)
            style_line(h, color, width=3, linestyle=2)
            sr_draw[proc] = h

    if hcal_bkg:
        hcal_bkg = hcal_bkg.Clone("hcal_bkg_draw")
        hcal_bkg.SetDirectory(0)
        style_line(hcal_bkg, COLORS["total_bkg"], width=3, linestyle=1)

    if hcal_sb:
        hcal_sb = hcal_sb.Clone("hcal_sb_draw")
        hcal_sb.SetDirectory(0)
        style_line(hcal_sb, COLORS["total_sb"], width=3, linestyle=1)

    if hsr_bkg:
        hsr_bkg = hsr_bkg.Clone("hsr_bkg_draw")
        hsr_bkg.SetDirectory(0)
        style_line(hsr_bkg, COLORS["total_bkg"], width=3, linestyle=2)

    if hsr_sb:
        hsr_sb = hsr_sb.Clone("hsr_sb_draw")
        hsr_sb.SetDirectory(0)
        style_line(hsr_sb, COLORS["total_sb"], width=3, linestyle=2)

    data_cal = None
    if "data_obs" in calib_hists:
        data_cal = calib_hists["data_obs"].Clone("data_cal_draw")
        data_cal.SetDirectory(0)
        style_data(data_cal, marker=20)

    data_sr = None
    if "data_obs" in sr_hists:
        data_sr = sr_hists["data_obs"].Clone("data_sr_draw")
        data_sr.SetDirectory(0)
        style_data(data_sr, marker=24)

    # draw order: uncertainty first, then lines, then data
    if band_cal_tot:
        band_cal_tot.Draw("2 SAME")
    if band_cal_stat:
        band_cal_stat.Draw("2 SAME")
    if band_sr_tot:
        band_sr_tot.Draw("2 SAME")
    if band_sr_stat:
        band_sr_stat.Draw("2 SAME")

    draw_order = ["total_sb", "total_bkg", "sig", "bkg_wqq", "bkg_topbc", "bkg_other"]

    for proc in draw_order:
        if proc == "total_sb" and hcal_sb:
            hcal_sb.Draw("HIST SAME")
        elif proc == "total_bkg" and hcal_bkg:
            hcal_bkg.Draw("HIST SAME")
        elif proc in calib_draw:
            calib_draw[proc].Draw("HIST SAME")

    for proc in draw_order:
        if proc == "total_sb" and hsr_sb:
            hsr_sb.Draw("HIST SAME")
        elif proc == "total_bkg" and hsr_bkg:
            hsr_bkg.Draw("HIST SAME")
        elif proc in sr_draw:
            sr_draw[proc].Draw("HIST SAME")

    # redraw total_bkg after band for visibility
    if hcal_bkg:
        hcal_bkg.Draw("HIST SAME")
    if hsr_bkg:
        hsr_bkg.Draw("HIST SAME")

    if data_cal:
        data_cal.Draw("E1 SAME")
    if data_sr:
        data_sr.Draw("E1 SAME")

    line = ROOT.TLine(cv_cut, ymin, cv_cut, ymax)
    line.SetLineStyle(2)
    line.SetLineWidth(3)
    line.SetLineColor(ROOT.kBlack)
    line.Draw()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.DrawLatex(0.22, 0.92, "calib region")
    latex.DrawLatex(0.68, 0.92, "sr region")

    calib_data_y = integral(calib_hists["data_obs"]) if "data_obs" in calib_hists else 0.0
    sr_data_y = integral(sr_hists["data_obs"]) if "data_obs" in sr_hists else 0.0
    latex.SetTextSize(0.032)
    latex.DrawLatex(0.12, 0.84, f"calib data = {calib_data_y:.2f}")
    latex.DrawLatex(0.58, 0.84, f"sr data = {sr_data_y:.2f}")

    leg = ROOT.TLegend(0.62, 0.46, 0.93, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)

    if "sig" in calib_draw:        leg.AddEntry(calib_draw["sig"], "sig (calib)", "l")
    if "bkg_wqq" in calib_draw:    leg.AddEntry(calib_draw["bkg_wqq"], "bkg_wqq (calib)", "l")
    if "bkg_topbc" in calib_draw:  leg.AddEntry(calib_draw["bkg_topbc"], "bkg_topbc (calib)", "l")
    if "bkg_other" in calib_draw:  leg.AddEntry(calib_draw["bkg_other"], "bkg_other (calib)", "l")
    if hcal_bkg:                   leg.AddEntry(hcal_bkg, "total_bkg (calib)", "l")
    if hcal_sb:                    leg.AddEntry(hcal_sb, "total_s+b (calib)", "l")
    if data_cal:                   leg.AddEntry(data_cal, "data_obs (calib)", "lep")

    if "sig" in sr_draw:           leg.AddEntry(sr_draw["sig"], "sig (sr)", "l")
    if "bkg_wqq" in sr_draw:       leg.AddEntry(sr_draw["bkg_wqq"], "bkg_wqq (sr)", "l")
    if "bkg_topbc" in sr_draw:     leg.AddEntry(sr_draw["bkg_topbc"], "bkg_topbc (sr)", "l")
    if "bkg_other" in sr_draw:     leg.AddEntry(sr_draw["bkg_other"], "bkg_other (sr)", "l")
    if hsr_bkg:                    leg.AddEntry(hsr_bkg, "total_bkg (sr)", "l")
    if hsr_sb:                     leg.AddEntry(hsr_sb, "total_s+b (sr)", "l")
    if data_sr:                    leg.AddEntry(data_sr, "data_obs (sr)", "lep")

    if band_cal_stat:
        leg.AddEntry(band_cal_stat, "stat unc.", "f")
    if band_cal_tot:
        leg.AddEntry(band_cal_tot, "stat+syst unc.", "f")

    leg.AddEntry(line, "region boundary", "l")
    leg.Draw()

    c.SaveAs(outpath + ".png")
    c.SaveAs(outpath + ".pdf")
    c.Close()

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
def save_summary(rebinned, edges, out_json):
    summary = {
        "edges": edges,
        "regions": {}
    }

    for reg in REGIONS:
        summary["regions"][reg] = {}
        for hname, h in rebinned[reg].items():
            summary["regions"][reg][hname] = {
                "yield": integral(h),
                "max_rel_stat": 0.0 if hname == "data_obs" else max_rel_stat(h),
                "nbins": h.GetNbinsX(),
                "bin_contents": [h.GetBinContent(i) for i in range(1, h.GetNbinsX() + 1)],
                "bin_errors": [h.GetBinError(i) for i in range(1, h.GetNbinsX() + 1)],
            }

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Rebin and plot combine shape ROOT files")
    parser.add_argument("-i", "--input", required=True, help="Input ROOT file from make_shape_asimov.py")
    parser.add_argument("-o", "--outdir", required=True, help="Output base directory")
    parser.add_argument("--cv-cut", type=float, required=True,
                        help="CV boundary used to separate calib and sr")

    parser.add_argument("--schemes", nargs="*", default=["equal_6", "equal_8", "tail_7a"],
                        help="Preset/auto schemes, e.g. equal_6 equal_8 tail_7a signal_eq_7")

    parser.add_argument("--custom-edges", default=None,
                        help="Custom bin edges, e.g. '0,0.2,0.4,0.6,0.8,1.0'")
    parser.add_argument("--custom-name", default="custom",
                        help="Name for the custom binning scheme")

    parser.add_argument("--plot-systs", nargs="*", default=DEFAULT_PLOT_SYSTS,
                        help="Systematics to plot, base names only")
    parser.add_argument("--auto-region", default="sr", choices=REGIONS,
                        help="Region used for auto binning like signal_eq_N")
    parser.add_argument("--auto-proc", default="sig",
                        help="Process used for auto binning like signal_eq_N")
    parser.add_argument("--quiet-snap", action="store_true",
                        help="Do not print edge snapping messages")

    args = parser.parse_args()

    ensure_dir(args.outdir)

    fin = ROOT.TFile.Open(args.input)
    if not fin or fin.IsZombie():
        raise RuntimeError(f"Cannot open input file: {args.input}")

    all_hists = load_root_hists(fin)

    if "sr" not in all_hists or "sig" not in all_hists["sr"]:
        raise RuntimeError("Need 'sr/sig' as reference histogram for edge snapping")

    ref_hist = all_hists["sr"]["sig"]

    scheme_map = {}

    for s in args.schemes:
        raw_edges = build_scheme_edges(
            s,
            all_hists,
            region_for_auto=args.auto_region,
            proc_for_auto=args.auto_proc
        )
        edges = finalize_edges(raw_edges, args.cv_cut, ref_hist, verbose=(not args.quiet_snap))
        scheme_map[s] = edges

    if args.custom_edges is not None:
        custom_raw = parse_custom_edges(args.custom_edges)
        custom_edges = finalize_edges(custom_raw, args.cv_cut, ref_hist, verbose=(not args.quiet_snap))
        scheme_map[args.custom_name] = custom_edges

    if len(scheme_map) == 0:
        raise RuntimeError("No binning scheme specified")

    print("=== Schemes to run ===")
    for k, v in scheme_map.items():
        print(f"  {k:15s}: {v}")

    for scheme_name, edges in scheme_map.items():
        print(f"\n[INFO] Running scheme: {scheme_name}")

        scheme_dir = os.path.join(args.outdir, scheme_name)
        plot_dir = os.path.join(scheme_dir, "plots")
        syst_dir = os.path.join(plot_dir, "systematics")
        ensure_dir(scheme_dir)
        ensure_dir(plot_dir)
        ensure_dir(syst_dir)

        out_root = os.path.join(scheme_dir, f"shapes_{scheme_name}.root")
        rebinned = write_rebinned_root(all_hists, out_root, edges)

        for reg in REGIONS:
            make_stack_plot(
                reg,
                rebinned[reg],
                os.path.join(plot_dir, f"{reg}_stack"),
                title=f"{scheme_name} : {reg}"
            )

            for proc in MC_PROCS:
                for syst in args.plot_systs:
                    make_syst_plot(
                        reg,
                        proc,
                        syst,
                        rebinned[reg],
                        os.path.join(syst_dir, f"{reg}_{proc}_{syst}")
                    )

        make_combined_cv_plot(
            rebinned["calib"],
            rebinned["sr"],
            args.cv_cut,
            os.path.join(plot_dir, "combined_cvscore")
        )

        save_summary(
            rebinned,
            edges,
            os.path.join(scheme_dir, "summary.json")
        )

        print(f"[DONE] {scheme_name}")
        print(f"       final edges = {edges}")
        print(f"       root        = {out_root}")

    fin.Close()
    print("\nAll done.")

if __name__ == "__main__":
    main()
