#! /usr/bin/env python3
# Checkshape.py
# Sanity check nominal / up / down histograms in a ROOT shape file.
# Supports:
#   1) single-region mode:   --region sr
#   2) combined-region mode: --combine-regions calib sr --cv 0.7

import ROOT
import argparse
import os
import sys

ROOT.gROOT.SetBatch(True)


def get_hist(f, path):
    h = f.Get(path)
    if not h:
        return None
    h = h.Clone(path.replace("/", "_"))
    h.SetDirectory(0)
    return h


def sum_hists(hist_list, out_name):
    valid = [h for h in hist_list if h is not None]
    if len(valid) == 0:
        return None
    hsum = valid[0].Clone(out_name)
    hsum.SetDirectory(0)
    for h in valid[1:]:
        hsum.Add(h)
    return hsum


def safe_ratio(num, den):
    h = num.Clone(num.GetName() + "_ratio")
    h.SetDirectory(0)
    for i in range(1, h.GetNbinsX() + 1):
        n = num.GetBinContent(i)
        d = den.GetBinContent(i)
        if d != 0:
            h.SetBinContent(i, n / d)
            h.SetBinError(i, 0.0)
        else:
            h.SetBinContent(i, 0.0)
            h.SetBinError(i, 0.0)
    return h


def get_triplet_single_region(f, region, process, systematic):
    nominal_path = f"{region}/{process}"
    up_path = f"{region}/{process}_{systematic}Up"
    down_path = f"{region}/{process}_{systematic}Down"

    h_nom = get_hist(f, nominal_path)
    h_up = get_hist(f, up_path)
    h_down = get_hist(f, down_path)

    missing = []
    if not h_nom:
        missing.append(nominal_path)
    if not h_up:
        missing.append(up_path)
    if not h_down:
        missing.append(down_path)

    return h_nom, h_up, h_down, missing


def get_triplet_combined_regions(f, regions, process, systematic):
    h_nom_list = []
    h_up_list = []
    h_down_list = []
    missing = []

    for region in regions:
        nominal_path = f"{region}/{process}"
        up_path = f"{region}/{process}_{systematic}Up"
        down_path = f"{region}/{process}_{systematic}Down"

        h_nom = get_hist(f, nominal_path)
        h_up = get_hist(f, up_path)
        h_down = get_hist(f, down_path)

        if not h_nom:
            missing.append(nominal_path)
        if not h_up:
            missing.append(up_path)
        if not h_down:
            missing.append(down_path)

        h_nom_list.append(h_nom)
        h_up_list.append(h_up)
        h_down_list.append(h_down)

    h_nom_sum = sum_hists(h_nom_list, f"{process}_nom_combined")
    h_up_sum = sum_hists(h_up_list, f"{process}_{systematic}Up_combined")
    h_down_sum = sum_hists(h_down_list, f"{process}_{systematic}Down_combined")

    return h_nom_sum, h_up_sum, h_down_sum, missing


def draw_vertical_line(x, ymin, ymax, color=ROOT.kGray + 2, style=2, width=2):
    line = ROOT.TLine(x, ymin, x, ymax)
    line.SetLineColor(color)
    line.SetLineStyle(style)
    line.SetLineWidth(width)
    line.Draw()
    return line


def draw_shape_comparison(file_name, region, combine_regions, cv, process, systematic, outdir, logy=False):
    f = ROOT.TFile.Open(file_name)
    if not f or f.IsZombie():
        print(f"[ERROR] Cannot open ROOT file: {file_name}")
        return 1

    combined_mode = combine_regions is not None and len(combine_regions) > 0

    if combined_mode:
        h_nom, h_up, h_down, missing = get_triplet_combined_regions(
            f=f,
            regions=combine_regions,
            process=process,
            systematic=systematic,
        )
        label_mode = "combined"
        region_label = " + ".join(combine_regions)
    else:
        if region is None:
            print("[ERROR] Must provide --region in single-region mode, or use --combine-regions")
            f.Close()
            return 2
        h_nom, h_up, h_down, missing = get_triplet_single_region(
            f=f,
            region=region,
            process=process,
            systematic=systematic,
        )
        label_mode = "single"
        region_label = region

    if missing:
        print("[ERROR] Histograms not found.")
        print("  Looking for:")
        for x in missing:
            print(f"    {x}")
        f.Close()
        return 3

    # Style
    h_nom.SetLineColor(ROOT.kBlack)
    h_nom.SetLineWidth(2)
    h_nom.SetStats(0)
    h_nom.SetTitle(f"{process} - {systematic}")

    h_up.SetLineColor(ROOT.kRed + 1)
    h_up.SetLineStyle(2)
    h_up.SetLineWidth(2)
    h_up.SetStats(0)

    h_down.SetLineColor(ROOT.kBlue + 1)
    h_down.SetLineStyle(2)
    h_down.SetLineWidth(2)
    h_down.SetStats(0)

    # Ratios
    h_up_ratio = safe_ratio(h_up, h_nom)
    h_dn_ratio = safe_ratio(h_down, h_nom)

    h_up_ratio.SetLineColor(ROOT.kRed + 1)
    h_up_ratio.SetLineStyle(2)
    h_up_ratio.SetLineWidth(2)
    h_up_ratio.SetStats(0)

    h_dn_ratio.SetLineColor(ROOT.kBlue + 1)
    h_dn_ratio.SetLineStyle(2)
    h_dn_ratio.SetLineWidth(2)
    h_dn_ratio.SetStats(0)

    os.makedirs(outdir, exist_ok=True)

    if combined_mode:
        out_tag = "_".join(combine_regions)
    else:
        out_tag = region

    canvas_name = f"c_{out_tag}_{process}_{systematic}"
    c = ROOT.TCanvas(canvas_name, canvas_name, 800, 800)

    pad1 = ROOT.TPad("pad1", "pad1", 0.0, 0.30, 1.0, 1.0)
    pad2 = ROOT.TPad("pad2", "pad2", 0.0, 0.00, 1.0, 0.30)

    pad1.SetBottomMargin(0.02)
    pad2.SetTopMargin(0.03)
    pad2.SetBottomMargin(0.30)

    if logy:
        pad1.SetLogy()

    pad1.Draw()
    pad2.Draw()

    # Upper pad
    pad1.cd()

    max_val = max(h_nom.GetMaximum(), h_up.GetMaximum(), h_down.GetMaximum())
    positive_mins = []
    for h in [h_nom, h_up, h_down]:
        for ib in range(1, h.GetNbinsX() + 1):
            v = h.GetBinContent(ib)
            if v > 0:
                positive_mins.append(v)
    min_pos = min(positive_mins) if positive_mins else 1e-4

    if logy:
        h_nom.SetMaximum(max_val * 10.0 if max_val > 0 else 1.0)
        h_nom.SetMinimum(min_pos * 0.5)
    else:
        h_nom.SetMaximum(max_val * 1.35 if max_val > 0 else 1.0)
        h_nom.SetMinimum(0.0)

    h_nom.GetYaxis().SetTitle("Events")
    h_nom.GetYaxis().SetTitleSize(0.05)
    h_nom.GetYaxis().SetLabelSize(0.04)
    h_nom.GetXaxis().SetLabelSize(0.0)
    h_nom.Draw("HIST")
    h_up.Draw("HIST SAME")
    h_down.Draw("HIST SAME")

    leg = ROOT.TLegend(0.60, 0.70, 0.88, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(h_nom, "Nominal", "l")
    leg.AddEntry(h_up, f"{systematic} Up", "l")
    leg.AddEntry(h_down, f"{systematic} Down", "l")
    leg.Draw()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.038)
    latex.DrawLatex(0.14, 0.92, f"Mode: {label_mode}")
    latex.DrawLatex(0.14, 0.87, f"Region(s): {region_label}")
    latex.DrawLatex(0.14, 0.82, f"Process: {process}")
    latex.DrawLatex(0.14, 0.77, f"Syst: {systematic}")

    cv_lines = []

    if combined_mode and cv is not None:
        y1 = h_nom.GetMinimum()
        y2 = h_nom.GetMaximum()
        cv_lines.append(draw_vertical_line(cv, y1, y2))

        latex.SetTextSize(0.035)
        latex.DrawLatex(0.18, 0.72, "calib")
        latex.DrawLatex(0.72, 0.72, "sr")

    # Lower pad
    pad2.cd()

    h_up_ratio.GetYaxis().SetTitle("Var/Nom")
    h_up_ratio.GetYaxis().SetNdivisions(505)
    h_up_ratio.GetYaxis().SetTitleSize(0.10)
    h_up_ratio.GetYaxis().SetTitleOffset(0.45)
    h_up_ratio.GetYaxis().SetLabelSize(0.08)

    h_up_ratio.GetXaxis().SetTitle("class_score")
    h_up_ratio.GetXaxis().SetTitleSize(0.12)
    h_up_ratio.GetXaxis().SetTitleOffset(1.0)
    h_up_ratio.GetXaxis().SetLabelSize(0.10)

    h_up_ratio.SetMaximum(1.5)
    h_up_ratio.SetMinimum(0.5)
    h_up_ratio.Draw("HIST")
    h_dn_ratio.Draw("HIST SAME")

    line = ROOT.TLine(
        h_up_ratio.GetXaxis().GetXmin(), 1.0,
        h_up_ratio.GetXaxis().GetXmax(), 1.0
    )
    line.SetLineStyle(2)
    line.Draw()

    if combined_mode and cv is not None:
        cv_lines.append(draw_vertical_line(cv, 0.5, 1.5))

    # Integral check
    int_nom = h_nom.Integral(1, h_nom.GetNbinsX())
    int_up = h_up.Integral(1, h_up.GetNbinsX())
    int_dn = h_down.Integral(1, h_down.GetNbinsX())

    print(f"--- Integral Check ---")
    print(f"File    : {file_name}")
    print(f"Mode    : {label_mode}")
    print(f"Region  : {region_label}")
    print(f"Process : {process}")
    print(f"Syst    : {systematic}")
    print(f"Nominal : {int_nom:.6f}")
    if int_nom != 0:
        print(f"Up      : {int_up:.6f} ({(int_up / int_nom - 1.0) * 100:+.2f}%)")
        print(f"Down    : {int_dn:.6f} ({(int_dn / int_nom - 1.0) * 100:+.2f}%)")
    else:
        print(f"Up      : {int_up:.6f} (nominal is zero)")
        print(f"Down    : {int_dn:.6f} (nominal is zero)")

    if combined_mode:
        out_png = os.path.join(outdir, f"check_combined_{out_tag}_{process}_{systematic}.png")
    else:
        out_png = os.path.join(outdir, f"check_{region}_{process}_{systematic}.png")

    c.SaveAs(out_png)
    print(f"[Done] Plot saved to {out_png}")

    f.Close()
    return 0


def main():
    parser = argparse.ArgumentParser(description="Check nominal/up/down shapes in a ROOT file")

    parser.add_argument("-f", "--file", required=True, help="Input ROOT file")

    # Single-region mode
    parser.add_argument("-r", "--region", default=None,
                        help="Single region directory in ROOT file, e.g. calib or sr")

    # Combined-region mode
    parser.add_argument("--combine-regions", nargs="+", default=None,
                        help="Combine multiple regions into one plot, e.g. --combine-regions calib sr")
    parser.add_argument("--cv", type=float, default=None,
                        help="Draw vertical line at class_score = cv in combined mode")

    parser.add_argument("-p", "--process", required=True, help="Process name, e.g. sig, bkg_wqq")
    parser.add_argument("-s", "--systematic", required=True, help="Systematic name, e.g. PU, FT_JER, Theory_muR")
    parser.add_argument("-o", "--outdir", default="figure", help="Output directory")
    parser.add_argument("--logy", action="store_true", help="Use log scale on upper pad")

    args = parser.parse_args()

    if args.combine_regions is None and args.region is None:
        print("[ERROR] Please provide either --region or --combine-regions")
        sys.exit(1)

    if args.combine_regions is not None and args.region is not None:
        print("[ERROR] Please use either --region or --combine-regions, not both")
        sys.exit(1)

    ret = draw_shape_comparison(
        file_name=args.file,
        region=args.region,
        combine_regions=args.combine_regions,
        cv=args.cv,
        process=args.process,
        systematic=args.systematic,
        outdir=args.outdir,
        logy=args.logy,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()
