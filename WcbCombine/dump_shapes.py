#!/usr/bin/env python3
import argparse
import ctypes
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(False)
ROOT.gStyle.SetOptStat(0)


def get_hist(f, region, proc):
    path = f"{region}/{proc}"
    h = f.Get(path)
    if not h:
        raise RuntimeError(f"Cannot find histogram: {path}")
    h = h.Clone(f"{region}_{proc}_clone")
    h.SetDirectory(0)
    return h


def integral_and_err(h):
    err = ctypes.c_double(0.0)
    val = h.IntegralAndError(1, h.GetNbinsX(), err)
    return float(val), float(err.value)


def style_hist(h, color, width=3, linestyle=1):
    h.SetLineColor(color)
    h.SetLineWidth(width)
    h.SetLineStyle(linestyle)
    h.SetFillStyle(0)
    h.SetMarkerSize(0)


def style_data(h):
    h.SetMarkerStyle(20)
    h.SetMarkerSize(0.9)
    h.SetLineColor(ROOT.kBlack)
    h.SetLineWidth(1)


def make_total_bkg(hdict, region_name):
    h = hdict["bkg_wqq"].Clone(f"{region_name}_total_bkg")
    h.Add(hdict["bkg_topbc"])
    h.Add(hdict["bkg_other"])
    h.SetDirectory(0)
    return h


def make_total_sb(hdict, region_name):
    h = hdict["sig"].Clone(f"{region_name}_total_sb")
    h.Add(hdict["bkg_wqq"])
    h.Add(hdict["bkg_topbc"])
    h.Add(hdict["bkg_other"])
    h.SetDirectory(0)
    return h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input ROOT file")
    parser.add_argument("-o", "--output", default="overlay_calib_sr.png", help="Output figure")
    parser.add_argument("--calib", default="calib", help="Calibration region directory")
    parser.add_argument("--sr", default="sr", help="Signal region directory")
    parser.add_argument("--xmin", type=float, default=0.0, help="Global x min")
    parser.add_argument("--xmax", type=float, default=1.0, help="Global x max")
    parser.add_argument("--boundary", type=float, default=0.7, help="Boundary between calib and sr")
    parser.add_argument(
        "--processes",
        nargs="+",
        default=["sig", "bkg_wqq", "bkg_topbc", "bkg_other"],
        help="Processes to draw"
    )
    args = parser.parse_args()

    f = ROOT.TFile.Open(args.input)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open file: {args.input}")

    h_calib = {}
    h_sr = {}
    for proc in args.processes + ["data_obs"]:
        h_calib[proc] = get_hist(f, args.calib, proc)
        h_sr[proc] = get_hist(f, args.sr, proc)

    h_calib["total_bkg"] = make_total_bkg(h_calib, "calib")
    h_sr["total_bkg"] = make_total_bkg(h_sr, "sr")
    h_calib["total_sb"] = make_total_sb(h_calib, "calib")
    h_sr["total_sb"] = make_total_sb(h_sr, "sr")

    print("\n=== Yield summary ===")
    for region_name, hdict in [("calib", h_calib), ("sr", h_sr)]:
        print(f"\n[{region_name}]")
        for proc in args.processes + ["total_bkg", "total_sb", "data_obs"]:
            y, ye = integral_and_err(hdict[proc])
            print(f"  {proc:12s} : {y:12.6f} +/- {ye:10.6f}")

    # style
    colors = {
        "sig": ROOT.kRed + 1,
        "bkg_wqq": ROOT.kAzure + 1,
        "bkg_topbc": ROOT.kGreen + 2,
        "bkg_other": ROOT.kMagenta + 1,
        "total_bkg": ROOT.kGray + 2,
        "total_sb": ROOT.kOrange + 7,
    }

    for proc in args.processes:
        style_hist(h_calib[proc], colors[proc], width=3, linestyle=1)
        style_hist(h_sr[proc], colors[proc], width=3, linestyle=2)

    style_hist(h_calib["total_bkg"], colors["total_bkg"], width=3, linestyle=1)
    style_hist(h_sr["total_bkg"], colors["total_bkg"], width=3, linestyle=2)

    style_hist(h_calib["total_sb"], colors["total_sb"], width=3, linestyle=1)
    style_hist(h_sr["total_sb"], colors["total_sb"], width=3, linestyle=2)

    style_data(h_calib["data_obs"])
    style_data(h_sr["data_obs"])
    h_calib["data_obs"].SetMarkerColor(ROOT.kBlack)
    h_sr["data_obs"].SetMarkerColor(ROOT.kBlack)
    h_sr["data_obs"].SetMarkerStyle(24)

    ymax = 0.0
    ymin_pos = 1e30
    for hdict in [h_calib, h_sr]:
        for proc in args.processes + ["total_bkg", "total_sb", "data_obs"]:
            h = hdict[proc]
            for b in range(1, h.GetNbinsX() + 1):
                y = h.GetBinContent(b)
                if y > ymax:
                    ymax = y
                if y > 0 and y < ymin_pos:
                    ymin_pos = y

    ymin = max(1e-3, ymin_pos * 0.5 if ymin_pos < 1e30 else 1e-3)
    ymax = max(10.0, ymax * 30.0)

    c = ROOT.TCanvas("c", "c", 1200, 800)
    c.SetLogy()
    c.SetMargin(0.10, 0.04, 0.12, 0.08)

    frame = ROOT.TH1D("frame", "", 50, args.xmin, args.xmax)
    frame.SetMinimum(ymin)
    frame.SetMaximum(ymax)
    frame.GetXaxis().SetTitle("class_score")
    frame.GetYaxis().SetTitle("Events / bin")
    frame.GetXaxis().SetTitleSize(0.045)
    frame.GetYaxis().SetTitleSize(0.045)
    frame.GetXaxis().SetLabelSize(0.035)
    frame.GetYaxis().SetLabelSize(0.035)
    frame.Draw()

    draw_order = ["total_sb", "total_bkg", "sig", "bkg_wqq", "bkg_topbc", "bkg_other"]

    # draw calib first
    for proc in draw_order:
        h_calib[proc].Draw("HIST SAME")

    # draw sr on same axis, dashed
    for proc in draw_order:
        h_sr[proc].Draw("HIST SAME")

    h_calib["data_obs"].Draw("E1 SAME")
    h_sr["data_obs"].Draw("E1 SAME")

    line = ROOT.TLine(args.boundary, ymin, args.boundary, ymax)
    line.SetLineStyle(2)
    line.SetLineWidth(3)
    line.SetLineColor(ROOT.kBlack)
    line.Draw()

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.040)
    latex.DrawLatex(0.22, 0.92, f"{args.calib} region")
    latex.DrawLatex(0.68, 0.92, f"{args.sr} region")

    y_calib_data, _ = integral_and_err(h_calib["data_obs"])
    y_sr_data, _ = integral_and_err(h_sr["data_obs"])
    latex.SetTextSize(0.032)
    latex.DrawLatex(0.12, 0.84, f"{args.calib} data = {y_calib_data:.2f}")
    latex.DrawLatex(0.58, 0.84, f"{args.sr} data = {y_sr_data:.2f}")

    leg = ROOT.TLegend(0.62, 0.50, 0.93, 0.88)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)

    for proc in args.processes:
        leg.AddEntry(h_calib[proc], f"{proc} (calib)", "l")
    leg.AddEntry(h_calib["total_bkg"], "total_bkg (calib)", "l")
    leg.AddEntry(h_calib["total_sb"], "total_s+b (calib)", "l")
    leg.AddEntry(h_calib["data_obs"], "data_obs (calib)", "lep")

    for proc in args.processes:
        leg.AddEntry(h_sr[proc], f"{proc} (sr)", "l")
    leg.AddEntry(h_sr["total_bkg"], "total_bkg (sr)", "l")
    leg.AddEntry(h_sr["total_sb"], "total_s+b (sr)", "l")
    leg.AddEntry(h_sr["data_obs"], "data_obs (sr)", "lep")

    leg.AddEntry(line, "region boundary", "l")
    leg.Draw()

    c.SaveAs(args.output)
    if args.output.endswith(".png"):
        c.SaveAs(args.output.replace(".png", ".pdf"))

    print(f"\nSaved plot to: {args.output}")
    if args.output.endswith(".png"):
        print(f"Saved plot to: {args.output.replace('.png', '.pdf')}")

    f.Close()


if __name__ == "__main__":
    main()
