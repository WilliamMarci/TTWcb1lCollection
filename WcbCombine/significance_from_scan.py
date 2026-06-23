#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
原理说明
========

本脚本用于从 Combine 的 MultiDimFit 扫描结果中估计信号显著度。

假设我们扫描的参数是信号强度 mu（在 Combine 中常常对应分支名 r），
Combine 输出的 tree 中通常包含：

    mu_scan_value   -> 例如 r
    deltaNLL        -> 与最优点相比的 deltaNLL

而我们常用的 profile likelihood 曲线纵轴是：

    2ΔNLL = 2 * deltaNLL

对于检验“无信号假设” H0: mu = 0，在渐近近似（Wald approximation）下，

    q0 ≈ 2ΔNLL(mu=0) - 2ΔNLL(mu_hat)

如果 scan 的最小值已经归一化为 0，即：

    2ΔNLL(mu_hat) ≈ 0

那么就有：

    q0 ≈ 2ΔNLL(mu=0)

从而信号显著度近似为：

    Z ≈ sqrt(q0)

也就是：

    Z ≈ sqrt( 2ΔNLL(mu=0) )

注意：
1. 这给出的是渐近近似下的显著度，不是 toy-based 精确 p-value。
2. 如果扫描点中没有恰好 mu=0，本脚本会在最接近 mu=0 的邻域做线性插值。
3. 如果你的 Combine 输出不是标准 MultiDimFit 格式，分支名可能需要调整。
"""

import argparse
import math
import sys

import numpy as np
import uproot
import matplotlib.pyplot as plt


def load_scan(root_file, tree_name="limit", poi_name="r"):
    with uproot.open(root_file) as f:
        tree = f[tree_name]
        arr = tree.arrays([poi_name, "deltaNLL"], library="np")

    x = arr[poi_name].astype(float)
    y = 2.0 * arr["deltaNLL"].astype(float)  # 2ΔNLL

    # 去掉 NaN / inf
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # 按 x 排序
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    return x, y


def remove_duplicate_x_keep_min_y(x, y):
    """
    如果同一个 x 有多个点，保留 y 最小的那个。
    """
    unique = {}
    for xi, yi in zip(x, y):
        if xi not in unique:
            unique[xi] = yi
        else:
            unique[xi] = min(unique[xi], yi)

    xs = np.array(sorted(unique.keys()), dtype=float)
    ys = np.array([unique[k] for k in xs], dtype=float)
    return xs, ys


def interpolate_y_at_x0(x, y, x0=0.0, tol=0.1):
    """
    在线性插值/轻微外推下求 y(x0)。

    规则：
    1. 若 x 中恰有 x0，则直接返回；
    2. 若 x0 在扫描范围内，则线性插值；
    3. 若 x0 刚好在范围外但离边界很近（默认 tol=0.1），则允许线性外推；
    4. 若超出太多，则报错。

    tol 的单位与 x 相同，比如这里 mu 的单位通常就是无量纲 signal strength。
    """
    # exact match
    idx_exact = np.where(np.isclose(x, x0, atol=1e-12))[0]
    if len(idx_exact) > 0:
        return float(np.min(y[idx_exact]))

    xmin, xmax = np.min(x), np.max(x)

    # inside range: interpolation
    if xmin < x0 < xmax:
        idx_right = np.searchsorted(x, x0)
        idx_left = idx_right - 1

        x1, y1 = x[idx_left], y[idx_left]
        x2, y2 = x[idx_right], y[idx_right]

        if np.isclose(x2, x1):
            return float(min(y1, y2))

        return float(y1 + (y2 - y1) * (x0 - x1) / (x2 - x1))

    # slightly below lower edge: extrapolation
    if x0 < xmin and (xmin - x0) <= tol:
        x1, y1 = x[0], y[0]
        x2, y2 = x[1], y[1]

        if np.isclose(x2, x1):
            return float(y1)

        print(f"[WARN] x0={x0} is slightly below scan range; using linear extrapolation from first two points.")
        return float(y1 + (y2 - y1) * (x0 - x1) / (x2 - x1))

    # slightly above upper edge: extrapolation
    if x0 > xmax and (x0 - xmax) <= tol:
        x1, y1 = x[-2], y[-2]
        x2, y2 = x[-1], y[-1]

        if np.isclose(x2, x1):
            return float(y2)

        print(f"[WARN] x0={x0} is slightly above scan range; using linear extrapolation from last two points.")
        return float(y1 + (y2 - y1) * (x0 - x1) / (x2 - x1))

    raise ValueError(
        f"目标点 x0={x0} 不在扫描范围内，当前范围为 [{xmin}, {xmax}]，且超出边界超过 tol={tol}"
    )



def compute_significance(x, y, x_null=0.0):
    """
    从扫描曲线计算显著度：
        q0 = y(x_null) - y_min
        Z  = sqrt(max(q0, 0))
    其中 y = 2ΔNLL
    """
    y_min = float(np.min(y))
    x_best = float(x[np.argmin(y)])

    y_null = interpolate_y_at_x0(x, y, x_null)
    q0 = y_null - y_min
    if q0 < 0:
        q0 = 0.0

    Z = math.sqrt(q0)
    return {
        "x_best": x_best,
        "y_min": y_min,
        "x_null": x_null,
        "y_null": y_null,
        "q0": q0,
        "Z": Z,
    }


def plot_scan(x, y, result, poi_name="r", output="scan_significance.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, "o-", ms=4, lw=1.5, label=r"$2\Delta \mathrm{NLL}$ scan")

    plt.axvline(result["x_best"], color="C2", ls="--", label=f"best fit = {result['x_best']:.4g}")
    plt.axvline(result["x_null"], color="C3", ls="--", label=f"null = {result['x_null']:.4g}")
    plt.axhline(result["y_null"], color="C3", ls=":", label=f"2ΔNLL(null) = {result['y_null']:.4g}")

    plt.xlabel(poi_name)
    plt.ylabel(r"$2\Delta \mathrm{NLL}$")
    plt.title(f"Scan of {poi_name}: Z ≈ {result['Z']:.3f} sigma")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"[INFO] 图已保存到: {output}")


def main():
    parser = argparse.ArgumentParser(
        description="从 Combine 的 MultiDimFit 扫描结果估计信号显著度"
    )
    parser.add_argument("rootfile", help="Combine 的 MultiDimFit 输出 ROOT 文件")
    parser.add_argument("--tree", default="limit", help="TTree 名称，默认: limit")
    parser.add_argument("--poi", default="r", help="扫描的 POI 分支名，默认: r")
    parser.add_argument("--null", type=float, default=0.0, help="无信号假设的 mu 值，默认: 0")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="是否输出扫描图"
    )
    parser.add_argument(
        "--output",
        default="scan_significance.png",
        help="输出图文件名，默认: scan_significance.png"
    )

    args = parser.parse_args()

    try:
        x, y = load_scan(args.rootfile, tree_name=args.tree, poi_name=args.poi)
        x, y = remove_duplicate_x_keep_min_y(x, y)
        result = compute_significance(x, y, x_null=args.null)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print("==== Combine scan -> significance ====")
    print(f"Input file           : {args.rootfile}")
    print(f"POI name             : {args.poi}")
    print(f"Best-fit {args.poi}  : {result['x_best']:.6g}")
    print(f"Minimum 2ΔNLL        : {result['y_min']:.6g}")
    print(f"Null hypothesis mu   : {result['x_null']:.6g}")
    print(f"2ΔNLL at null        : {result['y_null']:.6g}")
    print(f"q0                   : {result['q0']:.6g}")
    print(f"Significance Z       : {result['Z']:.6g} sigma")

    if args.plot:
        plot_scan(x, y, result, poi_name=args.poi, output=args.output)


if __name__ == "__main__":
    main()
