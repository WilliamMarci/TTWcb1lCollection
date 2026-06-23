#!/usr/bin/env python3
import os
import glob
import fnmatch
import argparse
import numpy as np
import awkward as ak
import uproot
from tqdm import tqdm

from dbc_tools import DbcEvaluator


def get_leading(array, default=0):
    return ak.fill_none(ak.firsts(array), default)


def make_cache_path(root_path, cache_dir):
    base = os.path.basename(root_path)
    return os.path.join(cache_dir, base.replace(".root", ".npz"))


def compute_dbc_for_file(filepath, dbc_eval, chunksize=100000):
    with uproot.open(filepath) as f:
        tree = f["Events"]
        num_entries = tree.num_entries

        if num_entries == 0:
            return np.empty(0, dtype=np.float32), 0

        branches = [
            "ak8_gpt_bc",
            "ak8_gpt_bb",
            "ak8_gpt_cc",
            "ak8_gpt_qcd",
            "ak8_gpt_bs",
            "ak8_gpt_qq",
            "ak8_gpt_cs",
            "ak8_gpt_topbw",
        ]

        dbc_chunks = []

        for events in tree.iterate(branches, step_size=chunksize, library="ak"):
            g_bc    = ak.to_numpy(get_leading(events.ak8_gpt_bc)).astype(np.float32)
            g_bb    = ak.to_numpy(get_leading(events.ak8_gpt_bb)).astype(np.float32)
            g_cc    = ak.to_numpy(get_leading(events.ak8_gpt_cc)).astype(np.float32)
            g_qcd   = ak.to_numpy(get_leading(events.ak8_gpt_qcd)).astype(np.float32)
            g_bs    = ak.to_numpy(get_leading(events.ak8_gpt_bs)).astype(np.float32)
            g_qq    = ak.to_numpy(get_leading(events.ak8_gpt_qq)).astype(np.float32)
            g_cs    = ak.to_numpy(get_leading(events.ak8_gpt_cs)).astype(np.float32)
            g_topbw = ak.to_numpy(get_leading(events.ak8_gpt_topbw)).astype(np.float32)

            dbc = dbc_eval.get_Dbc(g_bc, g_bb, g_cc, g_qcd, g_bs, g_qq, g_cs, g_topbw)
            dbc_chunks.append(np.asarray(dbc, dtype=np.float32))

        dbc_score = np.concatenate(dbc_chunks) if dbc_chunks else np.empty(0, dtype=np.float32)

        if len(dbc_score) != num_entries:
            raise RuntimeError(
                f"Length mismatch for {filepath}: len(dbc_score)={len(dbc_score)} "
                f"but tree.num_entries={num_entries}"
            )

        return dbc_score, num_entries


def collect_files(pattern, is_mc):
    files = glob.glob(pattern)
    if is_mc:
        files = [f for f in files if not fnmatch.fnmatch(f, "*QCD_*.root")]
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Build per-file BDT Dbc cache")
    parser.add_argument("--mc_path", type=str, required=True,
                        help="Glob for MC ROOT files")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Glob for data ROOT files")
    parser.add_argument("--cache_dir", type=str, default="./bdt_cache",
                        help="Output cache directory")
    parser.add_argument("--model_path", type=str, default="./bdt_dbc_model.pkl",
                        help="Path to trained BDT model")
    parser.add_argument("--chunksize", type=int, default=100000)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing cache files")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    dbc_eval = DbcEvaluator(
        mode="bdt",
        model_path=args.model_path
    )

    mc_files = collect_files(args.mc_path, is_mc=True)
    data_files = collect_files(args.data_path, is_mc=False)
    all_files = mc_files + data_files

    print(f"Found {len(mc_files)} MC files")
    print(f"Found {len(data_files)} data files")
    print(f"Total files to cache: {len(all_files)}")
    print(f"Cache dir: {args.cache_dir}")
    print("")

    for filepath in tqdm(all_files):
        outpath = make_cache_path(filepath, args.cache_dir)

        if os.path.exists(outpath) and not args.overwrite:
            continue

        try:
            dbc_score, num_entries = compute_dbc_for_file(
                filepath=filepath,
                dbc_eval=dbc_eval,
                chunksize=args.chunksize
            )
            np.savez_compressed(
                outpath,
                dbc_score=dbc_score,
                num_entries=np.int64(num_entries),
            )
        except Exception as e:
            print(f"[ERROR] Failed on {filepath}: {e}")

    print("Done building BDT cache.")


if __name__ == "__main__":
    main()
