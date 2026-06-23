import argparse
import os
import ROOT
from ROOT import RDataFrame, TFile

# cannot set this as it will shuffle the entries
# ROOT.EnableImplicitMT()


def join(infile, friend_file, outfile, cut, treename='Events'):
    # apply selection to the original tree
    print('Reading input file %s' % infile)
    df = RDataFrame(treename, infile)
    df = df.Filter(cut)
    tmpfn = os.path.join(os.path.dirname(outfile), os.path.basename(infile.replace('.root', '_tmp.root')))
    df.Snapshot(treename, tmpfn)

    print('Reading friend tree file %s' % friend_file)
    f = TFile(tmpfn)
    tree = f.Get(treename)
    ff = TFile(friend_file)
    friend_tree = ff.Get(treename)
    tree.AddFriend(friend_tree, "Friend")
    assert(tree.GetEntries('event!=Friend.event') == 0)
    df = RDataFrame(tree)
    df.Snapshot(treename, outfile)
    print('==> Written output to %s' % outfile)

    os.remove(tmpfn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', help='input file directory')
    parser.add_argument('-f', '--friend-dir', help='friend tree file directory')
    parser.add_argument('-o', '--outdir', help='output directory')
    parser.add_argument('-c', '--cut', default='(event%2==1)', help='cut to apply on the input tree')

    args = parser.parse_args()

    infiles = [f for f in os.listdir(args.indir) if f.endswith('.root')]
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    for f in infiles:
        friend_file = os.path.join(args.friend_dir, 'output_%s.root' % f.replace('_tree.root', ''))
        if not os.path.exists(friend_file):
            print('!!! Friend tree %s does not exist !!!' % friend_file)
            continue
        outfile = os.path.join(args.outdir, f)
        join(infile=os.path.join(args.indir, f), friend_file=friend_file, outfile=outfile, cut=args.cut)
