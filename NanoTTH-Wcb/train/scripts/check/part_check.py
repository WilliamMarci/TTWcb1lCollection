import ROOT
import sys

# 换成你觉得有问题的那个 part 文件路径
f_path ='/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/MC/filetagged_samples_1merged_/part/ST_s-channel_4f_hadronicDecays_TuneCP5_13TeV-amcatnlo-pythia8_part.root'

f = ROOT.TFile.Open(f_path)
t = f.Get("Events")
r = f.Get("Runs")

print(f"File: {f_path}")

# 1. 检查 xsecWeight 分支情况
l = t.GetListOfBranches()
count = 0
for b in l:
    if b.GetName() == "xsecWeight":
        count += 1
print(f"Branch 'xsecWeight' count: {count} (Should be 1)")

# 2. 检查 xsecWeight 的值的分布
print("Scanning xsecWeight values (first 20 events):")
t.Scan("xsecWeight", "", "", 20)

# 3. 检查 Runs tree 和 genEventSumw
print("Checking Runs tree:")
r.Scan("genEventSumw")

# 4. 检查是否为数组
r.GetEntry(0)
val = r.genEventSumw
print(f"genEventSumw type: {type(val)}")
if hasattr(val, '__len__'):
    print(f"genEventSumw length: {len(val)}")
    print(f"genEventSumw values: {list(val)}")
f.Close()