#! /usr/bin/env python3
import uproot
import sys

def inspect_root_file(file_path, tree_name="Events"):
    try:
        # 以只读模式打开，不读取具体数据，只读 Header，速度极快
        with uproot.open(file_path) as file:
            if tree_name not in file:
                print(f"[Error] Tree '{tree_name}' not found in file.")
                print(f"Available keys: {file.keys()}")
                return

            tree = file[tree_name]
            
            # 获取所有分支名称和类型 (C++ type)
            # typenames() 比 keys() 包含更多信息且开销很小
            branches = tree.typenames()

            print(f"{'Branch Name':<50} | {'Type':<20}")
            print("-" * 75)
            
            for name, type_name in branches.items():
                print(f"{name:<50} | {type_name:<20}")

            print("-" * 75)
            print(f"Total Branches: {len(branches)}")
            print(f"Total Events:   {tree.num_entries}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python printbranches.py <path_to_root_file>")
    else:
        inspect_root_file(sys.argv[1])
