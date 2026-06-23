#!/usr/bin/env python3
import os
import subprocess
import logging
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# 默认配置（可被命令行参数覆盖）
DEFAULT_INPUT_DIR = '/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/MC/pieces/'
DEFAULT_OUTPUT_DIR = '/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/MC/rename_sample/'
DEFAULT_RENAME_CONFIG = 'patch/rename_v1.yaml'
DEFAULT_RENAME_SCRIPT = './rename_branch.py'
DEFAULT_LOG_FILE = 'rename_error.log'

isdata=True  #
if isdata==True:
    DEFAULT_INPUT_DIR = '/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/data/pieces/'
    DEFAULT_OUTPUT_DIR = '/afs/cern.ch/user/y/youpeng/eos/RunTTH/_2017_1L/data/rename_data/'

def setup_logger(log_file):
    """配置主进程日志（只在主进程写文件，子进程将 stderr 返还给主进程）"""
    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def process_single_file(args_tuple):
    """
    在子进程中执行重命名脚本，返回结果字典。
    使用 args_tuple 以便在 ProcessPoolExecutor 中安全传参。
    """
    filename, input_dir, output_dir, rename_script, rename_config = args_tuple
    input_path = os.path.join(input_dir, filename)

    # 使用脚本的绝对路径以减少 cwd 相关问题
    script_path = os.path.abspath(rename_script)

    cmd = [script_path, '-i', input_path, '-o', output_dir, '-c', rename_config]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stderr = (proc.stderr or '').strip()
        stdout = (proc.stdout or '').strip()

        if proc.returncode == 0:
            return {'filename': filename, 'ok': True, 'returncode': 0, 'stderr': stderr, 'stdout': stdout}
        else:
            # 如果 stderr 为空，就把 stdout 也记录以便排查
            msg = stderr or stdout or f"process exited with code {proc.returncode}"
            return {'filename': filename, 'ok': False, 'returncode': proc.returncode, 'stderr': msg, 'stdout': stdout}
    except Exception as e:
        return {'filename': filename, 'ok': False, 'returncode': None, 'stderr': str(e), 'stdout': ''}


def parse_args():
    p = argparse.ArgumentParser(description="Parallel rename runner for ROOT files")
    p.add_argument('--input', '-i', default=DEFAULT_INPUT_DIR, help='Input directory containing .root files')
    p.add_argument('--output', '-o', default=DEFAULT_OUTPUT_DIR, help='Output directory')
    p.add_argument('--config', '-c', default=DEFAULT_RENAME_CONFIG, help='Rename config file')
    p.add_argument('--script', '-s', default=DEFAULT_RENAME_SCRIPT, help='Rename script to run')
    p.add_argument('--workers', '-w', type=int, default=None, help='Number of parallel workers (env RENAME_WORKERS overrides)')
    p.add_argument('--log', default=DEFAULT_LOG_FILE, help='Error log file written by main process')
    p.add_argument('--dry-run', action='store_true', help='Show planned commands without executing')
    return p.parse_args()


def main():
    args = parse_args()

    input_dir = args.input
    output_dir = args.output
    rename_config = args.config
    rename_script = args.script
    log_file = args.log

    # 决定 worker 数
    env_workers = os.environ.get('RENAME_WORKERS')
    if env_workers is not None:
        try:
            max_workers = int(env_workers)
        except ValueError:
            max_workers = None
    else:
        max_workers = args.workers

    if max_workers is None:
        max_workers = os.cpu_count() or 4

    setup_logger(log_file)

    # 基本检查
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(rename_script):
        print(f"Error: Rename script {rename_script} does not exist.")
        logging.error(f"Rename script {rename_script} does not exist.")
        return

    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.root')])
    total_files = len(files)
    print(f"Found {total_files} .root files in {input_dir}")
    print(f"Using config: {rename_config}")
    print(f"Running with {max_workers} workers (set RENAME_WORKERS env to change)")

    if total_files == 0:
        return

    # 如果只做 dry-run，打印命令并退出
    if args.dry_run:
        for f in files:
            input_path = os.path.join(input_dir, f)
            cmd = [os.path.abspath(rename_script), '-i', input_path, '-o', output_dir, '-c', rename_config]
            print(' '.join(cmd))
        return

    # 准备子进程参数列表
    task_args = [(f, input_dir, output_dir, rename_script, rename_config) for f in files]

    success_count = 0
    fail_count = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_file, ta): ta[0] for ta in task_args}
        for fut in as_completed(futures):
            completed += 1
            try:
                res = fut.result()
            except Exception as e:
                # 极少发生：子进程提交或 pickling 异常
                filename = futures.get(fut, '<unknown>')
                logging.error(f"Unexpected executor error for {filename}: {str(e)}")
                print(f"[{completed}/{total_files}] ERROR: {filename} (Check log)")
                fail_count += 1
                continue

            filename = res['filename']
            if res['ok']:
                success_count += 1
                print(f"[{completed}/{total_files}] Success: {filename}")
            else:
                fail_count += 1
                err_msg = res.get('stderr') or f"exit code {res.get('returncode')}"
                logging.error(f"Failed to process {filename}. Error: {err_msg}")
                print(f"[{completed}/{total_files}] ERROR: {filename} (Check log)")

    print("-" * 30)
    print("Processing complete.")
    print(f"Total: {total_files}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")

    if fail_count > 0:
        print(f"Errors have been logged to: {os.path.abspath(log_file)}")


if __name__ == "__main__":
    main()
   