import subprocess
import argparse
import sys
from utils import read_json

def parse_args():
    config = read_json("../config/config.json")
    parser = argparse.ArgumentParser(description="Run both Stage 1 and Stage 2 scripts sequentially.")
    parser.add_argument('--dataset_names', type=str, nargs='+',
                       default=config.get("default_datasets", []),
                       help="数据集名称列表，默认为config.json中的default_datasets")
    parser.add_argument('--model_names', type=str, nargs='+',
                       default=config.get("default_models", []),
                       help="模型名称列表，默认为config.json中的default_models")
    parser.add_argument('--prompt_path', type=str, default="../prompt/cot.txt", help="Prompt file for Stage 1")
    parser.add_argument('--base_input_path', type=str, default="../eval/ours_processed_data/", help="Base input path for Stage 1")
    parser.add_argument('--base_other_path', type=str, default="../eval/others_results/", help="Base other path for Stage 1")
    parser.add_argument('--mid_output_dir', type=str, default="../eval/mid", help="Intermediate output dir from Stage 1")
    parser.add_argument('--final_output_dir', type=str, default="../eval/ours_results", help="Final output dir from Stage 2")
    parser.add_argument('--sim_threshold', type=float, default=0.85, help="Similarity threshold for Stage 2")
    return parser.parse_args()

def run_stage_1(args):
    print("\n=========================")
    print("Running Stage 1: Initial Judgement and Verb Checking...")
    print("=========================\n")

    stage1_command = [
        sys.executable, "split_second_1.py",  # 替换成你的第一阶段脚本名
        "--dataset_names", *args.dataset_names,
        "--model_names", *args.model_names,
        "--prompt_path", args.prompt_path,
        "--base_input_path", args.base_input_path,
        "--base_other_path", args.base_other_path,
        "--output_dir", args.mid_output_dir
    ]

    result = subprocess.run(stage1_command)
    if result.returncode != 0:
        print("\n[Error] Stage 1 failed. Exiting.")
        sys.exit(1)

def run_stage_2(args):
    print("\n=========================")
    print("Running Stage 2: Embedding Similarity Analysis...")
    print("=========================\n")

    stage2_command = [
        sys.executable, "split_second_2.py",  # 替换成你的第二阶段脚本名
        "--dataset_names", *args.dataset_names,
        "--model_names", *args.model_names,
        "--base_input_path", args.mid_output_dir,
        "--output_dir", args.final_output_dir,
        "--sim_threshold", str(args.sim_threshold)
    ]

    result = subprocess.run(stage2_command)
    if result.returncode != 0:
        print("\n[Error] Stage 2 failed.")
        sys.exit(1)

def main():
    args = parse_args()
    run_stage_1(args)
    run_stage_2(args)
    print("\n All stages completed successfully!")

if __name__ == "__main__":
    main()
