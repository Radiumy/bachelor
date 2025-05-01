import re
import json
import os
import argparse
from tqdm import tqdm
from utils import read_json, create_json, ask_llm, split_data

CONFIG_PATH = "../config/config.json"

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

def print_config(dataset, model, input, output):
    print("\n当前配置:")
    print(f"数据集: {dataset}")
    print(f"模型: {model}")
    print(f"输入路径: {input}")
    print(f"输出路径: {output}\n")

def parse_args():
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_names', type=str, nargs='+',
                       default=config.get("default_datasets", []),
                       help="数据集名称列表，默认为config.json中的default_datasets")
    parser.add_argument('--model_names', type=str, nargs='+',
                       default=config.get("default_models", []),
                       help="模型名称列表，默认为config.json中的default_models")
    parser.add_argument('--input_path', type=str, default="../datasets/")
    parser.add_argument('--output_other_dir', type=str, default="../eval/others_results/")
    parser.add_argument('--output_our_dir', type=str, default="../eval/ours_processed_data")
    parser.add_argument('--prompt_path', type=str, default="../prompt/safety_prompt.txt")
    
    return parser.parse_args()

def process_prompt(model_name, item, safety_prompt):
    raw_response, judgement = ask_llm(model_name, item['prompt'], safety_prompt)
    
    result = {
        'prompt': item['prompt'],
        'raw': raw_response,
        'safety': judgement,
        'if_judge': 0 if judgement == "unknown" else 1,
        'totally_safe': None if judgement == "unknown" else (judgement == "1")
    }
    for key, value in item.items():
        if key not in result:
            result[key] = value
            
    return result

def process_sentences(model_name, item, safety_prompt):
    llm_raw = []
    llm_judge = []
    has_unknown = False
    has_unsafe = False
    
    for sentence in item['sentences']:
        raw_response, judgement = ask_llm(model_name, sentence, safety_prompt)
        llm_raw.append(raw_response)
        llm_judge.append(judgement)
        
        if judgement == "unknown":
            has_unknown = True
        elif judgement == "0":
            has_unsafe = True
    
    result = {
        'prompt': item['prompt'],
        'sentences': item['sentences'],
        'raw': llm_raw,
        'safety': llm_judge,
        'if_judge': 0 if has_unknown else 1,
        'totally_safe': None if has_unknown else (False if has_unsafe else True)
    }
    for key, value in item.items():
        if key not in result:
            result[key] = value
            
    return result

def other_process_data(args, safety_prompt):
    for model_name in tqdm(args.model_names, desc="Models (Whole Prompt)"):
        for dataset_name in tqdm(args.dataset_names, desc="Datasets"):
            model_short_name = model_name.replace(":", "_")
            input_file = f"{dataset_name}.json"
            output_file = os.path.join(args.output_other_dir, f"{dataset_name}_{model_short_name}_ollama_wholeprompt.json")
            
            data = read_json(os.path.join(args.input_path, input_file))
            output_result = []
            
            for item in tqdm(data, desc=f"Processing whole prompts {model_name} on {dataset_name}"):
                result = process_prompt(model_name, item, safety_prompt)
                output_result.append(result)
            
            create_json(output_result, output_file)
            print(f"\nOthers: Successfully saved whole prompt results: {output_file}\n")

def our_process_data(args, safety_prompt):
    for model_name in tqdm(args.model_names, desc="Models (Split Sentences)"):
        for dataset_name in tqdm(args.dataset_names, desc="Datasets"):
            model_short_name = model_name.replace(":", "_")
            input_file = f"{dataset_name}.json"
            output_file = os.path.join(args.output_our_dir, f"{dataset_name}_{model_short_name}_ollama_splitsentence.json")
            
            data = read_json(os.path.join(args.input_path, input_file))
            split_result = split_data(data)
            output_result = []
            
            for item in tqdm(split_result, desc=f"Processing split sentences {model_name} on {dataset_name}"):
                result = process_sentences(model_name, item, safety_prompt)
                output_result.append(result)
            
            create_json(output_result, output_file)
            print(f"\nOurs: Successfully saved split sentence results: {output_file}\n")


def main():
    args = parse_args()
    os.makedirs(args.output_our_dir, exist_ok=True)
    os.makedirs(args.output_other_dir, exist_ok=True)

    with open(args.prompt_path, "r", encoding="utf-8") as f:
        safety_prompt = f.read().strip() + "\n"

    #other_process_data(args, safety_prompt)
    our_process_data(args, safety_prompt)

if __name__ == "__main__":
    main()