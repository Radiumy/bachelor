import re
import json
import spacy
import os
import argparse
import time
from tqdm import tqdm
from ollama import chat
from utils import read_json, ask_llm, create_json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_names', type=str, nargs='+', required=True)
    parser.add_argument('--model_names', type=str, nargs='+', required=True)
    parser.add_argument('--base_input_path', type=str, default="../eval/ours_processed_data/")
    parser.add_argument('--base_other_path', type=str, default="../eval/others_results/")
    parser.add_argument('--output_dir', type=str, default="../eval/mid")
    parser.add_argument('--prompt_path', type=str, default="../prompt/cot.txt")
    return parser.parse_args()

def pos_tag_and_check_verbs(sentence, nlp):
    doc = nlp(sentence)
    return any(token.pos_ == "VERB" for token in doc)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    nlp = spacy.load("en_core_web_sm")
    with open(args.prompt_path, "r", encoding="utf-8") as f:
        pre_prompt = f.read().strip() + "\n"

    for dataset_name in args.dataset_names:
        for model_name in args.model_names:
            all_data = []
            print(f"Processing {dataset_name} with model {model_name}")
            model_tag = model_name.replace(":", "_").replace("/", "_")

            input_file = os.path.join(args.base_input_path, f"{dataset_name}_{model_tag}_ollama_splitsentence.json")
            other_file = os.path.join(args.base_other_path, f"{dataset_name}_{model_tag}_ollama_wholeprompt.json")
            output_file = os.path.join(args.output_dir, f"mid_{dataset_name}_{model_tag}_ollama_splitsentence.json")
            data = read_json(input_file)
            other_data = read_json(other_file)

            if data is None or other_data is None:
                continue

            for idx, item in enumerate(tqdm(data, desc=f"Processing {dataset_name}-{model_tag}", ncols=100)):
                if item['if_judge'] != 1 or other_data[idx]['if_judge'] != 1:
                    continue

                item_result = {"prompt": item['prompt'], "sentences": item['sentences'], "safety": item['safety'], "totally_safe": item['totally_safe'], "result": None}

                if item['totally_safe']:
                    if other_data[idx]['totally_safe']:
                        item_result['result'] = True
                    else:
                        raw_response, judgement = ask_llm(model_name, item['prompt'], pre_prompt)
                        item_result['result'] = (judgement == '1')
                        item_result['raw'] = raw_response
                        ##TODO: 因为进入cot自我审查已经是不安全的因素了，所以只有安全的时候才能说是安全的

                else:
                    sentence_verb_flags = []
                    sentence_results = []

                    for sentence, safety in zip(item['sentences'], item['safety']):
                        if safety == '0':
                            has_verb = pos_tag_and_check_verbs(sentence, nlp)
                            if not has_verb:
                                sentence_results.append(True)
                                sentence_verb_flags.append(False)
                            else:
                                sentence_results.append(None)
                                sentence_verb_flags.append(True)
                        else:
                            sentence_results.append(True)
                            sentence_verb_flags.append(False)

                    item_result['sentence_results'] = sentence_results
                    item_result['sentence_verbs'] = sentence_verb_flags
                for key, value in item.items():
                    if key not in item_result:
                        item_result[key] = value
                all_data.append(item_result)
    
            create_json(all_data, output_file)
            print(f"Saved middle results to {output_file}")

if __name__ == "__main__":
    main()
