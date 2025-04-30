import re
import json
import os
import csv
from ollama import chat
from tqdm import tqdm

config = {
    "model_names": ["deepseek-r1:1.5b"],
    "prompt_path": "../prompt/safety_prompt.txt",
    "input_paths": ["../datasets/jailbreak_dataset_full_balanced.csv"],
    "output_dir": "../final/other_results/",
}

def read_csv(input_path):
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'prompt': row['prompt'],
                'type': row.get('type', '') 
            })
    return data

def ask_llm(model_name, prompt, safety_prompt):
    full_prompt = safety_prompt + prompt
    response = chat(
        model=model_name,
        messages=[
            {"role": "user", "content": full_prompt}
        ]
    )
    result = response['message']['content']
    match = re.search(r'<\s*([01])\s*>', result)
    judgement = match.group(1) if match else "unknown"
    return result, judgement

def save_to_json(output_result, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_result, f, ensure_ascii=True, indent=4)

def process_dataset(model_name, input_path, safety_prompt):
    model_short_name = model_name.replace(":", "_")
    input_file_base = os.path.splitext(os.path.basename(input_path))[0]
    output_file = os.path.join(config["output_dir"], f"{input_file_base}_{model_short_name}_nouni.json")

    data = read_csv(input_path)
    
    output_result = []
    
    for item in tqdm(data, desc=f"Processing {model_name}", unit="prompt"):
        raw_response, judgement = ask_llm(model_name, item['prompt'], safety_prompt)
        output_result.append({
            'prompt': item['prompt'],
            'type': item['type'],
            'raw': raw_response,
            'safety': judgement,
            'totally_safe': (judgement == '1') 
        })
    
    save_to_json(output_result, output_file)
    print(f"successfully saved: {output_file}")

def main():
    with open(config["prompt_path"], "r", encoding="utf-8") as f:
        safety_prompt = f.read().strip() + "\n"
    
    for model_name in tqdm(config["model_names"], desc="Models"):
        for input_path in tqdm(config["input_paths"], desc="Datasets"):
            process_dataset(model_name, input_path, safety_prompt)

if __name__ == "__main__":
    main()