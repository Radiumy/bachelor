import re
import json
import os
from ollama import chat
from tqdm import tqdm

config = {
    "model_names": ["llama3.2:1b", "qwen2.5:1.5b", "deepseek-r1:1.5b"],
    "prompt_path": "../prompt/safety_prompt.txt",
    "input_paths": ["../datasets/NotInject_three.json", "../datasets/NotInject_two.json", "../datasets/NotInject_one.json"], 
    "output_dir": "../final/other_results/",
}

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

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_result = []
    
    for item in tqdm(data, desc=f"Processing {model_name} on {input_file_base}"):
        llm_raw = []
        llm_judge = []
        flag = True
        raw_response, judgement = ask_llm(model_name, item['prompt'], safety_prompt)
        if judgement == '0':
            flag = False
        llm_raw.append(raw_response)
        llm_judge.append(judgement)
        output_result.append({'prompt': item['prompt'], 'raw': llm_raw, 'safety': llm_judge, 'totally_safe': flag})
    
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