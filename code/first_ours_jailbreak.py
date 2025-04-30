import re
import json
import csv
import os
from ollama import chat
from tqdm import tqdm

config = {
    "model_names": ["llama3.2:1b", "qwen2.5:1.5b", "deepseek-r1:1.5b"],
    "prompt_path": "../prompt/safety_prompt.txt",
    "input_paths": ["../datasets/jailbreak_dataset_full_balanced.csv"], 
    "output_dir": "../final/ours_processed_data",
}

def split_sentences(text):
    pattern = r'(?<=[。！？!?])|(?<=[\.\?!])(?=\s)'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]

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

def split_csv(input_path):
    data = read_csv(input_path)
    prompt_sentences = []
    for item in data:
        prompt = item['prompt']
        prompt_sentence = split_sentences(prompt)
        prompt_sentences.append({
            'prompt': prompt,
            'type': item['type'],
            'sentences': prompt_sentence
        })
    return prompt_sentences

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
        json.dump(output_result, f, ensure_ascii=False, indent=4)

def process_dataset(model_name, input_path, safety_prompt):
    model_short_name = model_name.replace(":", "_")
    input_file_base = os.path.splitext(os.path.basename(input_path))[0]
    output_file = os.path.join(config["output_dir"], f"{input_file_base}_{model_short_name}_nouni.json")
    
    split_result = split_csv(input_path)
    output_result = []
    
    for item in tqdm(split_result, desc=f"Processing {model_name}", unit="prompt"):
        llm_raw = []
        llm_judge = []
        flag = True
        for sentence in item['sentences']:
            raw_response, judgement = ask_llm(model_name, sentence, safety_prompt)
            if judgement != '1':
                flag = False
            llm_raw.append(raw_response)
            llm_judge.append(judgement)
        output_result.append({
            'prompt': item['prompt'],
            'type': item['type'],
            'sentences': item['sentences'],
            'raw': llm_raw,
            'safety': llm_judge,
            'totally_safe': flag
        })
    save_to_json(output_result, output_file)
    print(f"Successfully saved: {output_file}")

def main():
    os.makedirs(config["output_dir"], exist_ok=True)

    with open(config["prompt_path"], "r", encoding="utf-8") as f:
        safety_prompt = f.read().strip() + "\n"

    for model_name in tqdm(config["model_names"], desc="Models"):
        for input_path in config["input_paths"]:
            process_dataset(model_name, input_path, safety_prompt)

if __name__ == "__main__":
    main()