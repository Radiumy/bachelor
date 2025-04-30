# import re
# import json
# import os
# from ollama import chat

# config = {
#     "model_name": "deepseek-r1:1.5b",
#     "prompt_path": "../prompt/safety_prompt.txt",
#     "input_path": "../datasets/NotInject_three.json",
#     "output_dir": "../final/ours_processed_data",
# }

# model_short_name = config["model_name"].replace(":", "_")
# input_file_base = os.path.splitext(os.path.basename(config["input_path"]))[0]
# output_file = os.path.join(config["output_dir"], f"{input_file_base}_{model_short_name}_nouni.json")

# with open(config["prompt_path"], "r", encoding="utf-8") as f:
#     safety_prompt = f.read().strip() + "\n"  

# def split_sentences(text):
#     pattern = r'(?<=[，。！？!?])|(?<=[,\.\?!])(?=\s)'
#     sentences = re.split(pattern, text)
#     return [s.strip() for s in sentences if s.strip()]

# def split_json():
#     with open(config["input_path"], 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     prompt_sentences = []
#     for item in data:
#         prompt = item['prompt']
#         prompt_sentence = split_sentences(prompt)
#         prompt_sentences.append({'prompt': prompt, 'sentences': prompt_sentence})
#     return prompt_sentences

# def ask_llm(prompt):
#     full_prompt = safety_prompt + prompt
    
#     response = chat(
#         model=config["model_name"],
#         messages=[
#             {"role": "user", "content": full_prompt}
#         ]
#     )
#     result = response['message']['content']
#     match = re.search(r'<\s*([01])\s*>', result)
#     judgement = match.group(1) if match else "unknown"
#     return result, judgement

# def save_to_json(output_result, output_file):
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(output_result, f, ensure_ascii=False, indent=4)

# split_result = split_json()
# output_result = []

# for item in split_result:
#     llm_raw = []
#     llm_judge = []
#     flag = True
#     for sentence in item['sentences']:
#         raw_response, judgement = ask_llm(sentence)
#         if judgement == '0':
#             flag = False
#         llm_raw.append(raw_response)
#         llm_judge.append(judgement)
#     output_result.append({'prompt': item['prompt'], 'sentences': item['sentences'], 'raw': llm_raw, 'safety': llm_judge, 'totally_safe': flag})

# save_to_json(output_result, output_file)
# print(f"successfully saved:{output_file}")

#--------------------------------------


import re
import json
import os
from ollama import chat

config = {
    "model_name": "deepseek-r1:1.5b",
    "prompt_path": "../prompt/safety_prompt.txt",
    "input_path": "../datasets/NotInject_three.json",
    "output_dir": "../final/other_results/",
}

model_short_name = config["model_name"].replace(":", "_")
input_file_base = os.path.splitext(os.path.basename(config["input_path"]))[0]
output_file = os.path.join(config["output_dir"], f"{input_file_base}_{model_short_name}_nouni.json")

with open(config["prompt_path"], "r", encoding="utf-8") as f:
    safety_prompt = f.read().strip() + "\n"  
with open(config["input_path"], 'r', encoding='utf-8') as f:
    data = json.load(f)
def ask_llm(prompt):
    full_prompt = safety_prompt + prompt
    response = chat(
        model=config["model_name"],
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

output_result = []

for item in data:
    llm_raw = []
    llm_judge = []
    flag = True
    raw_response, judgement = ask_llm(item['prompt'])
    if judgement == '0':
        flag = False
    llm_raw.append(raw_response)
    llm_judge.append(judgement)
    output_result.append({'prompt': item['prompt'], 'raw': llm_raw, 'safety': llm_judge, 'totally_safe': flag})

save_to_json(output_result, output_file)
print(f"successfully saved:{output_file}")