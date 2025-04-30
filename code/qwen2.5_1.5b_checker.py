import re
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

config = {
    "model_name": "Qwen/Qwen2.5-3B-Instruct",
    "prompt_path": "../prompt/safety_prompt.txt",
    "input_path": "../datasets/NotInject_three.json",
    "output_dir": "../final/ours_processed_data/",
    "max_new_tokens": 512,
    "batch_size": 4
}
model_short_name = config["model_name"].split("/")[-1].replace(".", "_")
input_file_base = os.path.splitext(os.path.basename(config["input_path"]))[0]
output_file = os.path.join(config["output_dir"], f"trans_{input_file_base}_{model_short_name}.json")

tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
model = AutoModelForCausalLM.from_pretrained(config["model_name"])
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", max_new_tokens=config["max_new_tokens"])

with open(config["prompt_path"], "r", encoding="utf-8") as f:
    safety_prompt = f.read().strip() + "\n"  

def split_sentences(text):
    pattern = r'(?<=[，。！？!?])|(?<=[\.\?!])(?=\s)'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]

def split_json():
    with open(config["input_path"], 'r', encoding='utf-8') as f:
        data = json.load(f)
    prompt_sentences = []
    for item in data:
        prompt = item['prompt']
        prompt_sentence = split_sentences(prompt)
        prompt_sentences.append({'prompt': prompt, 'sentences': prompt_sentence})
    return prompt_sentences

def ask_llm_batch(prompts):
    full_prompts = [safety_prompt + prompt for prompt in prompts]
    responses = pipe(full_prompts) 
    
    results = []
    judgements = []
    for response in responses:
        output = response[0]['generated_text']
        result = output[len(full_prompts[responses.index(response)]):].strip()
        match = re.search(r'<\s*([01])\s*>', result)
        judgement = match.group(1) if match else "unknown"
        
        results.append(result)
        judgements.append(judgement)
    
    return results, judgements

def save_to_json(output_result, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_result, f, ensure_ascii=True, indent=4)

split_result = split_json()
output_result = []

for item in split_result:
    llm_raw = []
    llm_judge = []
    flag = True
    sentences_batch = []

    for sentence in item['sentences']:
        sentences_batch.append(sentence)
        if len(sentences_batch) == config["batch_size"] or sentence == item['sentences'][-1]:
            raw_responses, judgements = ask_llm_batch(sentences_batch)
            for raw_response, judgement in zip(raw_responses, judgements):
                llm_raw.append(raw_response)
                llm_judge.append(judgement)
                if judgement == '0':
                    flag = False
            sentences_batch = []
    output_result.append({'prompt': item['prompt'], 'sentences': item['sentences'], 'raw': llm_raw, 'safety': llm_judge, 'totally_safe': flag})

save_to_json(output_result, output_file)
print(f"successfully saved:{output_file}")


# #---------------------------------------


# import re
# import json
# import os
# from ollama import chat

# config = {
#     "model_name": "qwen2.5:1.5b",
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


#---------------------------------------

# import re
# import json
# import os
# from ollama import chat

# config = {
#     "model_name": "qwen2.5:1.5b",
#     "prompt_path": "../prompt/safety_prompt.txt",
#     "input_path": "../datasets/NotInject_three.json",
#     "output_dir": "../final/other_results/",
# }

# model_short_name = config["model_name"].replace(":", "_")
# input_file_base = os.path.splitext(os.path.basename(config["input_path"]))[0]
# output_file = os.path.join(config["output_dir"], f"{input_file_base}_{model_short_name}_nouni.json")

# with open(config["prompt_path"], "r", encoding="utf-8") as f:
#     safety_prompt = f.read().strip() + "\n"  
# with open(config["input_path"], 'r', encoding='utf-8') as f:
#     data = json.load(f)
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
#         json.dump(output_result, f, ensure_ascii=True, indent=4)

# output_result = []

# for item in data:
#     llm_raw = []
#     llm_judge = []
#     flag = True
#     raw_response, judgement = ask_llm(item['prompt'])
#     if judgement == '0':
#         flag = False
#     llm_raw.append(raw_response)
#     llm_judge.append(judgement)
#     output_result.append({'prompt': item['prompt'], 'raw': llm_raw, 'safety': llm_judge, 'totally_safe': flag})

# save_to_json(output_result, output_file)
# print(f"successfully saved:{output_file}")