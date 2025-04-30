import csv
import json
import re
import time
from ollama import chat
from typing import Any, Dict, List, Union

def read_json(file_path: str) -> Union[Dict[str, Any], List[Any]]:
    with open(file_path, mode='r', encoding='utf-8') as file:
        return json.load(file)
    
def create_json(data: Union[Dict[str, Any], List[Any]], file_path: str, indent: int = 4) -> None:
    with open(file_path, mode='w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)

def split_sentences(text):
    pattern = r'(?<=[，。！？!?])|(?<=[,\.\?!])(?=\s)'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]

def ask_llm(model_name, prompt, pre_prompt, max_retries=3, retry_delay=1):
    full_prompt = pre_prompt + prompt
    for _ in range(max_retries):
        try:
            response = chat(
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}]
            )
            result = response['message']['content']
            match = re.search(r'<\s*([01])\s*>', result)
            judgement = match.group(1) if match else "unknown"
            return result, judgement
        except Exception as e:
            print(f"Retry due to error: {e}")
            time.sleep(retry_delay)
    return "No response", "unknown"

def split_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompt_sentences = []
    for item in data:
        prompt = item['prompt']
        sentences = split_sentences(prompt)
        new_item = {
            'prompt': prompt,
            'sentences': sentences
        }
        for key, value in item.items():
            if key not in new_item:
                new_item[key] = value
        prompt_sentences.append(new_item)
    return prompt_sentences
