import re
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# 每个句子后面必须要有一个空白符
print("start")
tokenizer = AutoTokenizer.from_pretrained("deepset/deberta-v3-base-injection")
model = AutoModelForSequenceClassification.from_pretrained("deepset/deberta-v3-base-injection")
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device_map="auto")
print("test")
with open("../prompt/safety_prompt.txt", "r", encoding="utf-8") as f:
    safety_prompt = f.read().strip() + "\n"  
def split_sentences(text):
    pattern = r'(?<=[，。！？!?])|(?<=[,\.\?!])(?=\s)'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]

def ask_llm(prompt):
    full_prompt = safety_prompt + prompt
    response = pipe(full_prompt)
    print(response)

print("Input: ")
user_input = input()
split_result = split_sentences(user_input)

prompt_result = []
flag = True
for sentence in split_result:
    print("-------------")
    response = ask_llm(sentence)
    

