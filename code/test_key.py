import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# 每个句子后面必须要有一个空白符
print("start")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto", max_new_tokens=512)
print("test")
with open("../prompt/cot.txt", "r", encoding="utf-8") as f:
    safety_prompt = f.read().strip() + "\n"  

def ask_llm(prompt):
    full_prompt = safety_prompt + prompt
    response = pipe(full_prompt)
    output = response[0]['generated_text']
    print(output[len(full_prompt):].strip())

print("Input: ")
user_input = input()
print("-------------")
response = ask_llm(user_input)

# import re
# import json
# from ollama import chat

# config = {
#     "prompt_path": "../prompt/key_prompt.txt",
#     "input_path": "../datasets/NotInject_three.json",
#     "output_dir": "../processed_data/",
# }
# with open(config["prompt_path"], "r", encoding="utf-8") as f:
#     safety_prompt = f.read().strip() + "\n"  

# def ask_llm(prompt):
#     full_prompt = safety_prompt + prompt
#     response = chat(
#         model="qwen2.5:0.5b",
#         messages=[
#             {"role": "user", "content": full_prompt}
#         ]
#     )
#     print("----------")
#     print(response['message']['content'])

# print("Input: ")
# user_input = input()
# response = ask_llm(user_input)