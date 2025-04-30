from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import json

tokenizer = AutoTokenizer.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection-v2")
model = AutoModelForSequenceClassification.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection-v2")

classifier = pipeline(
  "text-classification",
  model=model,
  tokenizer=tokenizer,
  truncation=True,
  max_length=512,
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

json_file = '../datasets/NotInject_three.json'
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
prompts = [item['prompt'] for item in data]
model_response = classifier(prompts, batch_size=2)
model_result = []
for prompt, response in zip(prompts, model_response): 
    if response['label'] == 'INJECTION':
        model_result.append({'prompt': prompt, 'safety': 0})
    elif response['label'] == 'SAFE':
        model_result.append({'prompt': prompt, 'safety': 1})
output_file = '../other_results/NotInject_three_protectaiv2.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(model_result, f, ensure_ascii=True, indent=4)

# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# import torch
# import json
# import os

# # 检查文件路径
# json_file = '../injecGuard/datasets/NotInject_three.json'
# print(f"Checking if file exists: {os.path.exists(json_file)}")

# # 初始化模型
# try:
#     tokenizer = AutoTokenizer.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection-v2")
#     model = AutoModelForSequenceClassification.from_pretrained("ProtectAI/deberta-v3-base-prompt-injection-v2")
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device set to: {device}")
    
#     # 显式将模型移动到设备
#     model = model.to(device)
    
#     classifier = pipeline(
#         "text-classification",
#         model=model,
#         tokenizer=tokenizer,
#         truncation=True,
#         max_length=512,
#         device=device,
#         batch_size=2  # 减小batch_size
#     )

#     # 加载数据
#     with open(json_file, 'r', encoding='utf-8') as f:
#         data = json.load(f)
    
#     print(f"Loaded {len(data)} items from JSON file")
#     prompts = [item['prompt'] for item in data]  # 注意这里可能是拼写错误，应该是'prompt'还是'prompt'?
    
#     # 分批处理
#     model_response = classifier(prompts)
#     print("Classification completed")
    
#     model_result = []
#     for prompt, response in zip(prompts, model_response): 
#         if response['label'] == 'INJECTION':
#             model_result.append({'prompt': prompt, 'safety': 0})
#         elif response['label'] == 'SAFE':
#             model_result.append({'prompt': prompt, 'safety': 1})
    
#     output_file = '../data/NotInject_three_protectaiv2.json'
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(model_result, f, ensure_ascii=False, indent=4)
    
#     print("Processing completed successfully")

# except Exception as e:
#     print(f"An error occurred: {str(e)}")
#     raise