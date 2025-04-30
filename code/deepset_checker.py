from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import json

tokenizer = AutoTokenizer.from_pretrained("deepset/deberta-v3-base-injection")
model = AutoModelForSequenceClassification.from_pretrained("deepset/deberta-v3-base-injection")

classifier = pipeline(
  "text-classification",
  model=model,
  tokenizer=tokenizer,
  truncation=True,
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

json_file = '../datasets/.json'
output_file = '../final/other_results/NotInject_three_deepset.json'
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
prompts = [item['prompt'] for item in data]
model_response = classifier(prompts, batch_size=2)
model_result = []
for prompt, response in zip(prompts, model_response): 
    if response['label'] == 'INJECTION':
        model_result.append({'prompt': prompt, 'raw': response, 'safety': 0})
    else:
        model_result.append({'prompt': prompt, 'raw': response, 'safety': 1})

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(model_result, f, ensure_ascii=True, indent=4)