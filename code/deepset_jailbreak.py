from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from utils import read_csv, read_json, create_json
import csv
import json

tokenizer = AutoTokenizer.from_pretrained("deepset/deberta-v3-base-injection")
model = AutoModelForSequenceClassification.from_pretrained("deepset/deberta-v3-base-injection")

classifier = pipeline(
  "text-classification",
  model=model,
  tokenizer=tokenizer,
  truncation=True,
  max_length=512,
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

input_file = '../datasets/jailbreak_dataset.csv'
output_file = '../final/other_results/jailbreak_dataset_deepset.json'
data = read_csv(input_file)
prompts = [item['prompt'] for item in data]
model_response = classifier(prompts, batch_size=2)
model_result = []
for prompt, response in zip(prompts, model_response): 
    print(response)
    if response['label'] == 'INJECTION':
        model_result.append({'prompt': prompt, 'raw': response, 'safety': 0})
    else:
        model_result.append({'prompt': prompt, 'raw': response, 'safety': 1})
create_json(model_result, output_file, indent=4)