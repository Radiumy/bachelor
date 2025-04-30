import csv
import json
import os
from utils import read_csv, create_json
config = {
    "output_dir": "../datasets",
    "input_file": "../datasets/jailbreak_dataset.csv"

}

def read_csv(input_path):
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'prompt': row['prompt'],
                'label': row['type']
            })
    return data
def main():
    input_file = config["input_file"]
    input_file_base = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(config["output_dir"], f"{input_file_base}.json")
    
    data = read_csv(input_file)

    create_json(data, output_file)
    print(f"Successfully saved: {output_file}")


if __name__ == "__main__":
    main()