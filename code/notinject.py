import csv
import json
import os
from utils import create_json
config = {
    "output_dir": "../datasets",
    "input_file": ["../datasets/NotInject_one.json", "../datasets/NotInject_two.json", "../datasets/NotInject_three.json"]

}

def read_json(input_path):
    data = []
    with open(input_path, mode='r', encoding='utf-8') as file:
        json_data = json.load(file)
        for item in json_data:
            data.append({
                'prompt': item['prompt'],
                'label': "benign"
            })
    return data
def main():
    for input_file in config["input_file"]:
       
        input_file_base = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(config["output_dir"], f"{input_file_base}.json")
        
        data = read_json(input_file)

        create_json(data, output_file)
        print(f"Successfully saved: {output_file}")


if __name__ == "__main__":
    main()