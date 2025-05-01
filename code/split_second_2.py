import copy
import os
import argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from utils import read_json, create_json

def parse_args():
    parser = argparse.ArgumentParser(description="Embedding similarity stage for updating results")
    parser.add_argument('--dataset_names', type=str, nargs='+', required=True, help="List of dataset names")
    parser.add_argument('--model_names', type=str, nargs='+', required=True, help="List of model names")
    parser.add_argument('--base_input_path', type=str, default="../eval/mid", help="Directory where intermediate JSON files are stored")
    parser.add_argument('--output_dir', type=str, default="../eval/ours_results", help="Directory to save final updated JSONs")
    parser.add_argument('--threshold_range', type=float, nargs=3, 
                       default=[0.0, 1.0, 0.1],
                       help="Start, end and step for threshold exploration")
    return parser.parse_args()

def make_thresholds(start, end, step):
    return [round(x, 2) for x in np.arange(start, end + 1e-8, step)]

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading embedding model...")
    embedding_model = SentenceTransformer("/home/yzl/yzl/model/all-MiniLM-L6-v2")

    thresholds = make_thresholds(*args.threshold_range)
    for dataset_name in args.dataset_names:
        for model_name in args.model_names:
            print(f"Processing dataset: {dataset_name}, model: {model_name}")

            model_tag = model_name.replace(":", "_").replace("/", "_")
            input_file = os.path.join(args.base_input_path, f"mid_{dataset_name}_{model_tag}_ollama_splitsentence.json")
            data = read_json(input_file)
            if data is None:
                continue

            for threshold in thresholds:
                threshold_data = copy.deepcopy(data)
                for item in tqdm(threshold_data, desc=f"Processing {dataset_name}-{model_tag}", ncols=100):
                    if item.get('result') is not None:
                        continue

                    prompt_emb = embedding_model.encode(item['prompt'], normalize_embeddings=True)

                    sentence_results = []

                    for i, (sent, safety, has_verb) in enumerate(zip(item['sentences'], item['safety'], item['sentence_verbs'])):
                        if safety == '0' and has_verb:
                            text_without = item['prompt'].replace(sent, '')
                            emb_sent = embedding_model.encode(sent)
                            emb_without = embedding_model.encode(text_without)

                            # similarity_unsafe = emb_sent @ prompt_emb.T
                            # similarity_without = emb_without @ prompt_emb.T
                            similarity_unsafe = embedding_model.similarity(emb_sent, prompt_emb)
                            similarity_without = embedding_model.similarity(emb_without, prompt_emb)
                            
                            score = (1 - similarity_unsafe.item()) * 0.5 + similarity_without.item() * 0.5

                            sentence_results.append(score > threshold)
                        else:
                            sentence_results.append(True)

                    true_count = sentence_results.count(True)
                    false_count = sentence_results.count(False)

                    item['result'] = (true_count >= false_count)
                    item['sentence_results'] = sentence_results
                    item['threshold'] = threshold

                output_file = os.path.join(args.output_dir, f"final_{dataset_name}_{model_tag}_{threshold:.2f}_all_ollama_splitsentence.json")

                create_json(threshold_data, output_file, indent=4)

            print(f"Finished {dataset_name}-{model_name}-{threshold:.2f}, results saved to {output_file}")

    print("All datasets and models processed successfully!")

if __name__ == "__main__":
    main()

