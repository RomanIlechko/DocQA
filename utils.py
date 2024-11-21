import time
import json
from datasets import load_dataset
from PIL import Image

LLM_MODELS = (
        "nemotron-mini",
        "mistral",
        "llama2:7b-chat-q4_0",
        "llama3.1:8b",
        "llama3.2:1b",
        "gemma2:2b",
        "qwen2.5:0.5b",
        "qwen2.5:1.5b",
        "phi3.5",
        "elvee/capybarahermes-2.5-mistral:7b-q4_K_M",
        "Qwen2-7B-Instruct-Q4_K_M:latest")

def load_image(path):
    return Image.open(path)


def get_k_test_samples(k=100, dataset_path="nielsr/docvqa_1200_examples"):
    # sample N1 espetialy nice
    dataset = load_dataset(dataset_path)
    updated_dataset = dataset.map(lambda example: {"question": example["query"]["en"]}, remove_columns=["query"])
    return updated_dataset["test"][:k]


def read_questions(path):
    with open(path, "r") as f:
        if ".json" not in path:
            questions = f.read().split("\n")
        else:
            questions = json.load(f)
        return questions


def measure_inference(model, raw_doc, questions, db_name, k=10):
    total_runs = len(questions)*k
    preprocess = True
    all_timings = []
    all_qa_pairs = []
    
    for run_i in range(1, k + 1):
        for q_i, question in enumerate(questions):
            model.run(raw_doc, question, db_name=db_name, preprocess=preprocess)
            all_timings.append(model.runtime_history[-1])
            preprocess = False
    return all_timings

def save_inference_results(path, results):
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

