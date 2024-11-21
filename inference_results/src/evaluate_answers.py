import re
import glob
import json
import string
import pandas as pd
from anls import anls_score
from pathlib import Path


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    def remove_tags(text):
        #text = text.replace("<|im_start|>", "")
        #text = text.replace("</|im_end|>", "")
        text = text.replace("im_start", "")
        text = text.replace("im_end", "")
        text = text.replace("Answer", "")
        text = text.replace("\n", "")
        text = text.replace("\t", "")
        return text

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(remove_tags(lower(s)))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_ansl_score(eval_file, display_score=False):
    docs = list(eval_file.keys())
    results = dict()

    for doc in docs:
        results[str(Path(doc))] = list()
        ans, gt_ans = eval_file[doc]["qa_pairs"][1:]
        for answer, gt_answer in zip(ans, gt_ans):
            norm_answer = normalize_text(answer)
            norm_gt_gt_answer = normalize_text(gt_answer)
            score = anls_score(prediction=norm_answer, gold_labels=[norm_gt_gt_answer])
            if display_score and score == 0.:
                #print(f"Ans: {answer}, GT_Ans {gt_answer}")
                print(f"Ans: {norm_answer}, GT_Ans {norm_gt_gt_answer}")
            results[doc].append(score)
    return results


if __name__ == "__main__":
    json_folder = str(Path(__file__).absolute().parent.parent)
    eval_files_path = glob.glob(f"{json_folder}/*.json")
    display_model = "mistral:7b-instruct-q4_0.json"
   
    results = dict()

    for path in eval_files_path:
        if not (Path(path).name==display_model):
            continue
        with open(path, "r") as f:
            eval_file = json.load(f)
        scores = compute_ansl_score(eval_file, display_score=(Path(path).name==display_model))
        model_name = path.split("/")[-1].lower()
        results[model_name] = scores
    sorted_dict = dict(sorted(results.items()))
    aggregeted_results = dict()
    rounded_results = dict()
    for model, metrics in sorted_dict.items():
        aggregeted_results[model] = 0
        rounded_results[model] = 0
        for page, value in metrics.items():
            aggregeted_results[model] += sum(value)
            rounded_results[model] += sum([1 for v in value if v > 0])

            

    #print(json.dumps(sorted_dict['gemma2:2b.json'], indent=4))
    print(json.dumps(aggregeted_results, indent=4))
    print(json.dumps(rounded_results, indent=4))
    #print(results)

