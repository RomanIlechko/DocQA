import glob
import json
import statistics as stat
from pathlib import Path

def collect_timings(eval_file):
    docs = list(eval_file.keys())
    timings = list()

    for doc in docs:
        timings.extend(eval_file[doc]["timings"])
    return timings


if __name__ == "__main__":
    json_folder = str(Path(__file__).absolute().parent.parent)
    eval_files_path = glob.glob(f"{json_folder}/*.json")
    results = dict()

    for path in eval_files_path:
        with open(path, "r") as f:
            eval_file = json.load(f)
        timitngs = collect_timings(eval_file)
        model_name = path.split("/")[-1].lower()
        results[model_name] = dict()
        results[model_name]["AVG"] = stat.mean(timitngs)
        results[model_name]["MED"] = stat.median(timitngs)
    #sorted_dict = dict(sorted(results.items()))
    print(json.dumps(dict(sorted(results.items())), indent=4))
    #print(results)

