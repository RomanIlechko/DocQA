import glob
from llm import Model
from donut import DonutModel
from utils import LLM_MODELS, load_image
from pathlib import Path
from utils import measure_inference, read_questions, save_inference_results
from langchain.document_loaders import TextLoader

def text_loader(path):
    return TextLoader(path).load()


if __name__ == "__main__":
    OCR_FREE = False
    MODEL_PATH = "mistral:7b-instruct-q4_0"
    curr_dir = Path(__file__).absolute().parent
    q_path = str(curr_dir / "synth_data/datasets/QA_synth_db.json")
    if OCR_FREE:
        doc_folder_path = curr_dir / "synth_data/datasets/synth_db_images"
        doc_file_paths = glob.glob(str(doc_folder_path / "*.JPG"))
        VQAModel = DonutModel
        loader = load_image
        MODEL_NAME = str(Path(MODEL_PATH).name)
    else:
        doc_folder_path = curr_dir / "synth_data/datasets/synth_db_pages"
        doc_file_paths = glob.glob(str(doc_folder_path / "*.txt"))
        VQAModel = Model
        loader = text_loader
        MODEL_NAME = MODEL_PATH
    results = dict()

    print("Read questions")
    questions = read_questions(q_path)
    print("Load model")
    model = VQAModel(MODEL_PATH)

    for doc_file_path in doc_file_paths:
        print("Load document")
        doc_name = doc_file_path.split("/")[-1]
        doc_name = doc_name.replace(".JPG", ".txt")
        questions_list = [questions[doc_name][f"question_{i}"] for i in range(1, 4)]
        gt_ans_list = [questions[doc_name][f"answer_{i}"] for i in range(1, 4)]

        document_content  = loader(doc_file_path)

        print("Make measurements")
        all_timings = measure_inference(model, document_content, questions_list, doc_name, k=1)
        results[doc_name] = dict()
        results[doc_name]["timings"] = all_timings
        results[doc_name]["qa_pairs"] = (questions_list, model.answer_history[-len(questions_list):], gt_ans_list)

    save_inference_results(f"{curr_dir / 'inference_results' / (MODEL_NAME + '.json')}", results)
