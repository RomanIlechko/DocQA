import re
import time
from utils import load_image

from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_dataset

class DonutModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = "cpu"
        self.processor, self.donut_model = self.load_model()
        self.donut_model.to(self.device)
        self.context = None
        self.context_preprocessed = False

        self.questions_history = []
        self.answer_history = []
        self.runtime_history = []
        
    def get_method_name(self):
        return self.model_path
    
    def get_method_details(self):
        return {"method_name": self.model_path}
    
    def load_model(self):
        return DonutProcessor.from_pretrained(self.model_path), VisionEncoderDecoderModel.from_pretrained(self.model_path)
    
    def preprocess(self, image):
        self.context = self.processor(image, return_tensors="pt").pixel_values
        self.context_preprocessed = True

    def process_input(self):
        pass

    def run(self, image, question, db_name='rag_chroma', preprocess=True):
        self.questions_history.append(question)
        if preprocess:
            self.preprocess(image)
        
        task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
        prompt = task_prompt.replace("{user_input}", question)
        decoder_input_ids = self.processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        
        start_time = time.time()
        outputs = self.donut_model.generate(
            self.context.to(self.device),
            decoder_input_ids=decoder_input_ids.to(self.device),
            max_length=self.donut_model.decoder.config.max_position_embeddings,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        answer = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
        
        self.runtime_history.append(time.time() - start_time)
        self.answer_history.append(answer)
        return answer
