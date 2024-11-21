from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
import time

import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()


class Model:
    def __init__(self, model_name, emb_model='nomic-embed-text'):
        self.model_name = model_name
        self.emb_model = emb_model
        self.model_local = self.load_model()
        self.chunk_size, self.chunk_overlap = 7500, 100
        self.context_preprocessed = False
        
        self.context = ""
        self.vectorstore = None
        self.retriever = None

        self.questions_history = []
        self.answer_history = []
        self.runtime_history = []
        
    def get_method_name(self):
        return self.model_name
    
    def get_method_details(self):
        return {"method_name": self.model_name,
                "emb_model": self.emb_model,
                "chunk_size": self.chunk_size, 
                "chunk_overlap": self.chunk_overlap}
    
    def load_model(self):
        return Ollama(model=self.model_name)
    
    def preprocess(self, raw_doc, db_name='rag_chroma'):
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.context = text_splitter.split_documents(raw_doc)
        #convert text chunks into embeddings and store in vector database 
        self.vectorstore = Chroma.from_documents( # maybe smth wrong with loading
            documents=self.context,
            collection_name=db_name,
            embedding=OllamaEmbeddings(model=self.emb_model)
        )

        self.retriever = self.vectorstore.as_retriever()
        self.context_preprocessed = True
    

    def process_input(self):
        after_rag_template = """<|im_start|>system
        Answer the question based only on the provided context. Think twice and provide short accurate answer. Answer must follow the next STRICT roles:
        1. Answer must be SUPPER SHORT, as short as needed. 
        2. Additional explanation are prohibited.
        3. During your evaluation you will be penalised for each redundant word in your answer
        4. If you not sure or dont know say 'I don't know'
        Context: {context}
        <|im_end|>
        <|im_start|>user
        Give me short answer to question: {question} <|im_end|>
        <|im_start|>assistant

        """

        after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
        after_rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | after_rag_prompt
            | self.model_local
            | StrOutputParser()
        )
        results = after_rag_chain.invoke(self.questions_history[-1])
        return results

    def run(self, raw_doc, question, db_name='rag_chroma', preprocess=True):
        self.questions_history.append(question)
        if preprocess:
            self.preprocess(raw_doc, db_name)
        start_time = time.time()
        answer = self.process_input()
        self.runtime_history.append(time.time() - start_time)
        self.answer_history.append(answer)
        return answer

