import tkinter as tk
from tkinter import filedialog, ttk, Text
from langchain.document_loaders import TextLoader
from llm import Model
from utils import LLM_MODELS

def get_result(document, question, method):
    return "Analysis result goes here.\n"

def reload_model(current_model_name, model):
    if hash(current_model_name) == hash(model.get_method_name()):
        return False
    else:
        return True


class DocumentAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Analysis App")
        self.root.geometry("600x400")
        
        self.methods = self._get_methods()
        self.question_welcome_str = "Enter your question here..."
        self.document_content = ""

        self._configure_grid()

        # Initialize variables
        self.analysis_method = self._get_default_method()
        print("analysis_method", self.analysis_method.get())
        self.model = Model(self.analysis_method.get())
        
        # Create UI elements
        self._create_widgets()

    def _configure_grid(self):
        """Configure the grid layout to allow resizing."""
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
    
    def _get_methods(self):
        return LLM_MODELS

    def _get_default_method(self, default_method_id=0):
        return tk.StringVar(value=self.methods[default_method_id])

    def _create_widgets(self):
        """Create and place all widgets in the UI."""
        # Create dropdown menu for analysis methods
        self.method_dropdown = ttk.Combobox(
            self.root, textvariable=self.analysis_method, width=20, state="readonly"
        )
        self.method_dropdown['values'] = self.methods
        self.method_dropdown.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Create load button
        self.load_button = tk.Button(self.root, text="Load", command=self.load_document)
        self.load_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # Create analyze button
        self.analyze_button = tk.Button(self.root, text="Analyze", command=self.analyze_document)
        self.analyze_button.grid(row=0, column=2, padx=10, pady=10, sticky="ew")

        # Create an entry field for user input (question)
        self.question_entry = tk.Entry(self.root, width=60)
        self.question_entry.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        self.question_entry.insert(0, self.question_welcome_str)

        # Create result display area
        self.result_area = Text(self.root, wrap=tk.WORD)
        self.result_area.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

    def load_document(self):
        """Load a document and display its content in the result area."""
        file_path = filedialog.askopenfilename()
        
        if file_path:
            self.document_content  = TextLoader(file_path).load()
            self._clear_result_area()
            self.result_area.insert(tk.END, "Document Loaded \n")

    def analyze_document(self):
        """Analyze the loaded document based on the selected method and question."""
        question = self.question_entry.get().replace(self.question_welcome_str, "")
        if self.document_content and question:
            method = self.analysis_method.get()
            if reload_model(method, self.model):
                self.model = Model(method)
            self.result_area.insert(tk.END, f"\n\nAnalyzing document with method: {method}...\n")
            self.result_area.insert(tk.END, f"Processing question: {question}\n")
            # Placeholder for actual analysis logic
            # answer = get_result(self.document_content, question, method)
            answer = self.model.run(self.document_content, question)
            processing_time = self.model.runtime_history[-1]
            self.result_area.insert(tk.END, f"Answer: {answer}, processing_time: {processing_time}")
        else:
            self.result_area.insert(tk.END, "Please load a document, enter a question, and select an analysis method.\n")

    def _clear_result_area(self):
        """Clear the result area text."""
        self.result_area.delete(1.0, tk.END)


# Initialize the application
if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentAnalysisApp(root)
    root.mainloop()
