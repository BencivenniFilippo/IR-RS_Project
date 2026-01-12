from pathlib import Path
import json
import pandas as pd
from constants import RANDOM_STATE

class BenchmarkCollection():
    def __init__(self, documents_path="document_collection.json", queries_path="test_queries.json", qrels_path="test_qrels.json"):
        self.documents_path = Path(documents_path)
        if not self.documents_path.is_file():
            raise FileNotFoundError(f"Documents file does not exist: {self.documents_path}")
        self.queries_path = Path(queries_path)
        if not self.queries_path.is_file():
            raise FileNotFoundError(f"Queries file does not exist: {self.queries_path}")
        self.qrels_path = Path(qrels_path)
        if not self.qrels_path.is_file():
            raise FileNotFoundError(f"Qrels file does not exist: {self.qrels_path}")

    def load_documents(self):
        with self.documents_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        df = pd.DataFrame.from_dict(data)
        self.corpus_dataframe = df.rename(columns={"para_id": "docno", "context": "text"})[["docno", "text"]]
        print("Documents loaded successfully.")

    def load_queries(self):
        with self.queries_path.open('r', encoding='utf-8') as file:
            data = json.load(file)
        self.queries = pd.DataFrame.from_dict(data)
        self.queries.rename(columns={"query_id": "qid", "question": "query"}, inplace=True)
        print("Queries loaded successfully.")
        
    def load_qrels(self):
        with self.qrels_path.open('r', encoding='utf-8') as file:
            data = json.load(file)
        self.qrels = pd.DataFrame.from_dict(data)
        self.qrels.rename(columns={"query_id": "qid", "para_id": "docno"}, inplace=True)
        print("Qrels loaded successfully.")
    
    def sample_queries(self, n=1000, random_state=RANDOM_STATE):
        if not hasattr(self, "queries"):
            raise RuntimeError("Queries not loaded. Call load_queries() first.")
        self.queries_sample = self.queries.sample(n=min(n, len(self.queries)), random_state=random_state)
        print(f"Sampled {len(self.queries_sample)} queries for testing.")
    
    def corpus_summary(self, top_n=5):
        if not hasattr(self, "corpus_dataframe"):
            raise RuntimeError("Documents not loaded.")
        print(f"Number of documents: {len(self.corpus_dataframe)}")
        print("Sample documents:")
        print(self.corpus_dataframe.head(top_n))
    
    def queries_summary(self, top_n=5):
        if not hasattr(self, "queries"):
            raise RuntimeError("Queries not loaded.")
        print(f"Number of queries: {len(self.queries)}")
        print("Sample queries:")
        print(self.queries.head(top_n))

    def qrels_summary(self, top_n=5):
        if not hasattr(self, "qrels"):
            raise RuntimeError("Qrels not loaded.")
        print(f"Number of relevance judgments: {len(self.qrels)}")
        print("Sample qrels:")
        print(self.qrels.head(top_n))