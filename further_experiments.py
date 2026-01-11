import json
from operator import index
import os
import pandas as pd
import pyterrier as pt
from eval_metrics import EVAL_METRICS
from functions import keywords_extractor, thesaurus_based_expansion
from tqdm import tqdm
tqdm.pandas()
from baseline_experiments import BaselineExperiments
from pyterrier_dr import FlexIndex, RetroMAE



"""
CHILD CLASS OF BaselineExperiments
+ complete_index_path ="keywords_and_text_index" attribute
+ load_complete_index(self) method
+ run_further_experiment_1(self) method
"""

class FurtherExperiments(BaselineExperiments):

    def __init__(         
        self,
        queries_path,
        qrels_path,
        base_index_path="terrier_index",
        keywords_index_path="keywords_index",
        complete_index_path ="keywords_and_text_index"

    ):
    
        # Call the parent constructor
        super().__init__(
            queries_path,
            qrels_path,
            base_index_path,
            keywords_index_path
        )
        self.complete_index_path = complete_index_path
        self.load_complete_index()

    def load_complete_index(self):
        index_abs_path = os.path.abspath(self.complete_index_path)
        self.complete_index = pt.IndexFactory.of(index_abs_path)
    

    def run_further_experiment_1(self):
        # try bo1 on same setting as baseline_3 instead of rm3
        if not self.queries_expanded:
            self.expanded_queries_small = self.thesaurus_query_expansion(self.queries_small)

        bm_25 = pt.terrier.Retriever(self.keywords_index, wmodel="BM25")
        bo1_pipe_bm25 = bm_25 >> pt.rewrite.Bo1QueryExpansion(self.keywords_index) >> bm_25

        experiment3_results_a = pt.Experiment(
            [bm_25, bo1_pipe_bm25],
            self.queries_small,
            self.qrels,
            EVAL_METRICS
            )
        print("Further experiment 1 (not expanded queries):\n", experiment3_results_a)

        experiment3_results_b = pt.Experiment(
            [bm_25, bo1_pipe_bm25],
            self.expanded_queries_small,
            self.qrels,
            EVAL_METRICS
            )
        print("Further experiment 1 results (expanded queries):\n", experiment3_results_b)    


    def run_further_experiment_2(self):
        # try bo1 on same setting as baseline_3 instead of rm3
        # but use complete_index for matching keywords, and adding text from text_field
        print("NUMBER OF FIELDS", self.complete_index.getCollectionStatistics().getNumberOfFields()) # will print 2
        print("NAME OF FIELDS", self.complete_index.getCollectionStatistics().getFieldNames())
        print("USING 1000 queries")
        print("USING QUERIES EXPANDED WITH KEYWORDS ADDED")
        
        if not self.queries_expanded:
            self.expanded_queries_small = self.thesaurus_query_expansion(self.queries_small)
        # try bo1 on same setting as baseline_3 instead of rm3
        # but use complete_index for matching keywords, and adding text from text_field
        
        print()

        
        bm_25_simple = pt.terrier.Retriever(
            self.complete_index, 
            wmodel="BM25"
        )

        bm25_nofielded = pt.Experiment(
            [bm_25_simple],
            self.expanded_queries_small,
            self.qrels,
            EVAL_METRICS,
            names=["bm_25_simple"],
            save_dir="./",
            save_mode='reuse'
        )

        print("BM25 base, no fields:\n", bm25_nofielded)   


        # first matching only with keywords
        BM25F_keywords = pt.terrier.Retriever(
            self.complete_index,
            wmodel="BM25F",
            controls = {'w.0' : 0, 'w.1' : 1}
        )

        BM25F_keywords_results = pt.Experiment(
            [BM25F_keywords],
            self.expanded_queries_small,
            self.qrels,
            EVAL_METRICS,
            names=["BM25F_keywords"],
            save_dir="./",
            save_mode='reuse'
        )
        
        print("BM25F on keywords only:\n", BM25F_keywords_results) 

        BM25F_text = pt.terrier.Retriever(
            self.complete_index,
            wmodel="BM25F",
            controls = {'w.0' : 1, 'w.1' : 0},
        )

        BM25F_text_results = pt.Experiment(
            [BM25F_text],
            self.expanded_queries_small,
            self.qrels,
            EVAL_METRICS,
            names=["BM25F_text"],
            save_dir="./",
            save_mode='reuse'
        )

        print("BM25 on text only:\n", BM25F_text_results) 
 

        BM25F_both = pt.terrier.Retriever(
            self.complete_index, 
            wmodel="BM25F",
            controls = {'w.0' : 0.6, 'w.1' : 0.4}
        )

        BM25F_both_results = pt.Experiment(
            [BM25F_both],
            self.expanded_queries_small,
            self.qrels,
            EVAL_METRICS,
            names=["BM25F_both"],
            save_dir="./",
            save_mode='reuse'
        )
        
        print("BM25 on both text 0.6 and kwywords 0.4:\n", BM25F_both_results)   
        


        #####################

        print("Let's try Query expansion with bo1!")
        print()

        qe = pt.terrier.Retriever(
            self.complete_index,
            wmodel="DPH", 
            controls =
                {"qemodel" : "Bo1", 
                "qe" : "on", 
                'w.0' : 1, 'w.1' : 0}
        )

        bo1_field_text_pipe = BM25F_text >> qe >> BM25F_text
        
        bo1_field_text_results = pt.Experiment(
            [bo1_field_text_pipe],
            self.expanded_queries_small,
            self.qrels,
            EVAL_METRICS,
            names=["bo1_field_text_pipe"],
            save_dir="./",
            save_mode='reuse'
        )
        
        print("br25f on text, bo1 on text, br25f on text:\n", bo1_field_text_results)  


        bo1_field_keywords_text_pipe = BM25F_keywords >> qe >> BM25F_text

        bo1_field_keywords_text_results = pt.Experiment(
            [bo1_field_keywords_text_pipe],
            self.expanded_queries_small,
            self.qrels,
            EVAL_METRICS,
            names=["bo1_field_keywords_text_pipe"],
            save_dir="./",
            save_mode='reuse'
        )

        print("br25f on keywords, bo1 on text, br25f on text:\n", bo1_field_keywords_text_results)  

        bo1_field_both_pipe = BM25F_both >> qe >> BM25F_text

        bo1_field_both_pipe_results = pt.Experiment(
            [bo1_field_both_pipe],
            self.expanded_queries_small,
            self.qrels,
            EVAL_METRICS,
            names=["bo1_field_both_pipe"],
            save_dir="./",
            save_mode='reuse'
        )

        print("br25f on both fields 0.6 text 0.4 keywords, bo1 on text, br25f on text:\n", bo1_field_both_pipe_results)  


    def run_further_experiment_3(self):
    
        # build an indexing pipeline that first applies RetroMAE to get dense vectors, then indexes them into the FlexIndex
        dense_index = FlexIndex('dense_index.flex')

        # Which model should we use to get the embedding representation of the documents (remember we are offline)
        model = RetroMAE.msmarco_distill()

        bm25 = pt.terrier.Retriever(
            self.base_index,
            wmodel="BM25"
        )


        retrieval_pipeline_plainbi = model >> dense_index.retriever()
        retrieval_pipeline_with_bm25 = (bm25 % 100) >> retrieval_pipeline_plainbi

        bineural_results = pt.Experiment(
            [retrieval_pipeline_plainbi, retrieval_pipeline_with_bm25],
            self.expanded_queries_small,
            self.qrels,
            EVAL_METRICS,
            names=["bineural", "bm25_bineural"],
            save_dir="./",
            save_mode='reuse'
        )

        print("br25f on both fields 0.6 text 0.4 keywords, bo1 on text, br25f on text:\n", bineural_results)  


        ## What is the issue? This index requires the dataset to have docno and text fields! So, you have to make sure to add these.




fe = FurtherExperiments("test_queries.json", "test_qrels.json")
#fe.run_further_experiment_1()
fe.run_further_experiment_2()



"""
    properties={
        "index.document.class": "org.terrier.structures.FSAFieldDocumentIndex"
    }
"""