#NOTE: I BASICALLY JUST COPIED AND PASTED THE COLAB PARTS RELATED TO INDEXING

import pyterrier as pt
import pandas as pd
import json
import shutil
import os
from functions import keywords_extractor, thesaurus_based_expansion
from tqdm import tqdm
from pyterrier_dr import FlexIndex, RetroMAE

tqdm.pandas()

#Get the dataframe to index
with open("document_collection.json", "r", encoding = "utf-8") as file:
    data = json.load (file)
df = pd.DataFrame.from_dict(data)


# Prepare dataframe for PyTerrier: needs columns 'docno' and 'text'
corpus_dataframe = df.rename(columns={"para_id": "docno", "context": "text"})[["docno", "text"]]
corpus_dataframe = corpus_dataframe.to_dict(orient="records")

# build an indexing pipeline that first applies RetroMAE to get dense vectors, then indexes them into the FlexIndex
dense_index = FlexIndex('dense_index.flex', verbose = 1)

# Which model should we use to get the embedding representation of the documents (remember we are offline)
model = RetroMAE.msmarco_distill()

# How do we create the index pipeline?
offline_indexing_pipeline = model >> dense_index.indexer(mode="overwrite")

# How do we apply the pipeline on our documents?
offline_indexing_pipeline.index(corpus_dataframe)