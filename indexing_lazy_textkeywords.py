#NOTE: I BASICALLY JUST COPIED AND PASTED THE COLAB PARTS RELATED TO INDEXING

import pyterrier as pt
import pandas as pd
import json
import shutil
import os


#Get the dataframe to index
with open("corpus_with_keywords.json", "r", encoding = "utf-8") as file:
    data = json.load (file)
corpus_dataframe = pd.DataFrame.from_dict(data)


# Prepare dataframe for PyTerrier: needs columns 'docno' and 'text'
longest_len = corpus_dataframe["docno"].str.len().max()
print(corpus_dataframe["docno"])

#stats of keywords
stats = corpus_dataframe["keywords"].str.split().str.len().agg( min_words="min", avg_words="mean", max_words="max", )
print(stats)


# Create or reset an index folder
index_path = os.path.abspath("keywords_and_text_index")  # absolute path
if os.path.exists(index_path):
    shutil.rmtree(index_path)
os.makedirs(index_path, exist_ok=True)


# Build the indexer using the pt.IterDictIndexer 
# Store docno as metadata so we can recover it later if needed, do we remember why we did it?
# Key parameters now are: meta, text_attrs, meta_reverse, pretokenised, fields, threads
indexer = pt.IterDictIndexer(
    index_path,
    meta={"docno": longest_len},  #TO CHECK          # store docno as metadata (up to 200 characters)
    text_attrs=["text", "keywords"],           # which field(s) contain the text
    meta_reverse=["docno"],        # enable reverse lookup on docno
    pretokenised=False,
    threads=1, 
    fields=True,
    properties={
        "index.document.class":
        "org.terrier.structures.FSAFieldDocumentIndex"
    }
)



#perform the indexing and assign 
index_ref = indexer.index(corpus_dataframe.to_dict(orient="records"))
#index_ref is not the index itself, but a reference pointing to where the index was created

# Open the index to ensure it is valid
index = pt.IndexFactory.of(index_ref)

print(index.getCollectionStatistics().toString())

# Print a simple summary
print("Index location:", index_path)
print("Indexed documents:", index.getCollectionStatistics().getNumberOfDocuments())
