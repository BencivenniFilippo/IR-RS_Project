from collection import BenchmarkCollection
import pyterrier as pt
from tqdm import tqdm
from pathlib import Path
from functions import keywords_extractor, thesaurus_based_expansion
from constants import BASIC_INDEX_NAME, KEYWORDS_INDEX_NAME, TWO_FIELDS_INDEX_NAME, INDEXES_FOLDER, DENSE_INDEX_NAME
from pyterrier_dr import FlexIndex, RetroMAE

class BenchmarkIndex():
    def __init__(self, collection: BenchmarkCollection, indexes_folder=INDEXES_FOLDER):
        self.collection = collection
        self.indexes_folder = Path(indexes_folder).resolve()
        self.create_indexes_folder()
    
    def create_indexes_folder(self):
        self.indexes_folder.mkdir(parents=True, exist_ok=True)

    def create_basic_index(self):
        if not hasattr(self.collection, "corpus_dataframe"):
            raise RuntimeError("Documents not loaded. Call load_documents() before creating an index.")
        longest_len = self.collection.corpus_dataframe["docno"].str.len().max()
        longest_txt = self.collection.corpus_dataframe["text"].str.len().max()
        
        # Create index or raise error if it exists
        basic_index_path = self.indexes_folder / BASIC_INDEX_NAME
        if basic_index_path.exists():
            raise RuntimeError(f"Index already exists: {basic_index_path}")
        basic_index_path.mkdir(parents=True)

        # Build the indexer using the pt.IterDictIndexer
        indexer = pt.IterDictIndexer(
        str(basic_index_path),
        meta={"docno": longest_len, "text": longest_txt},  
        text_attrs=["text"],    # which field(s) contain the text
        meta_reverse=["docno"], # enable reverse lookup on docno
        pretokenised=False,
        fields=False,
        threads=1, 
        )

        index_ref = indexer.index(self.collection.corpus_dataframe.to_dict(orient="records"))

        # Open the index to ensure it is valid
        index = pt.IndexFactory.of(index_ref)

        # Print a simple summary
        print("Index location:", basic_index_path)
        print("Indexed documents:", index.getCollectionStatistics().getNumberOfDocuments())
    
    def create_keywords_expanded_index(self):
        if not hasattr(self.collection, "corpus_dataframe"):
            raise RuntimeError("Documents not loaded. Call load_documents() before creating an index.")
        longest_len = self.collection.corpus_dataframe["docno"].str.len().max()

        # Create index or raise error if it exists
        keywords_expanded_index_path = self.indexes_folder / KEYWORDS_INDEX_NAME
        if keywords_expanded_index_path.exists():
            raise RuntimeError(f"Index already exists: {keywords_expanded_index_path}")
        keywords_expanded_index_path.mkdir(parents=True)

        corpus_dataframe = self.collection.corpus_dataframe.copy() # Make a copy to avoid modifying the original   
        # Expand documents with keywords and synonyms
        tqdm.pandas(desc="Expanding documents with keywords and synonyms")
        corpus_dataframe["text"] = corpus_dataframe["text"].progress_apply(
            lambda q: " ".join(
                thesaurus_based_expansion(q, keywords_extractor(q)),
            )
        )

        # Build the indexer using the pt.IterDictIndexer
        indexer = pt.IterDictIndexer(
            str(keywords_expanded_index_path),
            meta={"docno": longest_len},
            text_attrs=["text"],    # which field(s) contain the text
            meta_reverse=["docno"], # enable reverse lookup on docno
            pretokenised=False,
            fields=False,
            threads=1, 
        )

        index_ref = indexer.index(corpus_dataframe.to_dict(orient="records"))
        
        # Open the index to ensure it is valid
        index = pt.IndexFactory.of(index_ref)

        # Print a simple summary
        print("Index location:", keywords_expanded_index_path)
        print("Indexed documents:", index.getCollectionStatistics().getNumberOfDocuments())
    
    def create_two_fields_index(self):
        if not hasattr(self.collection, "corpus_dataframe"):
            raise RuntimeError("Documents not loaded. Call load_documents() before creating an index.")
        longest_len = self.collection.corpus_dataframe["docno"].str.len().max()

        # Create index or raise error if it exists
        two_fields_index_path = self.indexes_folder / TWO_FIELDS_INDEX_NAME
        if two_fields_index_path.exists():
            raise RuntimeError(f"Index already exists: {two_fields_index_path}")
        two_fields_index_path.mkdir(parents=True)

        corpus_dataframe = self.collection.corpus_dataframe.copy() # Make a copy to avoid modifying the original
        # Create keywords field
        tqdm.pandas(desc="Creating keywords field")
        corpus_dataframe["keywords"] = corpus_dataframe["text"].progress_apply(
            lambda q: " ".join(
                thesaurus_based_expansion(q, keywords_extractor(q))
            )
        )

        # Build the indexer using the pt.IterDictIndexer
        indexer = pt.IterDictIndexer(
        str(two_fields_index_path),
        meta={"docno": longest_len},
        text_attrs=["text", "keywords"],    # which field(s) contain the text
        meta_reverse=["docno"],             # enable reverse lookup on docno
        pretokenised=False,
        threads=1, 
        fields=True,
        properties = {'index.document.class': 'FSADocumentIndexInMemFields'} # doesn't work
        )

        index_ref = indexer.index(corpus_dataframe.to_dict(orient="records"))

        # Open the index to ensure it is valid
        index = pt.IndexFactory.of(index_ref)

        # Print a simple summary
        print("Index location:", two_fields_index_path)
        print("Indexed documents:", index.getCollectionStatistics().getNumberOfDocuments())

    def create_dense_index(self):
        if not hasattr(self.collection, "corpus_dataframe"):
            raise RuntimeError("Documents not loaded. Call load_documents() before creating an index.")
        
        # Create index or raise error if it exists
        dense_index_path = self.indexes_folder / DENSE_INDEX_NAME
        if dense_index_path.exists():
            raise RuntimeError(f"Index already exists: {dense_index_path}")
        
        # build an indexing pipeline that first applies RetroMAE to get dense vectors, then indexes them into the FlexIndex
        dense_index = FlexIndex(str(dense_index_path), verbose = 1)
        model = RetroMAE.msmarco_distill()
        offline_indexing_pipeline = model >> dense_index.indexer(mode="overwrite")
        
        corpus_dataframe = self.collection.corpus_dataframe.copy()
        corpus_dataframe = corpus_dataframe.to_dict(orient="records")
        # create the index
        offline_indexing_pipeline.index(corpus_dataframe)

        # Print a simple summary
        print("Index location:", dense_index_path)

    def load_basic_index(self):
        index_path = self.indexes_folder / BASIC_INDEX_NAME
        if not index_path.exists():
            raise RuntimeError(f"Index does not exist: {index_path}")
        self.basic_index = pt.IndexFactory.of(str(index_path))
    
    def load_keywords_expanded_index(self):
        index_path = self.indexes_folder / KEYWORDS_INDEX_NAME
        if not index_path.exists():
            raise RuntimeError(f"Index does not exist: {index_path}")
        self.keywords_expanded_index = pt.IndexFactory.of(str(index_path))

    def load_two_fields_index(self):
        index_path = self.indexes_folder / TWO_FIELDS_INDEX_NAME
        if not index_path.exists():
            raise RuntimeError(f"Index does not exist: {index_path}")
        self.two_fields_index = pt.IndexFactory.of(str(index_path))
    
    def load_dense_index(self):
        index_path = self.indexes_folder / DENSE_INDEX_NAME
        if not index_path.exists():
            raise RuntimeError(f"Index does not exist: {index_path}")
        self.dense_index = FlexIndex(str(index_path))