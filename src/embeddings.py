import os
from typing import List
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss

from src import config

class CreateReviewEmbeddings:

    def __init__(self,
                 processed_data_path = None,
                 index_path = None,
                 embedding_model_name = None,
                 embed_device = None,
                 embed_batch_size = None,
                 col_restaurant_id = None,
                 index_metric = None,
                 normalize_embeddings = None,
                 parquet_engine = None
                 ):
        """
        @param processed_data_path: file path for storing and retrieving model-processed datasets
        @param index_path: export path for storing reviews as embeddings, along with metadata
        @param embedding_model_name: model name retrieved from HuggingFace and used to create embeddings of review text chunks
        @param embed_device: specifies CPU or GPU as the device that runs the embedding model
        @param embed_batch_size: controls the number of chunks that are embedded at once
        @param col_restaurant_id: column that stores the restaurant id in the Yelp dataset
        @param index_metric: metric (cosine or l2) used to generate vector search results from indexes
        @param parquet_engine: engine used for encoding parquet files
        """
        # defaults from config.py
        defaults = {
            "processed_data_path": config.DATA_DIR_PROC,
            "index_path": config.INDEX_DIR,
            "embedding_model_name": config.EMBEDDING_MODEL_NAME,
            "embed_device": config.EMBED_DEVICE,
            "embed_batch_size": config.EMBED_BATCH_SIZE,
            "col_restaurant_id": config.COL_RESTAURANT_ID,
            "index_metric": config.INDEX_METRIC,
            "normalize_embeddings":config.NORMALIZE_EMBEDDINGS,
            "parquet_engine":config.PARQUET_ENGINE
        }

        # overrides supplied by caller (example)
        overrides = {
            "processed_data_path": processed_data_path,
            "index_path": index_path,
            "embedding_model_name": embedding_model_name,
            "embed_device": embed_device,
            "embed_batch_size": embed_batch_size,
            "col_restaurant_id": col_restaurant_id,
            "index_metric": index_metric,
            "normalize_embeddings": normalize_embeddings,
            "parquet_engine":parquet_engine
        }

        for name,default in defaults.items():
            value = overrides[name] if overrides[name] is not None else default
            setattr(self,name,value)


    def load_chunked_reviews_data(self):
        """
        Import the cleaned and chunked reviews.
        @return: reviews data passed through cleaning and chunking steps.
        """
        return pd.read_parquet(self.processed_data_path)

    @staticmethod
    def load_embedding_model(model_name,device = "mps"):
        """
        Load a local embedding model using sentence_transformer library.
        @return: transformer model from HuggingFace.
        """
        return SentenceTransformer(model_name_or_path = model_name, device = device)


    @staticmethod
    def embed_texts(model:SentenceTransformer,
                    texts:List[str],
                    batch_size:int = 64,
                    normalize_flag:bool = True
                    ):
        """
        Create embeddings using the imported model.
        @param model: sentence transformer model imported to create embeddings
        @param texts: texts to be embedded
        @param batch_size: batch size used for the computation; standard parameter in the .encode function in SentenceTransformer class
        @param normalize_embeddings: whether to return vectors that have a length of 1; standard parameter in the .encode function in SentenceTransformer class
        @param show_progress_bar: shows embedding progress; standard parameter in the .encode function in SentenceTransformer class
        @param convert_to_numpy: whether to return vectors as a numpy array; standard parameter in the .encode function in SentenceTransformer class
        @return: embeddings of review text chunks created using the selected transformer model. 
        """

        embeddings = model.encode(
            sentences = texts,
            batch_size = batch_size,
            normalize_embeddings = normalize_flag,
            show_progress_bar = True,
            convert_to_numpy = True,
        )

        return embeddings.astype(np.float32,copy = False)


    @staticmethod
    def build_faiss_index(X,metric = "cosine"):
        """
        Creates FAISS indicies from review chunk embeddings.
        @param X: embeddings created using the 'embed_texts' function
        @param metric: search metric; must be cosine or l2
        @return: FAISS index data
        """

        assert X.dtype == np.float32
        n,d = X.shape

        if metric == "cosine":
            index = faiss.IndexFlatIP(d)
        elif metric == "l2":
            index = faiss.IndexFlatL2(d)
        else:
            raise ValueError("metric must be 'cosine' or 'l2'")

        index.add(X)

        return index


    def create_faiss_for_yelp_reviews(self):
        """
        Converts cleaned and chunked reviews data (text) into embeddings, then converts those embeddings into FAISS index files.
        Also creates metadatafiles to pair with each FAISS index to source restaurant name, review stars, etc. for retrieved indicies.
        Generates one index file per restaurant.
        @return: True if executed successfully
        """

        chunks_df = self.load_chunked_reviews_data()
        texts = chunks_df["chunk"].tolist()

        model = self.load_embedding_model(self.embedding_model_name,self.embed_device)
        print(f"  Loaded model '{self.embedding_model_name}' from HuggingFace.")
        print("  Starting embedding process:")
        embeddings = self.embed_texts(model,texts,batch_size = self.embed_batch_size,normalize_flag = self.normalize_embeddings)
        print("  Completed embedding process.")

        for restaurant_id, idx in chunks_df.groupby(self.col_restaurant_id).groups.items():
            idx_list = list(idx)
            X_r = embeddings[idx_list]
            meta_r = chunks_df.loc[idx_list].copy()

            index = self.build_faiss_index(X_r,metric = self.index_metric)
            faiss.write_index(index,os.path.join(self.index_path,f"{restaurant_id}.faiss"))

            meta_r.reset_index(drop = True).to_parquet(os.path.join(self.index_path,f"{restaurant_id}_meta.parquet"),engine = self.parquet_engine,index = False)

        faiss_created = list(self.index_path.glob("*.faiss"))
        print(f"  {len(faiss_created)} .faiss files created in the index directory.")

        meta_created = list(self.index_path.glob("*.parquet"))
        print(f"  {len(meta_created)} metadata files (.parquet) created in the index directory.")

        return True
