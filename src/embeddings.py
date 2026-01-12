import os
from sentence_transformers import SentenceTransformer
from typing import List
import pandas as pd
import numpy as np
import faiss

import src.config as config

class CreateReviewEmbeddings:

    def __init__(self,processed_data_path,index_path):
        """
        @param processed_data_path: import file path for retrieving cleaned and chunked reviews data
        @param index_path: export path for storing reviews as embeddings, along with metadata
        """
        self.processed_data_path = processed_data_path
        self.index_path = index_path

    
    def load_chunked_reviews_data(self):
        """
        Import the cleaned and chunked reviews.
        @returns:
        """
        return pd.read_parquet(self.processed_data_path)
    
    @staticmethod
    def load_embedding_model(model_name,device = "mps"):
        """
        Load a local embedding model using sentence_transformer library.
        @returns: sen
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
        @returns: 
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

        chunks_df = self.load_chunked_reviews_data()
        texts = chunks_df["chunk"].tolist()
        
        model = self.load_embedding_model(config.EMBEDDING_MODEL_NAME,"mps")
        print(f"Loaded model '{config.EMBEDDING_MODEL_NAME}' from HuggingFace.")
        print("Starting embedding process:")
        embeddings = self.embed_texts(model,texts,batch_size = config.EMBED_BATCH_SIZE,normalize_flag = True)
        print("Completed embedding process.")

        for restaurant_id, idx in chunks_df.groupby(config.COL_RESTAURANT_ID).groups.items():
            idx_list = list(idx)
            X_r = embeddings[idx_list]
            meta_r = chunks_df.loc[idx_list].copy()

            index = self.build_faiss_index(X_r,metric = config.INDEX_METRIC)
            faiss.write_index(index,os.path.join(self.index_path,f"{restaurant_id}.faiss"))

            meta_r.reset_index(drop = True).to_parquet(os.path.join(self.index_path,f"{restaurant_id}_meta.parquet"),engine = "pyarrow",index = False)

        faiss_created = list(config.INDEX_DIR.glob("*.faiss"))  
        print(f"{len(faiss_created)} .faiss files created in the index directory.")  

        meta_created = list(config.INDEX_DIR.glob("*.parquet"))  
        print(f"{len(meta_created)} metadata files (.parquet) created in the index directory.")  