import pandas as pd
import os
import faiss
import src.config as config
from sentence_transformers import SentenceTransformer
from pathlib import Path

class RetrieveRelevantText:

    def __init__(self,processed_data_path,index_path):
        """
        @param processed_data_path: import file path for retrieving cleaned and chunked reviews data
        @param index_path: export path for storing reviews as embeddings, along with metadata
        """
        self.processed_data_path = processed_data_path
        self.index_path = index_path


    def get_index_file_list(self):
        """
        Obtain a list of all filenames in the index directory for the project, which will be queried for the RAG output.
        @return: list of all filenames
        """
        return sorted(self.index_path.glob("*.faiss"))
    
    def get_index_files_with_metadata(self):
        """
        From a given list of index filenames, return the restaurant id, index file path, and metadata file path associated with each filename
        return: list of dictionaries (one per index filename) containing the content described above
        """
        faiss_files = self.get_index_file_list()
        index_pairs = []

        for faiss_path in faiss_files:
            business_id = faiss_path.stem
            meta_path = self.index_path / f"{business_id}_meta.parquet"

            if not meta_path.exists():
                raise KeyError(f"Missing metadata for restaurant id {business_id}")
            
            index_pairs.append({
                "business_id":business_id,
                "index_file":faiss_path,
                "metadata_file":meta_path
            })
        
        return index_pairs
    
    @staticmethod
    def load_embedding_model(model_name,device = "mps"):
        """
        Load a local embedding model using sentence_transformer library. Used here to encode the query text for the RAG.
        @return: transformer model from HuggingFace.
        """
        return SentenceTransformer(model_name_or_path = model_name, device = device)
    
    
    def encode_query_topics(self):
        """
        Create embeddings for all topic words relevant to the RAG.s
        @return: a dictionary of embeddings with one entry per topic word
        """
        model = self.load_embedding_model(config.EMBEDDING_MODEL_NAME,config.EMBED_DEVICE)
        query_embed = {}
        for topic,keywords in config.TOPICS.items():
            query = model.encode([keywords],convert_to_numpy=True,normalize_embeddings=True)
            query_embed.update({topic:query})
        
        return query_embed


    @staticmethod
    def convert_similarity_arrays_to_df(score_array,index_array,metadata,metadata_cols):
        """
        Converts raw similarity arrays from FAISS to a dataframe of relevant text chunks.
        Limits returned values to a preconfigured number of chunks per review to avoid duplicated statements.
        @param score_array: similarity scores returned by searching query verbatim in the corpus of reviews
        @param index_array: indicies associated with each chunk returned by the FAISS query search
        @param metadata: metadata dataframe associated with each index, such as business id and (non-encoded) text chunk
        @param metadata_cols: columns in the metadata dataframe relevant to the final result
        @return: dataframe of most similar text chunks and identifiers for each chunk
        """

        chunk_id_col = metadata_cols["chunk_id_col"]
        business_id_col = metadata_cols["business_id_col"]
        restaurant_name_col = metadata_cols["restaurant_name_col"]
        stars_col = metadata_cols["stars_col"]
        review_id_col = metadata_cols["review_id_col"]
        chunk_col = metadata_cols["chunk_col"]

        df = pd.DataFrame({
        "scores":score_array[0],
        "indicies":index_array[0]
        }).assign(
            chunk_id = lambda df: metadata.iloc[df["indicies"]][chunk_id_col].values,
            business_id = lambda df: metadata.iloc[df["indicies"]][business_id_col].values,
            restaurant_name = lambda df: metadata.iloc[df["indicies"]][restaurant_name_col].values,
            stars = lambda df: metadata.iloc[df["indicies"]][stars_col].values,
            review_id = lambda df: metadata.iloc[df["indicies"]][review_id_col].values,
            chunk = lambda df: metadata.iloc[df["indicies"]][chunk_col].values
        )

        df = df.sort_values(by = "scores",ascending=False)
        df = df.groupby("review_id",as_index=False).head(config.MAX_CHUNKS_PER_REVIEW)
        return df


    def retrieve_topic_relevant_text(self):
        """
        Use FAISS vector search to return the most relevant review text chunks for each restaurant and each topic in the RAG.
        @return: True if executed successfully
        """

        index_pairs = self.get_index_files_with_metadata()
        query_embed = self.encode_query_topics()
        result_df = pd.DataFrame()
        empties = {}

        for file in index_pairs:
            index_file_path = str(file["index_file"])
            meta_file_path = file["metadata_file"]
            index = faiss.read_index(index_file_path)
            meta = pd.read_parquet(meta_file_path)
            for topic in config.TOPICS.keys():
                D,I = index.search(query_embed[topic],k = config.TOP_K_PER_TOPIC)
                df = self.convert_similarity_arrays_to_df(D,I,meta,config.METADATA_COLS)
                if df.empty:
                    empties.update({file["business_id"]:topic})
                df.insert(0,"topic",topic)
                df = df.iloc[:config.MAX_CHUNKS_PER_TOPIC]
                result_df = pd.concat([result_df,df],axis = 0,ignore_index=True)


        print(f"Relevant text dataframe created using FAISS vector search with shape {result_df.shape}.")
        if empties:
            print(f"The following business ids and topics returned empty results:")
            for k,v in empties:
                print(f"business id = {k}, topic = {v}")
        else:
            print("No empty results found.")
        result_df.to_parquet(os.path.join(self.processed_data_path,"topic_relevant_review_chunks.parquet"),engine = "pyarrow")
        print(f"Relevant text dataframe uploaded to {os.path.join(self.processed_data_path,"topic_relevant_review_chunks.parquet")}")
        return True