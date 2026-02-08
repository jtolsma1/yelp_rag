import os
from src.data_io import ImportYelpReviewText
from src.cleaning import CleanChunkYelpReviews
from src.embeddings import CreateReviewEmbeddings
from src.retrieval import RetrieveRelevantText
from src.summarization import SummarizeRelevantReviewText
from src import config

class YelpRAGPipelineRunner:

    def __init__(self):
        print("Start Yelp RAG Pipeline")

    def run_pipeline(self):

        print("\nData download and sampling pipeline started.\n")
        io = ImportYelpReviewText()
        io.generate_final_restaurant_list_for_rag()
        print("\nData download and sampling pipeline complete.")

        print("\nCleaning and chunking pipeline started.\n")
        cl = CleanChunkYelpReviews(
            sampled_data_path=os.path.join(config.DATA_DIR_SAMP,"reviews_df.csv"),
            processed_data_path=os.path.join(config.DATA_DIR_PROC,"review_chunks.parquet")
            )
        cl.clean_chunk_export()
        print("\nCleaning and chunking pipeline complete.")

        print("\nEmbedding pipeline started.\n")
        emb = CreateReviewEmbeddings(processed_data_path=os.path.join(config.DATA_DIR_PROC,"review_chunks.parquet"))
        emb.create_faiss_for_yelp_reviews()
        print("\nEmbedding pipeline complete.")

        print("\nRelevant text retrieval pipeline started.\n")
        ret = RetrieveRelevantText()
        ret.retrieve_topic_relevant_text()
        print("\nRelevant text retrieval pipeline complete.")

        print("\nLLM summarization pipeline started.\n")
        su = SummarizeRelevantReviewText()
        su.summarize_relevant_review_text()
        print("\nLLM summarization pipeline complete.")

        print("\nYelp RAG pipeline execution complete.")
