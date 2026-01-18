import pandas as pd
import os
import unicodedata

import src.config as config

class CleanChunkYelpReviews:


    def __init__(self,sampled_data_path,processed_data_path):
        """
        @param sampled_data_path: import file path supplying data for restaurants selected for the RAG
        @param processed_data_path: export file path for storing cleaned and chunked reviews data
        """
        self.sampled_data_path = sampled_data_path
        self.processed_data_path = processed_data_path


    def load_rag_reviews_data(self):
        """
        Load the data saved from the import step
        @return: imported dataframe 
        """
        return pd.read_csv(self.sampled_data_path)

    @staticmethod
    def clean_review_text(df):
        """
        Removes nonstandard characters and drop unusable (too short) values
        @param df: dataframe containing reviews data to be cleaned
        @return: dataframe with nonstandard and non-useful text removed.
        """

        COL_TEXT = config.COL_TEXT
        before = len(df)
        # basic cleaning    
        df[COL_TEXT] = df[COL_TEXT].astype("string").str.strip()
        df = df.dropna(subset = [COL_TEXT])

        # remove ultra-short reviews
        df = df[df[COL_TEXT].str.len() >= config.MIN_REVIEW_CHARS]

        # remove nonstandard characters
        invalid_chars = {
            "\u00a0":" ",
            "\u002b":"",
            "\xa0":" ",
            "\x0b":" ",
            "“":'"',
            "’": "'",
        }
        
        for char,rep_str in invalid_chars.items():
            df[COL_TEXT] = df[COL_TEXT].str.replace(char,rep_str,regex = False)

        df[COL_TEXT] = df[COL_TEXT].str.replace(r"\s+", " ", regex=True).str.strip()
        
        print(f"{before - len(df)} reviews dropped in cleaning step.")
        return df

    @staticmethod
    def normalize_unicode(text):
        """
        Convert all characters in the review text to standard unicode.
        @return: reviews data normalized to include only unicode text.
        """
        return unicodedata.normalize("NFKC",text)

    @staticmethod
    def deduplicate_reviews(df):
        """
        Remove duplicate reviews from the dataset.
        @returns: deduplicated review dataframe
        """
        before = len(df)
        df = df.drop_duplicates(subset = [config.COL_TEXT])
        print(f"{before - len(df)} reviews dropped in deduplicating step.")
        return df


    def clean_normalize_deduplicate(self,df):
        """
        Apply cleaning, unicode normalization, and deduplication steps:
        @param df: dataframe containing reviews data to be cleaned
        @returns: cleaned, normalized, deduplicated dataframe of reviews
        """
        df = self.clean_review_text(df)
        df[config.COL_TEXT] = df[config.COL_TEXT].apply(self.normalize_unicode)
        df = self.deduplicate_reviews(df)
        return df


    @staticmethod
    def divide_reviews_into_chunks(text):
        """
        For a given piece of text, divide into chunks with a maximum chunk size set in config.py.
        Chunking avoids overwhelming the LLM with text strings that are too long.
        @param text: text string to be divided into chunks.
        @return: review texts divided into chunks of size specified in config file
        """
        chunk_dict = {}
        char_idx = 0
        chunk_idx = 0
        while char_idx < len(text):
            upcoming_chunk_length = len(text) - char_idx + config.OVERLAP_CHARS
            if len(text) <= config.CHUNK_CHARS:
                chunk_dict.update(
                    {0:text}
                    )
            elif char_idx == 0:
                chunk_dict.update(
                    {chunk_idx:text[0:config.CHUNK_CHARS]}
                    )
            elif upcoming_chunk_length > config.MIN_CHUNK_CHARS:
                chunk_dict.update(
                    {chunk_idx:text[(char_idx - config.OVERLAP_CHARS):(char_idx+config.CHUNK_CHARS-config.OVERLAP_CHARS)]}
                    )
            chunk_idx +=1
            char_idx+=(config.CHUNK_CHARS - config.OVERLAP_CHARS)

            return pd.DataFrame(chunk_dict,index = ["chunk"]).T.reset_index(names = ["chunk_index"])
        
    def generate_chunk_df(self,df):
        """
        Transform the reviews dataframe (with one review per row of any length) into chunks.
        Associate the important review metadata with each chunk.
        @param df: dataframe to be transformed into chunks.
        @return: chunked review data plus relevant metadata
        """
        chunk_df = pd.DataFrame()
        for row in df.itertuples():
            new_row = self.divide_reviews_into_chunks(row.text)
            new_row["business_id"] = getattr(row,config.COL_RESTAURANT_ID)
            new_row["review_id"] = getattr(row,config.COL_REVIEW_ID)
            new_row["restaurant_name"] = getattr(row,config.COL_RESTAURANT_NAME)
            new_row["chunk_id"] = f"{getattr(row,config.COL_REVIEW_ID)}_{new_row["chunk_index"].values[0]}"
            new_row["n_chars"] = new_row["chunk"].str.len()
            new_row["stars"] = getattr(row,config.COL_STARS)
            new_row["date"] = getattr(row,config.COL_DATE)
            chunk_df = pd.concat([chunk_df,new_row],axis = 0)
            chunk_df = chunk_df.reset_index(drop = True)
        return chunk_df
    

    def clean_chunk_export(self):
        """
        Execute all cleaning and chunking functions. 
        Store the chunked data as a parquet file in the 'processed' data directory.
        @return: True if executed successfully
        """
        reviews_df = self.load_rag_reviews_data()

        print("Executing clean/unicode normalization/deduplication step.")
        cleaned_reviews = self.clean_normalize_deduplicate(reviews_df)

        print(f"Splitting reviews into chunks of maximum {config.CHUNK_CHARS} characters")
        cleaned_chunked = self.generate_chunk_df(cleaned_reviews)

        print(f"Storing chunked data as parquet at {self.processed_data_path}")
        cleaned_chunked.to_parquet(self.processed_data_path,engine = "pyarrow")

        return True