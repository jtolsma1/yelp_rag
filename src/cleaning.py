import unicodedata
import pandas as pd

from src import config

class CleanChunkYelpReviews:


    def __init__(self,
                 sampled_data_path = None,
                 processed_data_path = None,
                 col_text = None,
                 min_review_chars = None,
                 overlap_chars = None,
                 chunk_chars = None,
                 min_chunk_chars = None,
                 col_restaurant_id = None,
                 col_review_id = None,
                 col_restaurant_name = None,
                 col_stars = None,
                 col_date = None,
                 parquet_engine = None
                 ):
        """
        Defaults to all parameters as set in config.py; overrides parameters when stated in function call.
        @param sampled_data_path: file path for retrieving data sampled from the Yelp dataset
        @param processed_data_path: file path for storing model-processed datasets
        @param col_text: name of column containing review text chunks
        @param min_review_chars: RAG removes any review chunks with fewer chars than this value
        @param overlap_chars: number of chars that overlap between chunks from the same review (i.e. chars that two adjacent chunks will have in common)
        @param chunk_chars: maximum number of chars per chunk of review text
        @param col_restaurant_id: name of column in Yelp dataset containing the restaurant id
        @param col_review_id: name of column in Yelp dataset that stores unique review id
        @param col_restaurant_name: name of column in Yelp dataset that stores the restaurant name
        @param col_stars: name of column in Yelp dataset that stores the star rating for each review
        @param col_date: name of column in Yelp dataset that stores the date that the review was submitted
        @param parquet_engine: engine used for encoding parquet files
        """

        # constants imported from config.py
        # defaults from config.py
        defaults = {
            "sampled_data_path":config.DATA_DIR_SAMP,
            "processed_data_path":config.DATA_DIR_PROC,
            "col_text": config.COL_TEXT,
            "min_review_chars": config.MIN_REVIEW_CHARS,
            "overlap_chars": config.OVERLAP_CHARS,
            "chunk_chars": config.CHUNK_CHARS,
            "min_chunk_chars": config.MIN_CHUNK_CHARS,
            "col_restaurant_id": config.COL_RESTAURANT_ID,
            "col_review_id": config.COL_REVIEW_ID,
            "col_restaurant_name": config.COL_RESTAURANT_NAME,
            "col_stars": config.COL_STARS,
            "col_date": config.COL_DATE,
            "parquet_engine": config.PARQUET_ENGINE
        }

        # overrides supplied by caller (example pattern)
        overrides = {
            "sampled_data_path":sampled_data_path,
            "processed_data_path":processed_data_path,
            "col_text": col_text,
            "min_review_chars": min_review_chars,
            "overlap_chars": overlap_chars,
            "chunk_chars": chunk_chars,
            "min_chunk_chars": min_chunk_chars,
            "col_restaurant_id": col_restaurant_id,
            "col_review_id": col_review_id,
            "col_restaurant_name": col_restaurant_name,
            "col_stars": col_stars,
            "col_date": col_date,
            "parquet_engine": parquet_engine
        }

        for name,default in defaults.items():
            value = overrides[name] if overrides[name] is not None else default
            setattr(self,name,value)


    def load_rag_reviews_data(self):
        """
        Load the data saved from the import step
        @return: imported dataframe 
        """
        return pd.read_csv(self.sampled_data_path)


    def clean_review_text(self,df):
        """
        Removes nonstandard characters and drop unusable (too short) values
        @param df: dataframe containing reviews data to be cleaned
        @return: dataframe with nonstandard and non-useful text removed.
        """

        col_text = self.col_text
        before = len(df)

        # basic cleaning
        df[col_text] = df[col_text].astype("string").str.strip()
        df = df.dropna(subset = [col_text])

        # remove ultra-short reviews
        df = df[df[col_text].str.len() >= self.min_review_chars]

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
            df[col_text] = df[col_text].str.replace(char,rep_str,regex = False)

        df[col_text] = df[col_text].str.replace(r"\s+", " ", regex=True).str.strip()

        print(f"  {before - len(df)} reviews dropped in cleaning step.")
        return df

    @staticmethod
    def normalize_unicode(text):
        """
        Convert all characters in the review text to standard unicode.
        @return: reviews data normalized to include only unicode text.
        """
        return unicodedata.normalize("NFKC",text)

    def deduplicate_reviews(self,df):
        """
        Remove duplicate reviews from the dataset.
        @returns: deduplicated review dataframe
        """
        before = len(df)
        df = df.drop_duplicates(subset = [self.col_text])
        print(f"  {before - len(df)} reviews dropped in deduplicating step.")
        return df


    def clean_normalize_deduplicate(self,df):
        """
        Apply cleaning, unicode normalization, and deduplication steps:
        @param df: dataframe containing reviews data to be cleaned
        @returns: cleaned, normalized, deduplicated dataframe of reviews
        """
        df = self.clean_review_text(df)
        df[self.col_text] = df[self.col_text].apply(self.normalize_unicode)
        df = self.deduplicate_reviews(df)
        return df


    def divide_reviews_into_chunks(self,text):
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
            upcoming_chunk_length = len(text) - char_idx + self.overlap_chars
            if len(text) <= self.chunk_chars:
                chunk_dict.update(
                    {0:text}
                    )
            elif char_idx == 0:
                chunk_dict.update(
                    {chunk_idx:text[0:self.chunk_chars]}
                    )
            elif upcoming_chunk_length > self.min_chunk_chars:
                chunk_dict.update(
                    {chunk_idx:text[(char_idx - self.overlap_chars):(char_idx+self.chunk_chars-self.overlap_chars)]}
                    )
            chunk_idx +=1
            char_idx+=(self.chunk_chars - self.overlap_chars)

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
            new_row["business_id"] = getattr(row,self.col_restaurant_id)
            new_row["review_id"] = getattr(row,self.col_review_id)
            new_row["restaurant_name"] = getattr(row,self.col_restaurant_name)
            new_row["chunk_id"] = f"{getattr(row,self.col_review_id)}_{new_row["chunk_index"].values[0]}"
            new_row["n_chars"] = new_row["chunk"].str.len()
            new_row["stars"] = getattr(row,self.col_stars)
            new_row["date"] = getattr(row,self.col_date)
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

        print("  Executing clean/unicode normalization/deduplication step.")
        cleaned_reviews = self.clean_normalize_deduplicate(reviews_df)

        print(f"  Splitting reviews into chunks of maximum {self.chunk_chars} characters")
        cleaned_chunked = self.generate_chunk_df(cleaned_reviews)

        print(f"  Storing chunked data as parquet at {self.processed_data_path}")
        cleaned_chunked.to_parquet(self.processed_data_path,engine = self.parquet_engine)

        return True
