import sys
import os
import pandas as pd
import numpy as np
import json

import src.config as config
class ImportYelpReviewText:

    def __init__(self,
                 raw_data_path = None,
                 sampled_data_path = None,
                 n_import_rows = None,
                 col_business_category = None,
                 col_restaurant_id = None,
                 col_review_id = None,
                 min_reviews = None,
                 n_restaurants = None
                 ):
        """
        Defaults to all parameters as set in config.py; overrides parameters when stated in function call.
        @param raw_data_path: file path for sourcing the review content and business identifiers from the entire Yelp dataset
        @param sampled_data_path_reviews: file path for storing data sampled from the Yelp dataset
        @param n_import_rows: number of rows to sample from the entire Yelp dataset (>= 100k recommended for RAG)
        @param col_business_category: name of column in Yelp dataset that stores business category ("restaurant", "hotel")
        @param col_review_id: name of column in Yelp dataset that stores unique review id
        @param min_reviews: minimum number of reviews needed for inclusion in RAG
        @param n_restaurants: number of restaurants to use for the RAG results
        """

        defaults = {
            "raw_data_path":config.DATA_DIR_RAW,
            "sampled_data_path":config.DATA_DIR_SAMP,
            "n_import_rows": config.N_IMPORT_ROWS,
            "col_business_category": config.COL_BUSINESS_CATEGORY,
            "col_restaurant_id": config.COL_RESTAURANT_ID,
            "col_review_id": config.COL_REVIEW_ID,
            "min_reviews": config.MIN_REVIEWS,
            "n_restaurants": config.N_RESTAURANTS,
        }

        overrides = {
            "raw_data_path": raw_data_path,
            "sampled_data_path": sampled_data_path,
            "n_import_rows": n_import_rows,
            "col_business_category": col_business_category,
            "col_restaurant_id": col_restaurant_id,
            "col_review_id": col_review_id,
            "min_reviews": min_reviews,
            "n_restaurants": n_restaurants,
        }

        for name, default in defaults.items():
            value = overrides[name] if overrides[name] is not None else default
            setattr(self, name, value)


    def import_sample_from_complete_dataset(self,import_path):
        """
        Extract a preset slice of data from the Yelp dataset for use in the RAG.
        The scale of the slice is set in config.py.
        @param import_path: file path for the entire Yelp dataset
        @return: sample consisting of a number of rows specified in a config file
        """

        sample =[]
        with open(import_path,"r") as f:
            for i,line in enumerate(f):
                if i >= self.n_import_rows:
                    break
                sample.append(json.loads(line))
        return sample


    def import_yelp_data_sample(self):
        """
        Execute data extraction from the Yelp review content and business content (stored separately).
        Merge the two extractions together into a single dataset.
        @return: combined Yelp dataset of review content and business identifiers
        """
        reviews = self.import_sample_from_complete_dataset(os.path.join(self.raw_data_path,"yelp_academic_dataset_review.json"))
        businesses = self.import_sample_from_complete_dataset(os.path.join(self.raw_data_path,"yelp_academic_dataset_business.json"))
        
        reviews_sample = pd.merge(
            pd.DataFrame(reviews),
            pd.DataFrame(businesses),
            how = "inner",
            on = "business_id",
            suffixes = ["_reviews","_restaurant"]
        ).dropna()

        reviews_sample.to_csv(os.path.join(self.sampled_data_path,"reviews_samples.csv"))
        return reviews_sample
    
    def select_restaurants_for_rag(self,input_df):
        """
        Filter the extracted Yelp data to restaurants only with a minimum number of distinct reviews.
        Then reduce that list to a preset number of restaurants; that preset number is set in config.py.
        @param input_df: extracted Yelp dataset
        @return: Yelp data samples filtered for (1) restaurants only and (2) having a minimum number of reviews
        """

        cond1 = input_df[self.col_business_category].str.lower().str.contains("restaurant")
       
       # many hotels contain restaurants but are not the focus of this RAG
        cond2 = ~input_df[self.col_business_category].str.lower().str.contains("hotel|cinema",regex = True) 

        input_df = input_df[cond1 & cond2]

        ids_array = input_df.groupby(self.col_restaurant_id)[self.col_review_id].nunique()

        selected_ids = ids_array[ids_array > self.min_reviews].sample(self.n_restaurants,random_state = 5).index.tolist()

        output_df = input_df[input_df[self.col_restaurant_id].isin(selected_ids)]

        output_df.to_csv(os.path.join(self.sampled_data_path,"reviews_df.csv"))
        return output_df
    

    def generate_final_restaurant_list_for_rag(self):
        """
        Execute the Yelp dataset extraction process end-to-end, starting with the entire Yelp dataset and ending with 
        a small quantity of restaurants with a minimum number of reviews to use in the RAG.
        @return: True if executed successfully
        """
        print("  Sampling Yelp datasets...")
        
        reviews_sample = self.import_yelp_data_sample()
        
        print(f"  Yelp dataset sample with shape {reviews_sample.shape} created.")
        print(f"  Yelp dataset sample stored at {os.path.join(self.sampled_data_path,"reviews_samples.csv")}")
        print(f"  Sampling Yelp dataset for {self.n_restaurants} restaurants to use in this RAG demo.")
        
        reviews_df = self.select_restaurants_for_rag(reviews_sample)
        
        print(f"  RAG dataset consisting of {self.n_restaurants} and with shape {reviews_df.shape} created.")
        print(f"  RAG dataset sample stored at {os.path.join(self.sampled_data_path,"reviews_df.csv")}")
        return True