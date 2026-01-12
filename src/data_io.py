import sys
import os
import pandas as pd
import numpy as np
import json

import src.config as config
class ImportYelpReviewText:

    def __init__(self,
                 raw_data_path_reviews,
                 raw_data_path_business,
                 sampled_data_path_reviews,
                 sampled_data_path_rag):
        """
        @param raw_data_path_reviews: file path for sourcing the review content from the entire Yelp dataset
        @param raw_data_path_business: file path for sourcing the business identifiers from the entire Yelp dataset 
        @param sampled_data_path_reviews: file path for storing data sampled from the Yelp dataset
        @param sampled_data_path_rag: fole path for storing data for restaurants selected for the RAG
        """
        self.raw_data_path_reviews = raw_data_path_reviews
        self.raw_data_path_business = raw_data_path_business
        self.sampled_data_path_reviews = sampled_data_path_reviews
        self.sampled_data_path_rag = sampled_data_path_rag


    @staticmethod
    def import_sample_from_complete_dataset(import_path):
        """
        Extract a preset slice of data from the Yelp dataset for use in the RAG.
        The scale of the slice is set in config.py.
        @param import_path: file path for the entire Yelp dataset
        @returns: sample consisting of a number of rows specified in a config file
        """

        sample =[]
        with open(import_path,"r") as f:
            for i,line in enumerate(f):
                if i >= config.N_IMPORT_ROWS:
                    break
                sample.append(json.loads(line))
        return sample


    def import_yelp_data_sample(self):
        """
        Execute data extraction from the Yelp review content and business content (stored separately).
        Merge the two extractions together into a single dataset.
        @returns: combined Yelp dataset of review content and business identifiers
        """
        reviews = self.import_sample_from_complete_dataset(self.raw_data_path_reviews)
        businesses = self.import_sample_from_complete_dataset(self.raw_data_path_business)
        
        reviews_sample = pd.merge(
            pd.DataFrame(reviews),
            pd.DataFrame(businesses),
            how = "inner",
            on = "business_id",
            suffixes = ["_reviews","_restaurant"]
        ).dropna()

        reviews_sample.to_csv(self.sampled_data_path_reviews)
        return reviews_sample
    
    def select_restaurants_for_rag(self,input_df):
        """
        Filter the extracted Yelp data to restaurants only with a minimum number of distinct reviews.
        Then reduce that list to a preset number of restaurants; that preset number is set in config.py.
        @param input_df: extracted Yelp dataset
        @returns: Yelp data samples filtered for (1) restaurants only and (2) having a minimum number of reviews
        """

        cond1 = input_df[config.COL_BUSINESS_CATEGORY].str.lower().str.contains("restaurant")
       
       # many hotels contain restaurants but are not the focus of this RAG
        cond2 = ~input_df[config.COL_BUSINESS_CATEGORY].str.lower().str.contains("hotel|cinema",regex = True) 

        input_df = input_df[cond1 & cond2]

        ids_array = input_df.groupby(config.COL_RESTAURANT_ID)[config.COL_REVIEW_ID].nunique()

        selected_ids = ids_array[ids_array > config.MIN_REVIEWS].sample(config.N_RESTAURANTS,random_state = 5).index.tolist()

        output_df = input_df[input_df[config.COL_RESTAURANT_ID].isin(selected_ids)]

        output_df.to_csv(self.sampled_data_path_rag)
        return output_df
    

    def generate_final_restaurant_list_for_rag(self):
        """
        Execute the Yelp dataset extraction process end-to-end, starting with the entire Yelp dataset and ending with 
        a small quantity of restaurants with a minimum number of reviews to use in the RAG.
        """
        print("Sampling Yelp datasets...")
        
        reviews_sample = self.import_yelp_data_sample()
        
        print(f"Yelp dataset sample with shape {reviews_sample.shape} created.")
        print(f"Yelp dataset sample stored at {self.sampled_data_path_reviews}")
        print(f"Sampling Yelp dataset for {config.N_RESTAURANTS} restaurants to use in this RAG demo.")
        print("")
        
        reviews_df = self.select_restaurants_for_rag(reviews_sample)
        
        print(f"RAG dataset consisting of {config.N_RESTAURANTS} and with shape {reviews_df.shape} created.")
        print(f"RAG dataset sample stored at {self.sampled_data_path_rag}")