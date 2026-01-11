import sys
import os
import pandas as pd
import numpy as np
import json

import src.config as config

class ImportYelpReviewText:

    def __init__(self,raw_data_path,sampled_data_path):
        """
        """
        self.reviews_path = os.path.join(raw_data_path,"yelp_academic_dataset_review.json")
        self.business_path = os.path.join(raw_data_path,"yelp_academic_dataset_business.json")
        self.sampled_data_path = sampled_data_path


    @staticmethod
    def import_sample_from_complete_dataset(import_path):
        """
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
        """
        reviews = self.import_sample_from_complete_dataset(self.reviews_path)
        businesses = self.import_sample_from_complete_dataset(self.business_path)
        
        reviews_sample = pd.merge(
            pd.DataFrame(reviews),
            pd.DataFrame(businesses),
            how = "inner",
            on = "business_id",
            suffixes = ["_reviews","_restaurant"]
        ).dropna()

        reviews_sample.to_csv(os.path.join(self.sampled_data_path,"reviews_sample.csv"))
        return reviews_sample
    
    def select_restaurants_for_rag(self,input_df):
        """
        """

        cond1 = input_df[config.COL_BUSINESS_CATEGORY].str.lower().str.contains("restaurant")
       
       # many hotels contain restaurants but are not the focus of this RAG
        cond2 = ~input_df[config.COL_BUSINESS_CATEGORY].str.lower().str.contains("hotel|cinema",regex = True) 

        input_df = input_df[cond1 & cond2]

        ids_array = input_df.groupby(config.COL_RESTAURANT_ID)[config.COL_REVIEW_ID].nunique()

        selected_ids = ids_array[ids_array > config.MIN_REVIEWS].sample(config.N_RESTAURANTS,random_state = 5).index.tolist()

        output_df = input_df[input_df[config.COL_RESTAURANT_ID].isin(selected_ids)]

        output_df.to_csv(os.path.join(self.sampled_data_path,"reviews_df.csv"))
        return output_df
    

    def generate_final_restaurant_list_for_rag(self):
        """
        """
        print("Sampling Yelp datasets...")
        
        reviews_sample = self.import_yelp_data_sample()
        
        print(f"Yelp dataset sample with shape {reviews_sample.shape} created.")
        print(f"Yelp dataset sample stored as 'reviews_sample.csv' at {self.sampled_data_path}")
        print(f"Sampling Yelp dataset for {config.N_RESTAURANTS} restaurants to use in this RAG demo.")
        
        reviews_df = self.select_restaurants_for_rag(reviews_sample)
        
        print(f"RAG dataset consisting of {config.N_RESTAURANTS} and with shape {reviews_df.shape} created.")
        print(f"RAG dataset sample stored as 'reviews_df.csv' at {self.sampled_data_path}")
