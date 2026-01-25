import pandas as pd
import os
import requests
import src.config as config

class SummarizeRelevantReviewText:

    def __init__(self,
                 processed_data_path = None,
                 col_restaurant_id = None,
                 topics = None,
                 ollama_model = None,
                 temperature = None,
                 ollama_base_url = None,
                 parquet_engine = None
                 ):
        """
        @param processed_data_path: import file path for retrieving cleaned and chunked reviews data
        @param col_restaurant_id: column that stores the restaurant id in the Yelp dataset and metadata array
        @param topics: dictionary of topic headings (keys) and keywords associated with those topics (values)
        @param ollama_model: model called in Ollama to summarize the text chunks returned by the index search
        @param temperature: LLM randomness parameter
        @param ollama_base_url: URL for accessing Ollama via HTTP
        @param parquet_engine: engine used for encoding parquet files
        """

        defaults = {
            "processed_data_path": config.DATA_DIR_PROC,
            "col_restaurant_id": config.COL_RESTAURANT_ID,
            "topics": config.TOPICS,
            "ollama_model": config.OLLAMA_MODEL,
            "temperature": config.TEMPERATURE,
            "ollama_base_url": config.OLLAMA_BASE_URL,
            "parquet_engine": config.PARQUET_ENGINE
        }

        overrides = {
            "processed_data_path": processed_data_path,
            "col_restaurant_id": col_restaurant_id,
            "topics": topics,
            "ollama_model": ollama_model,
            "temperature": temperature,
            "ollama_base_url": ollama_base_url,
            "parquet_engine": parquet_engine
        }

        for name,default in defaults.items():
            value = overrides[name] if overrides[name] is not None else default
            setattr(self,name,value)

    def get_stored_relevant_review_text(self):
        """
        Retrieve the review chunks most relevant to each query topic for each restaurant
        @return relevant_chunk_df: relevant review text chunks as dataframe
        @return restaurant_ids: list of unique restaurant ids in relevant_chunk_df
        """
        relevant_chunk_df = pd.read_parquet(os.path.join(self.processed_data_path,"topic_relevant_review_chunks.parquet"))
        restaurant_ids = relevant_chunk_df[self.col_restaurant_id].unique().tolist()

        return relevant_chunk_df,restaurant_ids
    

    def call_ollama(self,prompt,model,temperature=0.2):
        """
        Pass a string-formatted prompt to Ollama and store the response as a JSON object
        @return: Ollama model response as JSON object
        """
    
        url = self.ollama_base_url
        payload = {
            "model":model,
            "prompt":prompt,
            "stream":False,
            "options":{"temperature":temperature}
        }

        resp = requests.post(url,json=payload,timeout=120)
        resp.raise_for_status()
        
        data = resp.json()
        return data["response"].strip()
    
    def retrieve_relevant_text_summaries_from_ollama(self,relevant_chunk_df,restaurant_ids):
        """
        Segregates review test chunks by restaurant and topic.
        Then constructs a prompt, passes to an LLM via Ollama, and returns the resulting summary.
        @param relevant_chunk_df: dataframe containing review text chunks, organized by restaurant and relevant topic.
        @param restaurant_ids: list of unique restaurant ids
        @return: dataframe containing summaries of each topic for each restaurants
        """


        summary_dict = {}
        for id in restaurant_ids:
            restaurant_df = relevant_chunk_df[relevant_chunk_df["business_id"] == id]
            restaurant_name = restaurant_df.iloc[0]["restaurant_name"]
            restaurant_summary_dict = {}
            for topic in self.topics.keys():
                topic_df = restaurant_df[restaurant_df["topic"]==topic]
                context = f"""
                You are analyzing and synthesizing customer reviews for a restaurant called {restaurant_name}, where the main topic is:
                
                Topic: {topic.upper()}

                Using only the reviews included here, provide a concise (2-3 sentence) summary of {restaurant_name}'s performance in the area of {topic}.

                Guidelines:
                - Capture common themes, impressions, and disagreements
                - Be factual and grounded in the evidence; do not invent details not supported by the excerpts
                - Begin by characterizing the group of reviews as "positive", "negative", or "mixed" 

                Output to avoid:
                - Content unrelated to the topic (for example: "noise levels" are ambiance, not food)
                - Review indicies (for example, "review 3"); those are not user-fsacing
                - Announcing the summary ("here is a concise summary..."); just start with summary content
                - Use of generic filler adjectives (e.g., cozy, delicious, amazing) unless the excerpts explicitly support them

                Important: if the excerpts don’t describe a topic clearly, say ‘<topic> details are limited in the provided reviews.’”

                """
                i = 0
                for row in topic_df.iterrows():

                    verbatim = f"""
                    [{i}] Review ({row[1].stars} stars):
                    {row[1].chunk}
                    """
                    context+=verbatim
                    i+=1
                
                summary = self.call_ollama(prompt = context,
                                    model = self.ollama_model,
                                    temperature=self.temperature
                                    ).replace("\n"," ")
                restaurant_summary_dict.update({topic:summary})
            print(f"     Summary for {restaurant_name} complete.")
            summary_dict.update({restaurant_name:restaurant_summary_dict})

        summary_df = pd.DataFrame(summary_dict).transpose().reset_index()
        return summary_df

    def summarize_relevant_review_text(self):
        """
        Transform a dataframe of text chunks into a dataframe of text summaries generated by an LLM, organized by restaurant and topic.
        @return: True if executed successfully
        """
        relevant_chunk_df, restaurant_ids = self.get_stored_relevant_review_text()
        summary_df = self.retrieve_relevant_text_summaries_from_ollama(relevant_chunk_df,restaurant_ids)
        summary_df.to_parquet(os.path.join(self.processed_data_path,"summaries.parquet"),engine = self.parquet_engine)
        print(f"  LLM summaries of review text chunks stored at {os.path.join(self.processed_data_path,"summaries.parquet")}")
        return True
