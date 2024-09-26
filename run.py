import sys
import pandas as pd
sys.path.append('scripts')  
from scripts.search import *
from scripts.harvester import *
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from openai import BadRequestError, OpenAI, RateLimitError
import os
import openai
os.environ['OPENAI_API_KEY'] = ''
from openai import OpenAI
client = OpenAI()

class run:

    #initialize a query and utilize the top-n websites content.
    def __init__(self, query, n):
        self.query = query
        self.n = n

    #may add some more features in later implementations
    def retrieve_knowledge(self):

        #scan the web for urls containing knowledge
        my_searcher = GoogleSearch('https://www.google.com')
        url_list = my_searcher.google_search(self.query, self.n + 3)

        #harvest the exterlan urls using a harvester instance
        my_harvester = Harvester(list(url_list), self.query, timeout=1000, claim_id=0, max_sources=3)
        df = my_harvester.run()

        #get the bodies of the top-n web sources that has the biggest "most_similar_par_cos" value
        try:
            result = df.nlargest(self.n, 'body_similarity')['body'] 
        except Exception as e:
            print('Could not find relevant sources.')
            return None
            
        
        return result
    
    def get_embedding(text):
        response = openai.Embedding.create(input=[text], model="text-embedding-small-3")
        return response["data"][0]["embedding"]
    
    def prompt_model(self):
        external_souces = self.retrieve_knowledge()
        if(external_souces is not None):
            info = "\n\n".join(self.retrieve_knowledge())
            print('\n')
            print('------------------External knowledge----------------------')
            print(info)
            print(print('----------------------------------------'))
            completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
            {"role": "system", "content": f"""
            You are a proffesional suspicious fact-checker that speaks Greek. 
            Your job is to classify claims as true, false, half-true or half false.
            Justify your classification
            """},
            {
                "role": "user",
                "content": f""" Βάσει των παρακάτω πληροφοριών: {info}
                Πάρε μία βαθιά ανάσα και έλενξε παρακάτω δήλωση: '{self.query}' 
                Δικαιολόγησε την απάντησή σου"""
            }
        ]
    )
            print(completion.choices[0].message.content)

        else: 
            print('Prompting the model without any external sources.....')
            print('\n\n')
            completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
            {"role": "system", "content": f"""
            You are a proffesional suspicious fact-checker that speaks Greek. 
            Your job is to classify claims as true, false, half-true or half false.
            Justify your classification
            """},
            {
                "role": "user",
                "content": f"""Πάρε μία βαθιά ανάσα και έλενξε παρακάτω δήλωση: '{self.query}' 
                Δικαιολόγησε την απάντησή σου"""
            }
        ]
    )
            print(completion.choices[0].message.content)


        

    
