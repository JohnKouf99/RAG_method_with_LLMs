import sys
import pandas as pd
sys.path.append('scripts')  
from scripts.search import *
from scripts.harvester import *
import urllib3
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from openai import BadRequestError, OpenAI, RateLimitError
import os
import openai
# os.environ['OPENAI_API_KEY'] = ''
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
        url_list = my_searcher.google_search(self.query, 5)

        #harvest the exterlan urls using a harvester instance
        my_harvester = Harvester(list(url_list)[:3], self.query, timeout=1000, claim_id=0)
        df = my_harvester.run()

        #get the bodies of the top-n web sources that has the biggest "most_similar_par_cos" value
        result = df.nlargest(self.n, 'most_similar_par_cos')['body'] 
        
        return result
    
    def get_embedding(text):
        response = openai.Embedding.create(input=[text], model="text-embedding-small-3")
        return response["data"][0]["embedding"]
    
    def prompt_model(self):
        info = "\n\n".join(self.retrieve_knowledge())
        completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
        {"role": "system", "content": f"""
         You are a proffesional suspicious fact-checker that speaks Greek. 
         Your job is to classify claims as true, false, partially true or partially false.
         """},
        {
            "role": "user",
            "content": f""" Βάσει των παρακάτω πληροφοριών: {info}
            Πάρε τον χρόνο να σκεφτείς την παρακάτω δήλωση: '{self.query}' 
            Δικαιολόγησε την απάντησή σου"""
        }
    ]
)
        print(completion.choices[0].message.content)

    
