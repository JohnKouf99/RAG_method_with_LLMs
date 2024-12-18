import os
import re
import ollama
from search import *
from harvester import *
from translate import *
from groq_api import *
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import time
import tiktoken
os.environ["OLLAMA_MODE"] = "remote"


class main:

    def __init__(self, query, n, model="llama3.1:8b"):
        self.query = query
        self.n = n
        self.model = model

    #function to truncate info if the prompt token limit is exceeded
    def count_tokens(self, text, info):
        tokenizer = tiktoken.get_encoding(self.model)
        token_count = len(tokenizer.encode(text))
        if token_count > 8192:
            pass

    #method to remove 20% of text for cases when the input token limit is exceeded 
    def truncate_text(self, text):
        words = text.split()
        total_words = len(words)
        words_to_keep = int(total_words * 0.8)
        truncated_text = ' '.join(words[:words_to_keep])
        sentence_end = re.search(r'\.|\?|!$', truncated_text)  
        if sentence_end:
            truncated_text = truncated_text[:sentence_end.end()]
        
        return truncated_text


            


    

    #based on a claim, implement a searcher and a harvester 
    def retrieve_knowledge(self, max_sources):

        #scan the web for urls containing knowledge
        my_searcher = GoogleSearch('https://www.google.com')
        url_list = my_searcher.google_search(self.query, self.n + 3)

        #harvest the external urls using a harvester instance
        my_harvester = Harvester(list(url_list), self.query, timeout=1000, claim_id=0, max_sources = max_sources)
        df = my_harvester.run()

        #get the bodies of the top-n web sources that has the biggest "body_similarity" value
        try:
            result = df.nlargest(self.n, 'body_similarity')['similar_sentences'] 
        except Exception as e:
            print('Could not find relevant sources.')
            return None
            
        return result
    
    def prompt_model(self, max_sources):
        external_sources = self.retrieve_knowledge(max_sources)
        if(external_sources is not None):
            info = "\n\n".join(external_sources)
            print('\n')
            print('------------------External knowledge----------------------')
            print(info)
            print('----------------------------------------------------------')

            input = f""" 
"Έχεις στη διάθεσή σου μια πληροφορία '{info}' και μία δήλωση: '{self.query}' της οποίας η ακρίβεια πρέπει "
    "να αξιολογηθεί. Χρησιμοποίησε μόνο την παρεχόμενη πληροφορία σε συνδυασμό με τις γνώσεις σου ώστε να αποφασίσεις "
    "εάν η δήλωση είναι ΑΛΗΘΗΣ, ΨΕΥΔΗΣ, ΜΕΡΙΚΩΣ-ΑΛΗΘΗΣ ή ΜΕΡΙΚΩΣ-ΨΕΥΔΗΣ.\n\n"
    "Πριν αποφασίσεις:\n\n"
    "1. Ανάλυσε με σαφήνεια τη δήλωση για να κατανοήσεις το περιεχόμενό της και να εντοπίσεις τα κύρια σημεία "
    "που πρέπει να αξιολογηθούν.\n"
    "2. Σύγκρινε τη δήλωση με την πληροφορία που έχεις, αξιολογώντας κάθε στοιχείο της δήλωσης ξεχωριστά.\n"
    "3. Χρησιμοποίησε τις γνώσεις σου ΜΟΝΟ σε συνδυασμό με την παρεχόμενη πληροφορία, αποφεύγοντας την αναφορά σε "
    "μη εξακριβωμένες πληροφορίες.\n\n"
    "Αποτέλεσμα: Δώσε μια ξεκάθαρη απάντηση επιλέγοντας μία από τις παρακάτω ετικέτες:\n\n"
    "- ΑΛΗΘΗΣ: Αν η δήλωση είναι απόλυτα επιβεβαιωμένη από την πληροφορία και τα στοιχεία σου.\n"
    "- ΨΕΥΔΗΣ: Αν η δήλωση διαψεύδεται ξεκάθαρα από την πληροφορία και τα στοιχεία σου.\n"
    "- ΜΕΡΙΚΩΣ-ΑΛΗΘΗΣ: Αν η δήλωση περιέχει κάποια σωστά στοιχεία, αλλά όχι απόλυτα ακριβή.\n"
    "- ΜΕΡΙΚΩΣ-ΨΕΥΔΗΣ: Αν η δήλωση περιέχει κάποια σωστά στοιχεία, αλλά περιέχει παραπλανητικές ή ανακριβείς πληροφορίες.\n\n"
    "Τέλος, εξήγησε τη λογική σου με σαφήνεια και επικεντρώσου στα δεδομένα που παρέχονται και στη δική σου γνώση. "
    "Απόφυγε περιττές λεπτομέρειες και προσπάθησε να είσαι ακριβής και περιεκτικός στην ανάλυσή σου."
    "Οι απαντήσεις σου πρέπει να έχουν την μορφή:"
            "Δήλωση: '{self.query}'"
            "Αποτέλεσμα δήλωσης:" 
            "Δικαιολόγηση:"
    """


            
        else:


            input = f""" 
                Πάρε μία βαθιά ανάσα και έλενξε παρακάτω δήλωση: '{self.query}' 
                Πες μου αν η δήλωση είναι αληθής, ψευδής, εν-μερει αληθής, η εν-μέρει ψευδής
                Δείξε μου διάφορες πηγές που δικαιολογούν την απάντησή σου.
                Οι απαντήσεις σου πρέπει να έχουν την μορφή:

                Δήλωση: '{self.query}'
                Αποτέλεσμα δήλωσης: 
                Πηγές που υποστηρίζουν αυτήν την αξιολόγηση:

                
                """

        
        while True:
            try:
                response = ollama.chat(model=self.model, messages=[{"role": "user", "content": translate_long_text(input, src_lang='el', target_lang='en')}])
                break
            except Exception as e:
                print(e)
                print("Retrying with smaller input....")
                info = self.truncate_text(info)
            
        answer = response['message']['content']
        answer = translate_long_text(answer, src_lang='en', target_lang='el')
        # print()
        print('----------------ANSWER----------------')
        print(answer)

        # Output the response from the model
        return answer
    
    def run_api(self, max_sources):
        external_sources = self.retrieve_knowledge(max_sources)
        
        if(external_sources is not None):
            info = "\n\n".join(external_sources)
            print('\n')
            print('------------------External knowledge----------------------')
            print(info)
            print('----------------------------------------------------------')
        api = groq_api(info, self.query)
        response = api.run()
        return response
    
def run():
    while True:
        print("You can quit by typing 'exit' ,'quit' or 'bye'")
        user_input1 = input("Insert claim here: ")
        if user_input1.lower().strip() in ["exit", "quit", "bye"]:
            print("Exiting...")
            break
        user_input2 = input("Insert number of relevant web-souces to harvest (max number is 5): ")  
        if user_input2.lower().strip() in ["exit", "quit", "bye"]:
            print("Exiting...")
            break
        start_time = time.time()   
        model = main(str(user_input1) , int(user_input2))
        # make the model run on groq api
        # response = model.run_api()
        # print(response['response'])

        #run the model with ollama, max sources are the number of web sources plus 2
        model.prompt_model(max_sources=int(user_input2)+2)
        
        end_time = time.time()
        print(f'RAG method was completed in {round(end_time - start_time,2)} "seconds"')    


if __name__ == "__main__":
    run()


        
