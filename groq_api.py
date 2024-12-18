import os
os.environ["GROQ_API_KEY"] = '' 
from langchain_groq import ChatGroq
import time
import numpy as np


class groq_api():

    def __init__(self, info, query):

        self.info = info
        self.query = query

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=1,)
        
        self.llm_2 = ChatGroq(
            model="llama-3.1-70b-versatile",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=1,)
        
        self.key_1 = ''
        self.key_2 = ''

  
        
    def run(self):
        max_retries = 10
        retries= 0
        messages = [
                    (
                        "system",
                       f"You are a fact-checker."

                    ),
                    ("human", f'''Έχεις στη διάθεσή σου μια πληροφορία '{self.info}' και μία δήλωση: '{self.query}' της οποίας η ακρίβεια πρέπει "
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
    '''),
                ]
        ai_msg = None

        while retries<max_retries:

            #alternate between api keys
            if retries % 2 ==0:
                 os.environ["GROQ_API_KEY"] = self.key_1
            else: 
                 os.environ["GROQ_API_KEY"] = self.key_2

            #try to create an api call with the first llm     
            try:

                start_time = time.time()
                ai_msg = self.llm.invoke(messages)

                end_time = time.time()
                break

            #if it fails to produce a result, try with the second llm
            except Exception as e:
                try:
                    start_time = time.time()
                    ai_msg = self.llm_2.invoke(messages)
                    end_time = time.time()
                    break

                #if a second llm doesn't work either, alternate between api keys by increasing the #retries                     
                except Exception as e:
                    #print(e)
                    retries+=1

        #if no answer could be generated, return none and invoke the local llm
        if ai_msg is None:  
            return {"response" : None, "elapsed_time": None}
      
        return {"response" : ai_msg.content,
                        "elapsed_time": np.round(end_time-start_time,2)}
            