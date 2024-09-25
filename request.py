import requests
import json
from googletrans import Translator
translator = Translator()
url = "http://localhost:11434/api/generate"

headers = {
    "Content-Type": "application/json"
}

data = {
    "model": "gemma2:9b",
    "prompt": 
    """ 
    Σκέψου προσεκτικά και απάντησε βήμα προς βήμα σαν ελεγκτής ψευδών ειδήσεων
   'Η Ραχήλ Μακρή πρώην βουλευτής των Ανεξαρτήτων Ελλήνων και του ΣΥ.ΡΙΖ.Α. συνέδεσε τα προγράμματα τροποποίησης του καιρού με την κλιματική αλλαγή, ύστερα από δηλώσεις ειδικού μετεωρολόγου, ότι τόσο στη χώρα μας όσο και παγκοσμίως τα προγράμματα ψεκασμού νεφών εφαρμόζονται στο πλαίσιο αντιμετώπισης της ξηρασίας και της ασφάλισης αγροτικών προϊόντων από χαλαζοπτώσεις.' 
    Μπορείς να μου πεις αν η παρακάτω φράση είναι αληθής, η μερικώς αληθής, η ψευδής, η μερικώς ψευδής, 
    με κατάλληλες αποδείξεις και πηγές; 
    """,
    "stream": False

}

response = requests.post (url, headers=headers, data=json.dumps(data))

if response.status_code==200:
    response_text = response.text
    data = json.loads(response_text)
    actual_response = data.get("response")
    print(actual_response)
    translated = translator.translate(actual_response, src='en', dest='el')
    print('\n')
    print(translated.text)
else:
    print("Error: ", response.status_code, response.text)