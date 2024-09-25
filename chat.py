from openai import BadRequestError, OpenAI, RateLimitError
import os
import openai
os.environ['OPENAI_API_KEY'] = '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'

def get_embedding(text):
    response = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
    return response["data"][0]["embedding"]


from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a proffesional suspicious fact-checker that speaks Greek. Your job is to classify claims as true, false, partially true or partially false."},
        {
            "role": "user",
            "content": "Πάρε τον χρόνο να σκεφτείς την παρακάτω δήλωση: η Ελλάδα δέχεται την μεγαλύτερη πίεση από την παράνομη μετανάστευση, σύμφωνα με έρευνα. Δείξε μου αξιόπηστες πηγές."
        }
    ]
)
print(completion.choices[0].message.content)

