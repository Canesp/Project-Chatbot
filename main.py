from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np

# TODO 
# pipenv run uvicorn main:app --reload

# Creates the API
app = FastAPI() 

# read in Test Json file
#-----------------------------

# path to test file
test_url = "test_data/test_file_01.JSON"

f = open(test_url)

# loads the data as a dictionary
j_file = json.load(f)

#-----------------------------

# Sets the url path
@app.get("/sentence")

# the function called when requesting an answer form the chatbot. 
def question(sentence = None):
    
    if sentence is None:

        return 'Hello! how can i help you?'
    
    else: 

        # sets a model name for sentence comp
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Creates the model
        model = SentenceTransformer(model_name)

        # list of answer keys
        answer_keys = list(j_file.keys())
     
        # makes the sentences into vecs 
        input_vecs = model.encode(sentence)
        answer_vecs = model.encode(answer_keys)

        # calculates sentence comp weights 
        weights = list(cosine_similarity([input_vecs], answer_vecs))

        # gets max weight index
        max_value = np.argmax(weights)

        return j_file[answer_keys[max_value]]


