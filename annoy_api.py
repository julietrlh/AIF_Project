from flask import Flask, request, jsonify
from annoy import AnnoyIndex
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the Annoy database
annoy_db = AnnoyIndex(576, metric='angular')  
annoy_bow = AnnoyIndex(5000, 'angular')
annoy_glove = AnnoyIndex(100, 'angular')
                                            
annoy_db.load('rec_imdb.ann')  
annoy_bow.load('annoy_index_bow.ann') 
annoy_glove.load('annoy_index_glove.ann')

count_vectorizer = pd.read_pickle('./count_vectorizer.pkl')
glove_model = pd.read_pickle('./glove_model.pkl')


@app.route('/') # This is the home route, it just returns 'Hello world!'
def index():    # I use it to check that the server is running and accessible it's not necessary
    return 'Hello world!'

@app.route('/reco', methods=['POST']) # This route is used to get recommendations
def reco():
    #vector = request.json['vector'] # Get the vector from the request
    vector = request.json.get('vector')
    closest_indices = annoy_db.get_nns_by_vector(vector, 5) # Get the 5 closest elements indices
    reco = closest_indices[:5]  # Assuming the indices are integers
    return jsonify(reco) # Return the reco as a JSON


# Définir un endpoint pour la recherche des plus proches voisins
@app.route('/api/search', methods=['POST'])
def search_nearest_neighbors():
    # Obtenir les données de la requête (aperçu du film)
    text = request.json.get('text')

    matrix= count_vectorizer.transform([text])
    
    vector = matrix.toarray()[0]

    closest_indices = annoy_bow.get_nns_by_vector(vector, 5) # Get the 5 closest elements indices
    reco = closest_indices[:5]  # Assuming the indices are integers

    return jsonify(reco) # Return the reco as a JSON


@app.route('/api/glove', methods=['POST'])
def compute_mean_embeddings():
    # Obtenir les données de la requête (aperçu du film)
    text = request.json.get('text')

    s=text.lower()
    words_list = glove_model.index_to_key
    emb_list = [glove_model[w] for w in s if w in words_list]

    if emb_list != []:
        embe_list = np.mean(emb_list, axis=0)
    else:
        embe_list = np.zeros(100)

    closest_indices = annoy_glove.get_nns_by_vector(embe_list, 5) # Get the 5 closest elements indices
    reco = closest_indices[:5]  # Assuming the indices are integers

    return jsonify(reco) # Return the reco as a JSON



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) # Run the server on port 5000 and make it accessible externally
