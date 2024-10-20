from sentence_transformers import SentenceTransformer, util
import numpy as np

# Charger le modèle SBERT
def init():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model

# Corpus d'articles (phrases ou documents)
def encode_corpus(corpus , model):
    # Encoder le corpus pour obtenir les embeddings des phrases
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    return corpus_embeddings

# Fonction de recherche sémantique basée sur SBERT
def semantic_search(query, model, corpus, corpus_embeddings, top_k=56):
    # Encoder la requête
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Calculer la similarité cosinus entre la requête et les documents
    cosine_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]

    # Ajuster top_k si nécessaire
    top_k = min(top_k, len(corpus))

    # Récupérer les indices des documents les plus similaires (top_k)
    top_results = np.argsort(cosine_scores.cpu())[-top_k:]

    # Inverser explicitement pour avoir les scores en ordre décroissant
    top_results = top_results.tolist()
    top_results.reverse()
    
    scores = {}
    for idx in top_results:
        scores[idx] = cosine_scores[idx].item()
    return scores

"""# Exemple de recherche
query = "techniques d'apprentissage profond"
semantic_search(query, corpus, corpus_embeddings)"""
