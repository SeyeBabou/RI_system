from sentence_transformers import SentenceTransformer, util
import numpy as np

# Charger le modèle SBERT
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Corpus d'articles (phrases ou documents)
def encode_corpus(corpus):
    # Encoder le corpus pour obtenir les embeddings des phrases
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    return corpus_embeddings

# Fonction de recherche sémantique basée sur SBERT
def semantic_search(query, corpus, corpus_embeddings, top_k=5):
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
    

    # Afficher les résultats
    print(f"Requête : '{query}'")
    print("\nTop résultats sémantiques :")
    
    for idx in top_results:
        print(f"Document ID : {idx}, Score : {cosine_scores[idx].item()}")
        print(f"Contenu : {corpus[idx]}")
        print()
    
    return top_results , cosine_scores

"""# Exemple de recherche
query = "techniques d'apprentissage profond"
semantic_search(query, corpus, corpus_embeddings)"""
