from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load the SBERT model
def init():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model

# Corpus of articles (sentences or documents)
def encode_corpus(corpus , model):
    # Encode the corpus to get the embeddings of the sentences
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    return corpus_embeddings

# Semantic search function based on SBERT
def semantic_search(query, model, corpus, corpus_embeddings, top_k=56):
    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Calculate the cosine similarity between the query and the documents
    cosine_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]

    # Adjust top_k if necessary
    top_k = min(top_k, len(corpus))

    # Get the indices of the most similar documents (top_k)
    top_results = np.argsort(cosine_scores.cpu())[-top_k:]

    # Explicitly reverse to have the scores in descending order
    top_results = top_results.tolist()
    top_results.reverse()
    
    scores = {}
    for idx in top_results:
        scores[idx] = cosine_scores[idx].item()
    return scores

"""# Example of search
query = "deep learning techniques"
semantic_search(query, corpus, corpus_embeddings)"""
