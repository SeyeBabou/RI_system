from sentence_transformers import SentenceTransformer, util
import numpy as np
import preprocess as pr
import paths 


# Load the SBERT model
def init():
    try:
        # Specify the path to the model
        model_path = paths.sbert_model_path
        
        # Attempt to load the model from the specified path
        model = SentenceTransformer(model_path)
        print(f"Model successfully loaded from {model_path}")
        return model
    except FileNotFoundError:
        # Handle the case where the specified path does not exist
        print(f"Error: The specified model path ({model_path}) was not found.")
    except Exception as e:
        # Handle any other unexpected exceptions
        print(f"An error occurred while loading the model: {e}")
    # Return None if an error occurs
    return None


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

"""corpus = ["Deep learning is a subset of machine learning in artificial intelligence (AI) that has networks capable of learning unsupervised from data that is unstructured or unlabeled.",
            "Machine learning (ML) is the study of computer algorithms that improve automatically through experience.",
            "Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs.",
            "Unsupervised learning is a type of machine learning that looks for previously undetected patterns in a data set with no pre-existing labels and with a minimum of human supervision."]
"""
#Example of search
chunks, chunk2file, tokens = pr.load_data(path="preprocessed_data2")
corpus = chunks
query = "la methode Bi-Conjugate Gradient Stabilized (BiCGStab)"
model = init()
corpus_embeddings = encode_corpus(corpus, model)
scores = semantic_search(query, model, corpus, corpus_embeddings)
count = 0
for doc_id, doc_sc in scores.items():
    print(f"score : {doc_sc}\tdocument : {chunk2file[doc_id][1]}")
    print(f"Document content : {chunks[doc_id][:300]}")
    count += 1
    if count == 3:
        break
