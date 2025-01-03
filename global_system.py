import paths
import preprocessing as pr
import tf_idf
import bm25_system
import semantic_search

def combine_scores(bm25_scores, embedding_scores, alpha=0.5, beta=0.5):
    """
    Combine les scores BM25 et Embeddings avec une pondération et renvoie
    les documents triés par ordre de pertinence.

    Args:
    - bm25_scores (dict): Dictionnaire des scores BM25 {index: score}.
    - embedding_scores (dict): Dictionnaire des scores d'embeddings {index: score}.
    - alpha (float): Pondération pour BM25.
    - beta (float): Pondération pour les embeddings sémantiques.

    Returns:
    - list: Liste des indexes triés par score de pertinence décroissant.
    """
    # Combine les deux systèmes de notation dans un seul dictionnaire
    combined_scores = {}

    # Fusionner les scores en utilisant les poids alpha et beta
    for doc_id in set(bm25_scores.keys()).union(embedding_scores.keys()):
        bm25_score = bm25_scores.get(doc_id, 0)   # Si l'index est manquant, on suppose un score de 0
        embedding_score = embedding_scores.get(doc_id, 0)

        # Calcul du score combiné
        combined_scores[doc_id] = (alpha * bm25_score) + (beta * embedding_score)

    # Trier les documents par leur score final (du plus grand au plus petit)
    sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    # Retourner seulement la liste des indexes, triée par pertinence
    return [doc_id for doc_id, score in sorted_docs]



def load_data() :
    #load data
    inverted_index, ind2tok, ind2file , ind2text = pr.load_data(f"{paths.preprocessed_data}/preprocessed_data")
    return inverted_index, ind2tok, ind2file , ind2text


def get_documents(query, inverted_index, ind2tok, ind2file , ind2text):
    
    #corpus (for tf_idf and semantic search)
    corpus = []
    for _ , text in ind2text.items():
        corpus.append(text)

    print("nombre de documents : ", len(corpus))

    #tokenized corpus for bm25 algo
    tokenized_corpus = []
    for _ , tokens in ind2tok.items():
        tokenized_corpus.append(tokens)
        

    #print("---------> TF-IDF")
    #tf-idf matrix
    X = tf_idf.get_matrix(corpus=corpus)

    
    tf_idf_scores = tf_idf.get_scores(query , X , inverted_index)
    #print(X.shape)

    """for index , score in tf_idf_scores.items():
        print(f"score : {score}\tdocument : {ind2file[index]}")"""
        

    #print("\n\n\n---------------> BM25")
    bm25 = bm25_system.train(tokenized_corpus=tokenized_corpus)
    bm25_scores = bm25_system.inference(query=query, model=bm25, index2file=ind2file)
    """for index , score in bm25_scores.items():
        print(f"score : {score}\tdocument : {ind2file[index]}")"""
        
        
        
    #print("\n\n\n---------------> SEMANTIC SEARCH")  
    sbert_model = semantic_search.init()
    encoded_corpus = semantic_search.encode_corpus(corpus=corpus,
                                                model=sbert_model)

    sr_scores = semantic_search.semantic_search(query=query, 
                                                                model=sbert_model,
                                                                corpus=corpus,
                                                                corpus_embeddings=encoded_corpus)
    """for index , score in sr_scores.items():
        print(f"score : {score}\tdocument : {ind2file[index]}")"""

    #combiner les résultats pour avoir le classement global :  (0.5 * bm25 + 0.4 * semantic_search + 0.1 * tf_idf)
    final_doc = combine_scores(tfidf_scores=tf_idf_scores,
                            bm25_scores=bm25_scores,
                            embedding_scores=sr_scores)

    #print(final_doc,"\n\n")
    print(f"requête : {query}")
    print("Les 5 meilleurs documents par ordre de pertinence sont : ")
    result = []
    for i in range(5):
        print(f"{i} {ind2file[final_doc[i]]}")
        result.append(ind2file[final_doc[i]])
    return result

#query = "Meshless methods have attracted much attention in recent years for a wide range of engineering sciences"
#inverted_index, ind2tok, ind2file , ind2text = load_data()        
#get_documents(query , inverted_index, ind2tok, ind2file , ind2text)
