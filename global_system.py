import paths
import preprocess as pr
import bm25_system
import sbert_system

def combine_scores(bm25_scores, embedding_scores, alpha=0.5, beta=0.5):
    """
    this function combines the scores of the bm25 and the semantic search
    :param bm25_scores: the scores of the bm25 model
    :param embedding_scores: the scores of the semantic search model
    :param alpha: the weight of the bm25 model
    :param beta: the weight of the semantic search model
    :return: the combined scores

    """
    # combine the scores of the bm25 and the semantic search in a dictionary
    combined_scores = {}

    for doc_id in set(bm25_scores.keys()).union(embedding_scores.keys()):
        bm25_score = bm25_scores.get(doc_id, 0)   # Si l'index est manquant, on suppose un score de 0
        embedding_score = embedding_scores.get(doc_id, 0)

        # to calculate the combined score
        combined_scores[doc_id] = (alpha * bm25_score) + (beta * embedding_score)

    # to sort the documents by their scores
    sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    # return the document ids sorted by their scores
    return [doc_id for doc_id, score in sorted_docs]



def initilaize_system():
    # load data
    chunks, chunk2file, tokens = pr.load_data(paths.preprocessed_data2)

    # train bm25 model
    bm25 = bm25_system.train(tokenized_corpus=tokens)

    # load sbert model
    sbert_model = sbert_system.init()
    encoded_corpus = sbert_system.encode_corpus(corpus=chunks,
                                                model=sbert_model)
    return bm25, sbert_model, encoded_corpus, chunk2file, chunks, tokens


def get_documents(query,tokens, bm25, sbert_model, encoded_corpus, chunk2file, chunks):
    
    

    #print("\n\n\n---------------> BM25")
    bm25_scores = bm25_system.inference(query=query, 
                                        model=bm25, 
                                        index2file=chunk2file)
    """for index , score in bm25_scores.items():
        print(f"score : {score}\tdocument : {ind2file[index]}")"""
        
        
        
    #print("\n\n\n---------------> SEMANTIC SEARCH")  
    sr_scores = sbert_system.semantic_search(query=query, 
                                            model=sbert_model,
                                            corpus=chunks,
                                            corpus_embeddings=encoded_corpus)

    final_doc = combine_scores(bm25_scores=bm25_scores,
                               embedding_scores=sr_scores)

    #print(final_doc,"\n\n")
    print(f"requête : {query}")
    print("Les 5 meilleurs documents par ordre de pertinence sont : ")
    result = []
    for i in range(5):
        print(f"{i} {chunk2file[final_doc[i]]}")
        result.append(chunk2file[final_doc[i]])  
    return result 

# test the system (terminal) before doing the interface
bm25, sbert_model, encoded_corpus, chunk2file, chunks, tokens = initilaize_system()
while True:
    query = input("Entrez votre requête : ")
    if query == "exit":
        break
    get_documents(query,tokens, bm25, sbert_model, encoded_corpus, chunk2file, chunks)
    print("\n\n\n")