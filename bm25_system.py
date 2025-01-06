from rank_bm25 import BM25Plus
import numpy as np
import preprocessing as pr
import test as pr2

def train(tokenized_corpus):
    bm25 = BM25Plus(tokenized_corpus)
    return bm25

def inference(query, model, index2file, k=56):
    def softmax(scores):
        """
        Applies the softmax function to an array of scores.

        :param scores: List or array of scores.
        :return: Array of probabilities resulting from applying softmax.
        """
        # Calculate the exponential of the scores, subtracting the max for numerical stability
        exp_scores = np.exp(scores - np.max(scores))

        # Normalize to get probabilities
        return exp_scores / np.sum(exp_scores)

    def k_argmax(scores, k=k):
        """
        Returns the indices of the top k highest scores in an array.

        :param scores: List or array containing the scores.
        :param k: The number of indices to return (corresponding to the highest scores).
        :return: List of indices of the top k scores, sorted in descending order.
        """
        # Use np.argsort to get the sorted indices
        indices_sorted = np.argsort(scores)[::-1]  # Descending sort
        return indices_sorted[:k]

    tokenized_query = pr.tokenize(query)
    doc_scores = softmax(model.get_scores(tokenized_query))
    doc_indexes = k_argmax(doc_scores)
    files_list = [index2file[index] for index in doc_indexes]
    doc_scores = np.sort(doc_scores)[::-1][:k]
    scores_dict = {}
    for i in range(len(files_list)):
        scores_dict[doc_indexes[i]] = doc_scores[i]
    return scores_dict

chunks, chunk2file, tokens = pr2.load_data(path="preprocessed_data2")
tokenized_corpus = tokens
bm25 = train(tokenized_corpus=tokenized_corpus)
query = "la methode Bi-Conjugate Gradient Stabilized (BiCGStab)"
index2file = chunk2file
sc_dict = inference(query=query, model=bm25, index2file=index2file)
count = 0
for index, score in sc_dict.items():
    print(f"score : {score}\tdocument : {index2file[index]}")
    print(f"Document content : {chunks[index][:300]}")
    count += 1
    if count == 3:
        break