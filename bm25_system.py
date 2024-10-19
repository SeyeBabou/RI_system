from rank_bm25 import BM25Plus
import numpy as np
import preprocessing as pr

def train(tokenized_corpus):
    bm25 = BM25Plus(tokenized_corpus)
    return bm25

def inference(query, model, index2file, k=56):
    def softmax(scores):
        """
        Applique la fonction softmax à un tableau de scores.

        :param scores: Liste ou tableau de scores.
        :return: Tableau de probabilités résultant de l'application de softmax.
        """
        # Calcul de l'exponentielle des scores en soustrayant le max pour la stabilité numérique
        exp_scores = np.exp(scores - np.max(scores))

        # Normalisation pour obtenir des probabilités
        return exp_scores / np.sum(exp_scores)

    def k_argmax(scores, k=k):
        """
        Renvoie les indices des k scores les plus élevés dans un tableau.

        :param scores: Liste ou tableau contenant les scores.
        :param k: Le nombre d'indices à renvoyer (correspondant aux scores les plus élevés).
        :return: Liste des indices des k scores les plus élevés, triés par ordre décroissant.
        """
        # Utilisation de np.argsort pour obtenir les indices triés
        indices_tries = np.argsort(scores)[::-1]  # Tri décroissant
        return indices_tries[:k]

    tokenized_query = pr.tokenize(query)
    doc_scores = softmax(model.get_scores(tokenized_query))
    doc_indexes = k_argmax(doc_scores)
    files_list = [index2file[index] for index in doc_indexes]
    return files_list , np.sort(doc_scores)[::-1][:k]