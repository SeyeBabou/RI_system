from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import preprocessing as pr



vectorizer = TfidfVectorizer(stop_words='english')

def get_matrix(corpus): #56 * nb_terms
    """Fonction pour avoir la matrice qui représente les documents"""
    X = vectorizer.fit_transform(corpus)
    return X




def boolean_search(query, inverted_index):
    """Fonction pour gérer les requêtes booléennes"""
    def intersect(list1, list2):
        # Retourne l'intersection de deux listes
        return list(set(list1) & set(list2))

    def union(list1, list2):
        # Retourne l'union de deux listes
        return list(set(list1) | set(list2))

    def negate(list1, all_docs):
        # Retourne les documents qui ne sont pas dans la liste
        return list(set(all_docs) - set(list1))

    # Obtenir la liste de tous les documents présents dans l'index
    all_docs = set(doc_id for doc_list in inverted_index.values() for doc_id in doc_list)

    # Diviser la requête en mots et opérateurs (simple parsing, une expression par mot)
    tokens = query.lower().split()

    # Initialisation de la liste des documents résultat
    result = []

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token in inverted_index:  # Mot normal
            current_docs = inverted_index[token]
        elif token == "and":  # Opérateur AND
            next_token = tokens[i + 1]
            if next_token in inverted_index:
                current_docs = intersect(result, inverted_index[next_token])
            else:
                current_docs = []
            i += 1
        elif token == "or":  # Opérateur OR
            next_token = tokens[i + 1]
            if next_token in inverted_index:
                current_docs = union(result, inverted_index[next_token])
            else:
                current_docs = result  # Si le mot n'est pas trouvé, on ne change rien
            i += 1
        elif token == "not":  # Opérateur NOT
            next_token = tokens[i + 1]
            if next_token in inverted_index:
                current_docs = negate(inverted_index[next_token], all_docs)
            else:
                current_docs = list(all_docs)  # Si le mot n'est pas trouvé, considérer tous les documents
            i += 1
        else:
            current_docs = []  # Si le mot n'est pas trouvé dans l'index

        # Si c'est la première opération, on initialise le résultat
        if i == 0 or tokens[i - 1] == "or":
            result = current_docs
        elif tokens[i - 1] == "and":
            result = intersect(result, current_docs)

        i += 1

    return result

def get_scores(query , X , inverted_index , bool_operator=False ):  # X : 56 * nb_of_term
    """Appretissage automatique intelligence artificielle"""
    def get_scores (matrix):
        #print(f"query : {query}")
        query_vector = vectorizer.transform([query]).toarray()
        #print(f"le vecteur représentant la requête '{query}' est {query_vector}")

        similarities = cosine_similarity(query_vector, matrix)[0]
        #print("similarités : ", similarities)

        documents = list(enumerate(similarities))
        documents.sort(key=lambda x: x[1], reverse=True)
        #print(documents)
        
        scores = {}
        for element in documents:
            index_doc , score = element
            scores[index_doc] = score
        return scores
    
    def transform_query(query):
        query_tokens = pr.tokenize(query)
        new_query = " AND ".join(query_tokens)
        return new_query
    
    if bool_operator:
        doc_list = boolean_search(transform_query(query) , inverted_index) # liste doc plus pertinents -> [5, 25 , 35]
        nb_of_doc = len(doc_list)
        if nb_of_doc == 0 : 
            return get_scores(X)
        
        else : 
            new_x =  X[doc_list, :]
            return get_scores (new_x)
    else:
        return get_scores(X)
    
