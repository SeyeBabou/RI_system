from collections import defaultdict

# Exemple d'index inversé (mot -> liste des IDs de documents contenant ce mot)
inverted_index = {
    "apprentissage": [0, 1, 2, 3],
    "automatique": [0, 2, 3],
    "intelligence": [0, 1],
    "artificielle": [0, 1],
    "profond": [2],
    "algorithmes": [3],
    "données": [3]
}

# Fonction pour gérer les requêtes booléennes
def boolean_search(query, inverted_index):
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

# Exemple d'utilisation de la fonction
query_1 = "apprentissage AND automatique AND intelligence"
"""query_2 = "intelligence OR profond"
query_3 = "apprentissage AND intelligence"""

print("Résultat pour '{}': {}".format(query_1, boolean_search(query_1, inverted_index)))

