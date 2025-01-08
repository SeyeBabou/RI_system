from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import paths

# Exemple de documents
docs = [
    Document(page_content="An adverb modifies a verb, an adjective, or another adverb."),
    Document(page_content="An adjective modifies a noun or pronoun."),
]

# Création d'un retriever avec FAISS et paraphrase-MiniLM-L6-v2
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
texts = text_splitter.split_documents(docs)

# Utiliser paraphrase-MiniLM-L6-v2 pour l'embedding
model_name = paths.sbert_model_path  # Assurez-vous que paths.sbert_model_path contient le chemin ou le nom correct du modèle
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Création du vectorstore avec FAISS
vectorstore = FAISS.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever()

# Utiliser Ollama pour le modèle génératif
llm = OllamaLLM(model="llama3.2:1B")

# Création de la chaîne RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,  # Fournir le retriever ici
    return_source_documents=True,
)

# Prompt
prompt = "What is the difference between an adverb and an adjective?"
response = qa(prompt)
print(response)

