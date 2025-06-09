import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings


# Load your previously scraped documents
with open("iphone_docs.pkl", "rb") as f:
    documents = pickle.load(f)

# Chunk the documents (important for RAG)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = splitter.split_documents(documents)

# # Use Ollama's local embedding model (nomic-embed-text)
# embedding = OllamaEmbeddings(model="nomic-embed-text")
# Use OpenAI embeddings
embedding = OpenAIEmbeddings()  # Defaults to 'text-embedding-ada-002'

# Create FAISS index from the documents
vectorstore = FAISS.from_documents(split_docs, embedding)

# Save the vectorstore to disk
vectorstore.save_local("faiss_index")
