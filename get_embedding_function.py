import os
from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    return OllamaEmbeddings(model=os.getenv('EMBEDDING_MODEL','nomic-embed-text'))
