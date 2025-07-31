# make sure to pip install langchain-community

from langchain_ollama import OllamaEmbeddings

# for later (web deployment), will need to set up credentials and paid service (GPT)
# encoder
import os
from getpass import getpass
# remember to pip install -qU langchain-openai
from langchain_openai import OpenAIEmbeddings

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass("Enter your OpenAI API key: ")
    # embeddings = OpenAIEmbeddings(name="text-embedding-3-large")
    return embeddings