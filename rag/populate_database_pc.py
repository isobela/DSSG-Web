import argparse
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function

# using pinecone for vector store bc chroma does not support cosine similarity well (lots of conversions need to be made)
# pip install -qU langchain-pinecone
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone as PineconeClient

# to iterate over multiple PDF files
from glob import glob

# for when we have access to the OpenAI Key 
from langchain_openai import OpenAIEmbeddings

# for semantic chunking (commented out for now)
# from langchain_text_splitters import SemanticChunker

# for retrieving values securely from .env
from dotenv import load_dotenv

# ---------------------------------------------------
# Configuration section
# ---------------------------------------------------
load_dotenv()
# open_api_key=os.getenv("OPEN_API_KEY")
pinecone_api_key=os.getenv("PINECONE_API_KEY")
pinecone_index_name=os.getenv("PINECONE_INDEX_NAME")


DATA_PATH = glob("rag/processed_pdfs/*.pdf")


def main():

    # the ArgumentParser helps to handle cmd line inputs
    parser = argparse.ArgumentParser()

    # create or update the data store
    documents = load_documents()
    print(f"First document content preview: {documents[0].page_content[:500] if documents else 'No documents loaded'}")

    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    print(f"First chunk content preview: {chunks[0].page_content[:500] if chunks else 'No chunks created'}")

    add_to_pinecone(chunks)

def load_documents():
    all_docs = []
    for path in DATA_PATH:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
    print(f"Loaded {len(all_docs)} pages from {len(DATA_PATH)} PDFs")
    return all_docs

def split_documents(documents: list [Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    # embeddings = OpenAIEmbeddings(open_api_key=_OPEN_API_KEY, model="text-embedding-3-large")
    # text_splitter = SemanticChunker(embeddings=embeddings,
    #                                 breakpoint_threshold_type="gradient",
    #                                 breakpoint_threshold_amount=0.8
    )
    return text_splitter.split_documents(documents)




def add_to_pinecone(chunks: list[Document]):

    # initialize pinecone client and index
    pc = PineconeClient(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)

    # load the current database
    db = PineconeVectorStore(
        index=index,
        embedding=get_embedding_function(),
        text_key="text",  # or "page_content" depending on how you chunk
    )
    
    # calculate page ids
    chunks_with_ids = calculate_chunk_ids(chunks)

    new_chunk_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]

    print(f"Upserting {len(new_chunk_ids)} chunks to Pinecone index...")
    db.add_documents(documents=chunks_with_ids, ids=new_chunk_ids)
    print("Documents uploaded to Pinecone successfully.")

def calculate_chunk_ids(chunks):
    # creates ids like rag/rag_data/mentor_canada_resources/rag/rag_data
    # /mentor_canada_resources/1. SRDC. MENTOR.Final Report - Youth Results_FINAL - Copy.pdf:6:2"
    # page source : page number : chunk index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # if the page id is the same as the last one, increment index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
                current_chunk_index = 0
        
        # calculate the chunk id
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # add to the page metadata
        chunk.metadata["id"] = chunk_id

    return chunks


if __name__ == "__main__":
    main()