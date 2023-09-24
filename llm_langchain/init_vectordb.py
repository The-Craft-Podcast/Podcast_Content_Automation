import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

def init_vectordb(transcript_path, chunk_size, chunk_overlap):
    # The input chunk_size and chunk_overlap should be determined while initializing the vector store app front end
    # Initialize FAISS database
    with open(transcript_path) as f:
        transcript = f.read()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap, length_function = len)
    docs = text_splitter.create_documents([transcript])
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    vectordb = FAISS.from_documents(docs, embeddings)

    return vectordb