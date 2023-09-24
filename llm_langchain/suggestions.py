import os
# Import necessary modules for LangChain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import create_extraction_chain

# Vector Database
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

# Chat Prompt templates for dynamic values
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from llm_langchain import summarizations

class EditingSuggestion:
    '''
    Provide edit suggestions based on user input podcast transcript, provide start/end timestamps for each clip with a summary and explanation of that cut, split the transcript with the given start/end timestamp and save it somewhere for later use
    '''
    # Inherit from Summarization class??? Figure out the relationship here
