import streamlit as st
import sys
from llm_direct.llm_direct import LLM_Summarization
from llm_langchain.init_vectordb import init_vectordb
from llm_langchain.summarizations import Summarization

try:
    st.title("Welcome to our Gen AI Summarizer Center")

    uploaded_audio_file = st.file_uploader("Choose an audio file")
    uploaded_text_file = st.file_uploader("Choose a text file (.txt/.pdf/)")


except:
    st.write("Sorry there was an error with the app =(")