import streamlit as st
import sys
from llm_direct.llm_direct import LLM_Summarization
from llm_langchain.init_vectordb import init_vectordb
from llm_langchain.summarizations import Summarization
from youtube_transcript.functions import get_youtube_transcript
from io import BytesIO
from pytube import YouTube

try:
    st.title("Sumamrize any youtube video and extract detailed insights")
    st.markdown('With the youtube url you entered, we will process the video and provide you with detailed summaries')
    st.markdown('1. Writing the youtube video content into a beautiful blog article')
    st.markdown('2. Extracting the main topics from the youtube video')
    
    youtube_url = st.text_input('Enter YouTube Video URL \U0001F447', 'youtube.com/watch?...')

    yt = YouTube(youtube_url)

    video_title = yt.title
    video_thumbnail = yt.thumbnail_url

    st.write(f"Video Title: {video_title}")
    st.image(video_thumbnail)



except:
    st.write("Sorry there was an error with the app =(")