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

import yaml

# Some basic tasks to achieve
# 1. Summarization 2. Topic Extraction and suggestions 3. Social media posts
### Notes: 
# 1. Dynamic Chunk Size, auto setup chunk size based on the len of characters (so user does not need to input themselves)
# 2. First check the input token/character size from the user, if it is small enough and within the API token limit, then do not need to use 3rd Party framework like LangChain
#   2.1 Save/Cache the results from direct API call and use it for later user QA or other tasks (wihtou relying on 3rd party API)
#   2.2 Prompt Templates need to be prepared for direct API call to achieve the same tasks as LangChian
# 3. Need to check whehter load_summarize_chain can specify token_limit for the final combined summary

class Summarization:
    '''
    Summarization tasks for user input podcast content with LangChain. Use this class when user input token size out of model limit
    '''
    # Define LLMs
    llm3 = ChatOpenAI(temperature=0,
                  openai_api_key=os.getenv('OPENAI_API_KEY'),
                  model_name="gpt-3.5-turbo-16k",
                  request_timeout = 180
                )

    llm4 = ChatOpenAI(temperature=0,
                  openai_api_key=os.getenv('OPENAI_API_KEY'),
                  model_name="gpt-4",
                  request_timeout = 180
                 )
    # embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    # Here using a fixed template for now, but can add dynamics templates with user-instructions inputs then combine both to better fit user needs
    def __init__(self, vectordb, chunk_size, chunk_overlap, transcript_path = None, transcript_string = None, prompt_book_path = "./prompts.yml"): # Change the prompt_book_path to pkg.filename after packaging?
        if transcript_path is None and transcript_string is None:
            raise ValueError("At least one transcript input must be provided")
        
        self.transcript = None
        if transcript_path:
            with open(transcript_path) as f:
                self.transcript = f.read()
        elif transcript_string:
            self.transcript = transcript_string
        
        with open(prompt_book_path, 'r') as file:
            self.prompt_book = yaml.safe_load(file)
        # self.chunk_size = chunk_size ## Maybe need to setup a chunck size range for different length of texts to make it dynamic. e.g for 10000 words or 30000 characters -> 1000 chunk size ... 
        # self.chunk_overlap = chunk_overlap ## Same as above 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap, length_function = len)
        self.docs = text_splitter.create_documents([self.transcript])

        # Docs Embedding into vectorstore
        self.vector_db = vectordb
        # self.vector_db = FAISS.from_documents(self.docs, self.embeddings) # Remember to keep proper chunks for token limitation
        self.topics_structured = []
        self.expanded_topics_sum = ""

    def save_to_file(self, content, output_path: str, output_file_name: str):
        file_path = os.path.join(output_path, output_file_name)
        
        # Write the content to the file
        with open(file_path, 'w') as file:
            file.write(content)
        
        print(f"Summarized content saved to: {file_path}")
        return None

    def show_result(self, result):
        print("Here is the reuslt: ", result) 
        return None

    def summarize_article(self):
        # Show portions of transcript divided
        print(f"The transcript has been chuncked into {len(self.docs)} portions")
        # Templates setup
        # This template now is only for podcast interviews which is very limited, need to update it into a more scalable one for different contents
        map_template = self.prompt_book['summarize-article']['mapreduce-template']
        system_message_map_template = SystemMessagePromptTemplate.from_template(map_template)
        human_template = "Transcript: {text}"
        human_message_template  = HumanMessagePromptTemplate.from_template(human_template)
        map_prompt_template = ChatPromptTemplate.from_messages(messages=[system_message_map_template, human_message_template])

        combined_template = self.prompt_book['summarize-article']['combination-template']
        system_message_combined_template = SystemMessagePromptTemplate.from_template(combined_template)
        combined_prompt_template = ChatPromptTemplate.from_messages(messages=[system_message_combined_template, human_message_template])

        # Load Chain
        chain = load_summarize_chain(self.llm3, chain_type="map_reduce", map_prompt=map_prompt_template, combine_prompt=combined_prompt_template)

        summarized_article = chain.run({"input_documents": self.docs})

        return summarized_article
    
    def summaraize_topics(self):
        # Portions of transcript divided
        print(f"The transcript has been chuncked into {len(self.docs)} portions")
        # Templates setup
        map_template = self.prompt_book['summarize-topics']['mapreduce-template']
        system_message_map_template = SystemMessagePromptTemplate.from_template(map_template)
        human_template = "Transcript: {text}"
        human_message_template  = HumanMessagePromptTemplate.from_template(human_template)
        map_prompt_template = ChatPromptTemplate.from_messages(messages=[system_message_map_template, human_message_template])

        combined_template = self.prompt_book['summarize-topics']['combination-template']
        system_message_combined_template = SystemMessagePromptTemplate.from_template(combined_template)
        combined_prompt_template = ChatPromptTemplate.from_messages(messages=[system_message_combined_template, human_message_template])

        # Load Chain
        chain = load_summarize_chain(self.llm3, chain_type="map_reduce", map_prompt=map_prompt_template, combine_prompt=combined_prompt_template)

        summarized_topics = chain.run({"input_documents": self.docs})
        
        function_call_schema = {
            "properties": {
                "topic_name": {
                    "type": "string",
                    "description": "The topic title listed"
                },
                "detail": {
                    "type": "string",
                    "description": "The detailed description of the topic"
                }
            },
            "required": ["topic_name", "detail"],
        }
        chain = create_extraction_chain(function_call_schema, self.llm3)
        self.topics_structured = chain.run(summarized_topics)

        return self.topics_structured

    def get_chapters_from_topics(self):
        system_template = """
        What is the first timestamp when the speakers started talking about a topic the user gives?
        Only respond with the timestamp, nothing else. Example: 0:18:24 or 300.18 (the timestamp format might differ so use your judgement to take out the timestamp)
        ----------------
        {context}"""
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)

        qa = RetrievalQA.from_chain_type(llm=self.llm3, chain_type="stuff", retriever=self.vector_db.as_retriever(k=4), chain_type_kwargs = {'prompt': CHAT_PROMPT})
        
        # Constructing chapters
        topic_timestamps = []
        for topic in self.topics_structured:
            timestamp = qa.run(f"{topic['topic_name']} - {topic['detail']}")
            topic_timestamps.append(f"{timestamp} - {topic['topic_name']}")

        sorted_timestamps = sorted(topic_timestamps, key=lambda x: float(x.split(' ')[0]))
        print ("\n".join(sorted(topic_timestamps)))
        print(topic_timestamps)
        print(sorted_timestamps)

        return sorted_timestamps
    
    def expanded_topics(self):
        system_template = """
        You will be given text from a podcast transcript which contains many topics.
        You goal is to write a summary (5 sentences or less) about a topic the user chooses
        Do not respond with information that isn't relevant to the topic that the user gives you
        ----------------
        {context}"""

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
        CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)
        qa = RetrievalQA.from_chain_type(llm=self.llm3, chain_type="stuff", retriever=self.vector_db.as_retriever(k=4),chain_type_kwargs = {'prompt': CHAT_PROMPT})

        for topic in self.topics_structured:
            query = f"""
                {topic['topic_name']}: {topic['detail']}
            """
            expanded_topic = qa.run(query)
            self.expanded_topics_sum += f"\n{topic['topic_name']}: {topic['detail']}\n" + f"{expanded_topic}\n"
        
        return self.expanded_topics_sum

    def youtube_title_description(self):
        # Templates setup
        # This template now is only for podcast interviews which is very limited, need to update it into a more scalable one for different contents
        map_template = self.prompt_book['youtube-title-description']['mapreduce-template']
        system_message_map_template = SystemMessagePromptTemplate.from_template(map_template)
        human_template = "Transcript: {text}"
        human_message_template  = HumanMessagePromptTemplate.from_template(human_template)
        map_prompt_template = ChatPromptTemplate.from_messages(messages=[system_message_map_template, human_message_template])

        combined_template = self.prompt_book['youtube-title-description']['combination-template']
        system_message_combined_template = SystemMessagePromptTemplate.from_template(combined_template)
        combined_prompt_template = ChatPromptTemplate.from_messages(messages=[system_message_combined_template, human_message_template])

        # Load Chain
        chain = load_summarize_chain(self.llm3, chain_type="map_reduce", map_prompt=map_prompt_template, combine_prompt=combined_prompt_template)

        description = chain.run({"input_documents": self.docs})

        return description
    
    def social_media_posts(self):
        map_template = self.prompt_book['social-media-posts']['mapreduce-template']
        system_message_map_template = SystemMessagePromptTemplate.from_template(map_template)
        human_template = "Transcript: {text}"
        human_message_template  = HumanMessagePromptTemplate.from_template(human_template)
        map_prompt_template = ChatPromptTemplate.from_messages(messages=[system_message_map_template, human_message_template])

        combined_template = self.prompt_book['social-media-posts']['combination-template']
        system_message_combined_template = SystemMessagePromptTemplate.from_template(combined_template)
        combined_prompt_template = ChatPromptTemplate.from_messages(messages=[system_message_combined_template, human_message_template])

        # Load Chain
        chain = load_summarize_chain(self.llm3, chain_type="map_reduce", map_prompt=map_prompt_template, combine_prompt=combined_prompt_template)

        posts = chain.run({"input_documents": self.docs})

        return posts


# Functions


if __name__ == '__main__':
    # Define inputs
    will_transcript_path = "../Transcripts/output_William_corrected.txt"
    chunk_size = 10000
    chunk_overlap = 2000
    file_output_path = "../Transcripts/"
    article_file_name = "William_sumamrized_article.txt"
    topics_file_name = "William_summarized_topics.txt"
    # Summarize
    sum = Summarization(transcript_path=will_transcript_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    article = sum.summarize_article()
    # topics = sum.summaraize_topics()
    # Save the summarized article into a text file
    sum_article = sum.save_to_file(article, file_output_path, article_file_name)
    # sum_topics = sum.save_to_file(topics, file_output_path, topics_file_name)
    sum.show_result(article)
