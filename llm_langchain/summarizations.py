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
    Summarization tasks for user input podcast content
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
    def __init__(self, vectordb, chunk_size, chunk_overlap, transcript_path = None, transcript_variable = None, prompt_book_path = "./prompts.yml"): # Change the prompt_book_path to pkg.filename after packaging?
        if transcript_path is None and transcript_variable is None:
            raise ValueError("At least one transcript input must be provided")
        
        self.transcript = None
        if transcript_path:
            with open(transcript_path) as f:
                self.transcript = f.read()
        elif transcript_variable:
            self.transcript = transcript_variable
        
        with open(prompt_book_path, 'r') as file:
            self.prompt_book = yaml.safe_load(file)
        # self.chunk_size = chunk_size ## Maybe need to setup a chunck size range for different length of texts to make it dynamic. e.g for 10000 words or 30000 characters -> 1000 chunk size ... 
        # self.chunk_overlap = chunk_overlap ## Same as above 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap, length_function = len)
        self.docs = text_splitter.create_documents([self.transcript])

        # Docs Embedding into vectorstore
        self.vector_db = vectordb
        # self.vector_db = FAISS.from_documents(self.docs, self.embeddings) # Remember to keep proper chunks for token limitation

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
        map_template = """
        You are a helpful assistant that are very good at summarizing a long podcast interview into a blog post style article based on the interview transcript.
        You are given a podcast transcript that contains the conversations of podcast interview with speaker names, timestamps etc.
        Your goal is to summarize a podcast interview into a blog post article with clear topics and highlights of conversations between the hosts and guest during the interview.
        The summarized article should follow a fun, engaging, and educational style that is easy to read and understand.
        The summarized article also has to be well-structured with a concise intro, body, and conclusion and easy to read. The article should be in a similar style of a great article in a famous magazine (e.g. The New Yorker).
        Here is an exmaple of the structure of the summarized article (You do not need to follow this if you have a better idea of how to write a better article):
            - Title: e.g. "The Evolution of AI Technology and How to Navigate it" (come up with a great title for that can best represent the interview)
            - Intro: introduce what the interview is about and what the interviewers are talking about.
            - Body: describe the topics and highlights of the interview and quotes from the interview if necessary.
            - Conclusion: conclude the interview that can make readers feel inspired
        Here are some rules you need to follow:
        - Do not mention anything that is not coherent with the interview.
        - Be creative, do not just repeat what were said in the interview. Your goal is to summarize the article that can attract thousands of readers to read and learn from it.
        - Do not use any inappropriate words.
        - Do not use any offensive language.
        - Do not use any profane words.
        - Do not write out "Intro", "Body", and "Conclusion" in the article, The example above is just to instruct you to implicitly follow this structure. Write the article as natrually as you can. 
        - Be sure to include all the key and important topcis in the transcript.
        Remember:
        The summarized article should have a proper length, not too long or too short. Use your common sense and refer to the article length that usually posted on blogging platform such as Medium or Substack.
        """
        system_message_map_template = SystemMessagePromptTemplate.from_template(map_template)
        human_template = "Transcript: {text}"
        human_message_template  = HumanMessagePromptTemplate.from_template(human_template)
        map_prompt_template = ChatPromptTemplate.from_messages(messages=[system_message_map_template, human_message_template])

        combined_template = """
        You are a creative and professional journalist that are very talented at writing. You are very good at summarizing interviews transcript into a fruitful article for readers.
        - You will be given a summarized article of a podcast interview, and the article should be having a structure from intro -> body -> conclusion.
        - Please come up with a catchy title for the article and add it at the beginning of the article.
        - Proof read this article, and revise and polish with better writing style and grammars.
        - Make the article shorter and concise if needed. 
        - Remove dupicate or unnecessary sentences, components or elements in the article if you found any.
        The resulting article after your revision should be like an article that can be published on famous magazine such as Time or The New York, that best summarize the content of the podcast interview and attract readers.
        """
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
        map_template = """
        You are a helpful assistant that are very good at summarizing podcast interview transcripts and organize the key information from the interview.
        You are given a podcast interview transcript that contains the conversation between the hosts and guest during the interview.
        You goal is to, based on your experience and knowledge, extract key topics discussed in the interview, write a short description for each topic, and include the start and end timestamps that covers the period of conversations you summarized for that topic.
        
        You can refer to the format example below (Note: This is just an example to demonstrate the format, you can ignore the content. The start and end timestamps might be in different formats for different transcript, you should follow what presented in the transcript):
        - NFT Business Model: John and Dan discussed he business ideas and models behind NFT, and the future of how NFT can be utilized for creating more values for... | Start Time: 0.00 End Time: 402.21
        - Building Open Source Community: Mike talked about his experience with contributing part time on multiple famous open source projects such as... | Start Time: 402.21 End Time: 653.72
        ...

        Here are some of the things that do not do:
        - Do not mention anything that is not coherent with the interview.
        - Do not extract any topics or create any topics that are not in the transcript.
        - Do not use any inappropriate words.
        - Do not create any duplicate topics.
        Please strictly follow the instruction and format example I provided to you. Please be descriptive and concise and list out all of the topics with bullet points in the format I showed you.
        """
        system_message_map_template = SystemMessagePromptTemplate.from_template(map_template)
        human_template = "Transcript: {text}"
        human_message_template  = HumanMessagePromptTemplate.from_template(human_template)
        map_prompt_template = ChatPromptTemplate.from_messages(messages=[system_message_map_template, human_message_template])

        combined_template = """
        You are a helpful assistant that are very good at summarizing podcast interview transcripts and extract key topics from the interview.
        - You will be given a list of topics and descriptions of the topics with timestamps that you extracted from the interview.
        - Your goal is to concisely extract the key topics discussed in the interview transcript, and write a short description for each topci you extract, and also include the start and end timestapms that covers the period of conversations you summarized for that topic.
        - Deduplicate any topics that you see.
        - Only extract topics from the interview transcript. Do not use the examples.

        The example topic extraction output is as following:
        - NFT Business Model: John and Dan discussed he business ideas and models behind NFT, and the future of how NFT can be utilized for creating more values for... | Start Time: 0.00 End Time: 402.21
        - Building Open Source Community: Mike talked about his experience with contributing part time on multiple famous open source projects such as... | Start Time: 402.21 End Time: 653.72
        ...
        """
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
        topics_structured = chain.run(summarized_topics)

        return topics_structured

    def get_chapters(self, structured_topcis):
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
        for topic in structured_topcis:
            timestamp = qa.run(f"{topic['topic_name']} - {topic['detail']}")
            topic_timestamps.append(f"{timestamp} - {topic['topic_name']}")

        sorted_timestamps = sorted(topic_timestamps, key=lambda x: float(x.split(' ')[0]))
        print ("\n".join(sorted(topic_timestamps)))
        print(topic_timestamps)
        print(sorted_timestamps)

        return sorted_timestamps
    
    def youtube_title_description(self):
        # Show portions of transcript divided
        print(f"The transcript has been chuncked into {len(self.docs)} portions")
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
    transcript_path = "../Transcripts/output_William_corrected.txt"
    chunk_size = 10000
    chunk_overlap = 2000
    file_output_path = "../Transcripts/"
    article_file_name = "William_sumamrized_article.txt"
    topics_file_name = "William_summarized_topics.txt"
    # Summarize
    sum = Summarization(transcript_path, chunk_size, chunk_overlap)
    article = sum.summarize_article()
    # topics = sum.summaraize_topics()
    # Save the summarized article into a text file
    sum_article = sum.save_to_file(article, file_output_path, article_file_name)
    # sum_topics = sum.save_to_file(topics, file_output_path, topics_file_name)
    sum.show_result(article)




